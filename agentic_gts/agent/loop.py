"""Stage C: per-box local refinement agent.

New-flow contract (user-directed architecture):
  global nadir 2D grounding (ground.py) -> per-box local refine
  (VLM box grounding -> SAM2 mask -> 3DGS backprojection -> metric OBB,
  plus the rack type-confirm guard) -> row split.

There is NO issue/repair loop anymore: the old rule-detected
(MERGED_ROW / FALSE_POSITIVE / OVERLAP / WIDTH_MISFIT) repair rounds,
god-view suspicious nomination and depth completion were the
pre-grounding pipeline's patch passes; grounding + per-box SAM evidence
made them redundant.
"""
from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

from agentic_gts.core.models import (
    BoxSource,
    Confidence,
    OrientedBox,
    Scene,
)
from agentic_gts.agent.judge import VLMJudge
from agentic_gts.tools import geometry as geo


@dataclass
class AgentReport:
    resolved: list[dict] = field(default_factory=list)
    unresolved: list[dict] = field(default_factory=list)
    actions_taken: list[dict] = field(default_factory=list)

    def to_dict(self) -> dict:
        return {
            "resolved": self.resolved,
            "unresolved": self.unresolved,
            "actions_taken": self.actions_taken,
        }


class LayoutAgent:
    def __init__(self, judge: VLMJudge | None = None,
                 opts: dict | None = None, out_dir: str | None = None):
        self.judge = judge or VLMJudge(backend="mock")
        self.opts = opts or {}
        self.out_dir = out_dir

    # ---------------- main flow ----------------
    def run(self, scene: Scene) -> AgentReport:
        report = AgentReport()
        # 1. local mask refinement: Qwen3-VL box-grounds the device in the
        # clean local render, SAM2 segments it with the box prompt, and
        # projected 3DGS centers fit the metric OBB. Runs for BOTH
        # externally supplied and VLM-grounded boxes; grounding finds
        # where, local masks refine edges.
        self._local_mask_refine(scene, report)
        # 2. row split AFTER the local refinement (user-directed order):
        # each grounded region is refined on its own evidence FIRST, then
        # the joined rows are divided into cabinets. Pre-merging / splitting
        # before the refinement decided structure membership before the SAM
        # masks had a vote.
        if self.opts.get("vlm_grounded"):
            from agentic_gts.agent.ground import split_stage
            split_stage(scene, self.judge, self.out_dir)
        # 3. confidence tagging from point support. The type-confirm LOW
        # marks from step 1 are the last word: a well-supported structure
        # the VLM says is NOT equipment must stay LOW for human review.
        for b in scene.boxes:
            if b.meta.get("type_suspect"):
                b.confidence = Confidence.LOW
                continue
            sup = geo.support_fraction(scene, b)
            if sup > 0.3 and b.source != BoxSource.ROW_COMPLETION:
                b.confidence = Confidence.HIGH
            elif sup > 0.15:
                b.confidence = Confidence.MID
            else:
                b.confidence = Confidence.LOW
        return report

    def _local_mask_refine(self, scene: Scene, report: AgentReport) -> None:
        """Local per-box VLM pass: SAM mask refinement + type confirmation.

        Both questions share ONE render_local_views call per box (front
        + side views). The type confirmation runs even when SAM is not
        configured -- it only needs the local view and the VLM.
        """
        from agentic_gts.agent.mask_refine import (SamPredictorAdapter,
                                                    confirm_device_type,
                                                    json_default,
                                                    refine_box,
                                                    render_local_views)
        import json as _json
        import os as _os

        sam = SamPredictorAdapter(
            checkpoint=self.opts.get("sam_checkpoint"),
            model_cfg=self.opts.get("sam_model_cfg"))
        if not sam.available:
            print("[mask-refine] SAM disabled: set --sam-checkpoint or "
                  "SAM_CHECKPOINT; existing boxes kept")
        audits, conf_audits = [], []
        for old in list(scene.boxes):
            if scene.get_box(old.box_id) is None:
                continue
            # one render per box, shared by both questions
            views = (render_local_views(scene, old, self.out_dir)
                     if (sam.available
                         or self.judge.backend != "mock") else [])
            # ---- type-level guard (no SAM needed) ----
            # Grounding guards reject hallucinated EMPTY regions, but a
            # real structure mislabelled equipment (pillar / UPS / wall
            # segment) passes them all: it has points, height and a good
            # SAM mask. One VLM yes/no on the front view (rack / IT
            # cabinet / AC unit = equipment; the rest = clutter) closes
            # that gap. A 'no' NEVER deletes -- it marks LOW confidence
            # and surfaces the box for human review (false-positive
            # deletion is the dangerous direction).
            try:
                r = confirm_device_type(self.judge, old, views)
            except Exception as e:
                print(f"[type-confirm] {old.box_id[:6]} failed "
                      f"({type(e).__name__}: {e})")
                r = None
            if r is not None:
                conf_audits.append({"box_id": old.box_id, **r})
                if not r["is_rack"]:
                    old.confidence = Confidence.LOW
                    old.meta["type_suspect"] = True
                    report.unresolved.append(
                        {"issue": {"type": "not_a_rack",
                                   "box_id": old.box_id,
                                   "confidence": r["confidence"]},
                         "ok": False})
                    print(f"[type-confirm] {old.box_id[:6]} NOT a rack "
                          f"(conf {r['confidence']:.2f}) -> LOW + review")
            # ---- SAM mask refinement ----
            if not sam.available:
                continue
            try:
                new, audit = refine_box(scene, old, self.judge, sam,
                                        self.out_dir, views=views)
            except Exception as e:
                print(f"[mask-refine] {old.box_id[:6]} failed "
                      f"({type(e).__name__}: {e}) -> keep")
                audits.append({"box_id": old.box_id, "accepted": False,
                               "reason": f"{type(e).__name__}: {e}"})
                continue
            audits.append(audit)
            if new is None:
                print(f"[mask-refine] {old.box_id[:6]} no valid mask -> keep")
                continue
            # conservative guard: a local mask may not jump to a neighbour
            if new.iou_2d(old) < 0.2:
                print(f"[mask-refine] {old.box_id[:6]} rejected: IoU<0.2")
                continue
            self._adopt_refit(scene, old, new)
            report.actions_taken.append({
                "issue_id": "sam_refine", "action": "mask_refine",
                "params": {"box_id": old.box_id,
                           "score": audit.get("score"),
                           "view": audit.get("view"),
                           "points": audit.get("points")}})
            print(f"[mask-refine] {old.box_id[:6]} accepted from "
                  f"{audit.get('view')} score={audit.get('score')}")
        if self.out_dir:
            try:
                if audits:
                    with open(_os.path.join(self.out_dir, "mask_refine.json"),
                              "w", encoding="utf-8") as f:
                        _json.dump(audits, f, ensure_ascii=False, indent=2,
                                   default=json_default)
                if conf_audits:
                    with open(_os.path.join(self.out_dir,
                                            "type_confirm.json"),
                              "w", encoding="utf-8") as f:
                        _json.dump(conf_audits, f, ensure_ascii=False,
                                   indent=2, default=json_default)
            except Exception as e:
                print(f"[mask-refine] audit save failed ({type(e).__name__})")

    @staticmethod
    def _adopt_refit(scene: Scene, old: OrientedBox,
                     refit: OrientedBox) -> OrientedBox:
        """Replace `old` with `refit` in the scene, keeping identity/meta."""
        refit.box_id = old.box_id
        refit.device_type = old.device_type
        refit.source = BoxSource.AGENT_FIX
        refit.row_id = old.row_id
        refit.meta = old.meta
        refit.confidence = old.confidence
        scene.remove_box(old.box_id)
        scene.boxes.append(refit)
        return refit
