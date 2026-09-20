"""Stage C: per-box local refinement agent.

New-flow contract (user-directed architecture):
  global nadir 2D grounding (ground.py) -> per-box local refine, where
  the FRONT view's mask surface guides how the seed box SPLITS (each
  visually distinct cabinet its own along-row span; the seed's yaw /
  height / depth are trusted), and the SIDE view (the profile along the
  row, where an open door sticks out beyond the body) corrects each
  piece's thickness; plus the rack type-confirm guard.

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

# grounding labels that already answer the type-confirm question: the
# local-grounding categories are exactly the equipment classes the
# type guard accepts
_EQUIP_TOKENS = ("rack", "cabinet", "air-con", "air con", "aircon",
                 "air conditioning", "ac unit", "crac", "pdu")


def _is_equipment_label(label) -> bool:
    return any(t in str(label or "").lower() for t in _EQUIP_TOKENS)


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
        # where, local masks refine edges -- and the local grounding
        # itself splits joined rows whose cabinets differ (the former
        # split_stage's job, now decided on the same SAM evidence).
        self._local_mask_refine(scene, report)
        # 2. confidence tagging from point support. The type-confirm LOW
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
        """Local per-box VLM pass: SAM split-correction + type confirmation.

        Both questions share ONE render_local_views call per box (front
        + side views). The FRONT view alone votes on how the seed box
        SPLITS (its mask surface's along-row spans; yaw / height /
        depth stay seed-trusted), and the SIDE view (the profile along
        the row, where an open door sticks out beyond the body) corrects
        each piece's THICKNESS. Per box the VLM cost is 2
        local-grounding calls + 1 type-confirm; the type-confirm is
        SKIPPED when the grounding already labelled every accepted
        instance as equipment with a strong score. The type
        confirmation runs even when SAM is not configured -- it only
        needs the local view and the VLM.
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
            # ---- type gate for CLUSTER-proposed boxes (BEFORE SAM) ----
            # The recall-first net proposes every density clump with NO
            # VLM label behind it; a pillar / UPS / junk block would
            # otherwise pay the FULL SAM refinement (local grounding
            # calls + masks + refit) before the type-confirm marks it
            # LOW. Ask the one cheap yes/no FIRST: 'not equipment' ->
            # LOW + skip the refinement entirely. VLM-grounded boxes
            # keep SAM-first (their grounding already said 'device';
            # the post-SAM skip_confirm logic covers them). A 'no'
            # NEVER deletes -- LOW + human review, as always.
            confirmed_rack: bool | None = None
            if (old.meta.get("cluster")
                    and (sam.available or self.judge.backend != "mock")):
                try:
                    r = confirm_device_type(self.judge, old, views)
                except Exception as e:
                    print(f"[type-confirm] {old.box_id[:6]} failed "
                          f"({type(e).__name__}: {e})")
                    r = None
                if r is not None:
                    conf_audits.append({"box_id": old.box_id, **r})
                    confirmed_rack = bool(r["is_rack"])
                    if not confirmed_rack:
                        old.confidence = Confidence.LOW
                        old.meta["type_suspect"] = True
                        report.unresolved.append(
                            {"issue": {"type": "not_a_rack",
                                       "box_id": old.box_id,
                                       "confidence": r["confidence"]},
                             "ok": False})
                        print(f"[type-confirm] {old.box_id[:6]} NOT a rack "
                              f"(conf {r['confidence']:.2f}) -> LOW, "
                              f"SAM refinement skipped")
                        continue
            # ---- SAM mask refinement (multi-instance), runs FIRST: its
            # grounding result also decides whether the type-confirm
            # question is worth asking ----
            instances = []
            if sam.available:
                try:
                    instances, audit = refine_box(scene, old, self.judge,
                                                 sam, self.out_dir,
                                                 views=views)
                except Exception as e:
                    print(f"[mask-refine] {old.box_id[:6]} failed "
                          f"({type(e).__name__}: {e}) -> keep")
                    audits.append({"box_id": old.box_id, "accepted": False,
                                   "reason": f"{type(e).__name__}: {e}"})
                else:
                    audits.append(audit)
                    if not instances:
                        print(f"[mask-refine] {old.box_id[:6]} no valid "
                              f"mask -> keep")
            # ---- type-level guard (no SAM needed) ----
            # Grounding guards reject hallucinated EMPTY regions, but a
            # real structure mislabelled equipment (pillar / UPS / wall
            # segment) passes them all: it has points, height and a good
            # SAM mask. One VLM yes/no on the front view (rack / IT
            # cabinet / AC unit = equipment; the rest = clutter) closes
            # that gap. A 'no' NEVER deletes -- it marks LOW confidence
            # and surfaces the box for human review (false-positive
            # deletion is the dangerous direction).
            # SKIP when the local grounding already answered it: every
            # accepted instance is equipment-labelled with a strong
            # score (>= 0.6) -- asking again would be a redundant third
            # VLM call per box. A cluster box the early gate already
            # confirmed (or rejected) is never re-asked either.
            if confirmed_rack:
                pass                    # early gate already recorded it
            else:
                skip_confirm = bool(instances) and all(
                    _is_equipment_label(e["label"]) and e["score"] >= 0.6
                    for e in instances)
                if skip_confirm:
                    conf_audits.append({
                        "box_id": old.box_id, "is_rack": True,
                        "skipped": "local grounding labelled every "
                                   "instance as equipment with "
                                   "score >= 0.6"})
                else:
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
                            print(f"[type-confirm] {old.box_id[:6]} NOT a "
                                  f"rack (conf {r['confidence']:.2f}) "
                                  f"-> LOW + review")
            # ---- adoption: the primary keeps the old identity, extra
            # split instances enter as new boxes ----
            if not instances:
                continue
            # conservative guard: every piece must stay ON the old box
            # (its centre within the seed, padded). Pieces are SPLITS
            # of the seed -- each carries ~1/N of its area -- so an IoU
            # threshold would wrongly reject all but the biggest piece.
            valid = [e["fitted"] for e in instances
                     if old.contains(
                         np.asarray(e["fitted"].center,
                                    dtype=float).reshape(1, 3),
                         margin=0.30)[0]]
            if not valid:
                print(f"[mask-refine] {old.box_id[:6]} rejected: "
                      f"piece centre outside the seed")
                continue
            if len(valid) == 1:
                self._adopt_refit(scene, old, valid[0])
                report.actions_taken.append({
                    "issue_id": "sam_refine", "action": "mask_refine",
                    "params": {"box_id": old.box_id,
                               "score": audit["instances"][0]["score"],
                               "view": audit["instances"][0]["view"],
                               "points": audit["instances"][0]["points"]}})
                print(f"[mask-refine] {old.box_id[:6]} accepted from "
                      f"{audit['instances'][0]['view']} "
                      f"score={audit['instances'][0]['score']}")
            else:
                # the local grounding split a joined row into distinct
                # cabinets: the primary keeps the old identity, the rest
                # enter as new boxes
                import uuid as _uuid
                self._adopt_refit(scene, old, valid[0])
                for b in valid[1:]:
                    b.box_id = _uuid.uuid4().hex[:8]
                    b.source = BoxSource.AGENT_FIX
                    b.row_id = old.row_id
                    scene.boxes.append(b)
                report.actions_taken.append({
                    "issue_id": "sam_refine", "action": "mask_refine_split",
                    "params": {"box_id": old.box_id,
                               "instances": [
                                   {"box_id": b.box_id}
                                   for b in valid]}})
                print(f"[mask-refine] {old.box_id[:6]} split into "
                      f"{len(valid)} instances by local grounding")
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
