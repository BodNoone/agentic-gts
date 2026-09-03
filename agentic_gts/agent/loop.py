"""Stage C: agent repair loop.

For each residual issue after the rule stage, the agent:
  1. gathers evidence (top-down density + box overlay)
  2. asks the VLM judge a discriminative question
  3. selects a discrete action (split / shrink / add / delete / merge / keep)
  4. executes it via geometry tools (which produce exact coordinates)
  5. verifies with rules + (optionally) VLM; accepts or rolls back

Max 2 retries per issue; unresolved issues get flagged LOW confidence
for human review.
"""
from __future__ import annotations

import copy
import math
from dataclasses import dataclass, field

import numpy as np

from agentic_gts.core.models import (
    BoxSource,
    Confidence,
    Issue,
    IssueType,
    OrientedBox,
    Scene,
)
from agentic_gts.agent.judge import Verdict, VLMJudge
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
    def __init__(self, judge: VLMJudge | None = None, max_retries: int = 2,
                 opts: dict | None = None, out_dir: str | None = None):
        self.judge = judge or VLMJudge(backend="mock")
        self.max_retries = max_retries
        self.opts = opts or {}
        self.out_dir = out_dir

    # ---------------- issue detection ----------------
    def detect_issues(self, scene: Scene) -> list[Issue]:
        """Audit current boxes and produce an issue list."""
        issues: list[Issue] = []
        width_unit = float(self.opts.get("width_unit", 0.6))
        for b in scene.boxes:
            sup = geo.support_fraction(scene, b)
            if sup < 0.12:
                issues.append(Issue(IssueType.FALSE_POSITIVE, [b.box_id],
                                    self._region(b), detail=f"support={sup:.2f}",
                                    severity=0.8))
                continue
            n_clusters, dom, _ = geo.center_field_clusters(scene, b)
            if n_clusters >= 2 and b.size[0] > width_unit * 1.5:
                issues.append(Issue(IssueType.MERGED_ROW, [b.box_id],
                                    self._region(b),
                                    detail=f"clusters={n_clusters}", severity=0.7))
        # overlapping boxes
        for i, a in enumerate(scene.boxes):
            for b in scene.boxes[i + 1:]:
                if abs(a.center[0] - b.center[0]) > 3 or abs(a.center[1] - b.center[1]) > 3:
                    continue
                iou = a.iou_2d(b)
                if iou > 0.25:
                    issues.append(Issue(IssueType.OVERLAP, [a.box_id, b.box_id],
                                        self._region(a), detail=f"iou={iou:.2f}",
                                        severity=0.6))
        return issues

    # ---------------- god-view global audit ----------------
    def godview_pass(self, scene: Scene) -> list[Issue]:
        """One global VLM call over the whole scene (top-down, all boxes).

        Returns FALSE_POSITIVE issues for boxes the VLM finds globally
        suspicious (e.g. floating in an aisle, off every row). Each flagged
        box then goes through the normal per-box local adjudication before
        any deletion happens -- the god-view only nominates, it never
        executes. Mock/failure -> no issues (rule-detected ones remain).
        """
        # persist the exact image the VLM sees: it is the single most
        # useful artifact when auditing why the agent flagged (or missed)
        # a box -- no guessing from logs
        if self.out_dir:
            try:
                from agentic_gts.agent.judge import render_godview_png
                import os as _os
                _os.makedirs(self.out_dir, exist_ok=True)
                path = _os.path.join(self.out_dir, "godview.png")
                with open(path, "wb") as f:
                    f.write(render_godview_png(scene.points, scene.boxes))
                print(f"[diag][C] godview render -> {path}")
            except Exception as e:
                print(f"[diag][C] godview render save failed ({type(e).__name__})")
        try:
            flagged = self.judge.adjudicate_godview(scene, scene.boxes)
        except Exception as e:
            print(f"[diag][C] godview pass error ({type(e).__name__}) -> skipped")
            return []
        issues: list[Issue] = []
        for f in flagged:
            b = scene.boxes[f["index"]]
            print(f"[diag][C] godview flagged #{f['index']} "
                  f"@({b.center[0]:.1f},{b.center[1]:.1f}): {f['reason']}")
            issues.append(Issue(IssueType.FALSE_POSITIVE, [b.box_id],
                                self._region(b),
                                detail=f"godview: {f['reason']}", severity=0.7))
        if not flagged:
            print("[diag][C] godview pass: no global suspicions")
        return issues

    # ---------------- repair loop ----------------
    def run(self, scene: Scene) -> AgentReport:
        report = AgentReport()
        issues = self.godview_pass(scene) + self.detect_issues(scene)
        from collections import Counter
        cnt = Counter(i.issue_type.value for i in issues)
        print(f"[diag][C] issues detected: {dict(cnt) if cnt else 'none'}")
        for issue in issues:
            ok = self._handle_issue(scene, issue, report)
            entry = {"issue": issue.to_dict(), "ok": ok}
            (report.resolved if ok else report.unresolved).append(entry)
        # final edge refinement: snap every box to its point support
        self._refine_edges(scene)
        # confidence tagging
        for b in scene.boxes:
            sup = geo.support_fraction(scene, b)
            if sup > 0.3 and b.source != BoxSource.ROW_COMPLETION:
                b.confidence = Confidence.HIGH
            elif sup > 0.15:
                b.confidence = Confidence.MID
            else:
                b.confidence = Confidence.LOW
        return report

    def _refine_edges(self, scene: Scene) -> None:
        """Snap box edges to point support for the final layout accuracy.

        Expansion is tiny (2cm): adjacent racks are only mm apart, so any
        larger search window would absorb the neighbour's surface points.
        """
        refined: list[OrientedBox] = []
        for b in scene.boxes:
            refit = geo.fit_box_to_points(scene, b.center[:2],
                                          (b.size[0] + 0.02, b.size[1] + 0.02, b.size[2] + 0.02),
                                          b.yaw)
            if refit is not None and refit.iou_2d(b) > 0.3:
                refit.box_id = b.box_id
                refit.device_type = b.device_type
                refit.source = b.source
                refit.confidence = b.confidence
                refit.row_id = b.row_id
                refit.meta = b.meta
                refined.append(refit)
            else:
                refined.append(b)
        scene.boxes = refined

    def _handle_issue(self, scene: Scene, issue: Issue, report: AgentReport) -> bool:
        for attempt in range(self.max_retries + 1):
            snapshot = copy.deepcopy(scene.boxes)
            action = self._decide(scene, issue)
            applied = self._execute(scene, issue, action)
            if not applied:
                scene.boxes = snapshot
                continue
            if self._verify(scene, issue):
                report.actions_taken.append({
                    "issue_id": issue.issue_id, "action": action.action,
                    "params": action.params, "attempt": attempt,
                })
                return True
            scene.boxes = snapshot  # rollback
        # leave as-is; mark involved boxes low confidence
        for bid in issue.box_ids:
            b = scene.get_box(bid)
            if b:
                b.confidence = Confidence.LOW
        return False

    # ---------------- decision ----------------
    def _decide(self, scene: Scene, issue: Issue) -> Verdict:
        box = scene.get_box(issue.box_ids[0]) if issue.box_ids else None
        if issue.issue_type == IssueType.MERGED_ROW and box is not None:
            n_clusters, dom, _ = geo.center_field_clusters(scene, box)
            verdict = self.judge.adjudicate_box(
                scene, box,
                question=("Does the red box contain one rack or multiple racks? "
                          "If multiple, they should be split."),
                options=["one rack", "multiple racks"],
            )
            choice = (verdict.params or {}).get("choice", "")
            if verdict.action == "split" or choice == "multiple racks" or n_clusters >= 2:
                width_unit = float(self.opts.get("width_unit", 0.6))
                n = max(2, int(round(box.size[0] / width_unit)))
                return Verdict(action="split", params={"n": n, "width_unit": width_unit})
            return Verdict(action="keep")
        if issue.issue_type == IssueType.FALSE_POSITIVE and box is not None:
            verdict = self.judge.adjudicate_box(
                scene, box,
                question="Is there truly a device at the red box, or is it empty space?",
                options=["real device", "empty space"],
            )
            choice = (verdict.params or {}).get("choice", "")
            sup = geo.support_fraction(scene, box)
            if choice == "empty space" or sup < 0.08:
                return Verdict(action="delete")
            return Verdict(action="shrink")
        if issue.issue_type == IssueType.OVERLAP and len(issue.box_ids) >= 2:
            return Verdict(action="resolve_overlap")
        return Verdict(action="keep")

    # ---------------- execution ----------------
    def _execute(self, scene: Scene, issue: Issue, verdict: Verdict) -> bool:
        if verdict.action == "keep":
            return True
        if verdict.action == "delete":
            return all(scene.remove_box(bid) for bid in issue.box_ids)
        if verdict.action == "split":
            box = scene.get_box(issue.box_ids[0])
            if box is None:
                return False
            params = verdict.params or {}
            subs = geo.split_box(scene, box, int(params.get("n", 2)),
                                 params.get("width_unit"))
            if len(subs) < 2:
                return False
            scene.remove_box(box.box_id)
            for s in subs:
                s.source = BoxSource.AGENT_FIX
                scene.boxes.append(s)
            return True
        if verdict.action == "shrink":
            box = scene.get_box(issue.box_ids[0])
            if box is None:
                return False
            refit = geo.fit_box_to_points(scene, box.center[:2], box.size, box.yaw)
            if refit is None:
                return False
            refit.source = BoxSource.AGENT_FIX
            refit.row_id = box.row_id
            scene.remove_box(box.box_id)
            scene.boxes.append(refit)
            return True
        if verdict.action == "resolve_overlap":
            a = scene.get_box(issue.box_ids[0])
            b = scene.get_box(issue.box_ids[1])
            if a is None or b is None:
                return False
            # keep the one with better support, refit the other
            sa, sb = geo.support_fraction(scene, a), geo.support_fraction(scene, b)
            loser = b if sa >= sb else a
            refit = geo.fit_box_to_points(scene, loser.center[:2], loser.size, loser.yaw)
            scene.remove_box(loser.box_id)
            if refit is not None:
                winner = a if loser is b else b
                if refit.iou_2d(winner) < 0.2:
                    refit.source = BoxSource.AGENT_FIX
                    scene.boxes.append(refit)
            return True
        return False

    # ---------------- verification ----------------
    def _verify(self, scene: Scene, issue: Issue) -> bool:
        """Rule-channel verification after a fix. VLM channel optional."""
        width_unit = float(self.opts.get("width_unit", 0.6))
        # all remaining involved boxes must have decent support and no big overlap
        for bid in issue.box_ids:
            b = scene.get_box(bid)
            if b is None:
                continue  # deleted is fine
            if geo.support_fraction(scene, b) < 0.1:
                return False
        # merged-row: after split every piece should be ~1 unit wide
        if issue.issue_type == IssueType.MERGED_ROW:
            pieces = [b for b in scene.boxes
                      if b.meta.get("split_from") in issue.box_ids or b.box_id in issue.box_ids]
            for p in pieces:
                if p.size[0] > width_unit * 1.8:
                    return False
        # overlap resolved?
        if issue.issue_type == IssueType.OVERLAP:
            boxes = [scene.get_box(bid) for bid in issue.box_ids]
            boxes = [b for b in boxes if b is not None]
            if len(boxes) == 2 and boxes[0].iou_2d(boxes[1]) > 0.25:
                return False
        return True

    @staticmethod
    def _region(b: OrientedBox) -> tuple[float, float, float, float]:
        c = np.asarray(b.center[:2])
        half = max(b.size[0], b.size[1]) / 2 + 0.3
        return (float(c[0] - half), float(c[1] - half),
                float(c[0] + half), float(c[1] + half))
