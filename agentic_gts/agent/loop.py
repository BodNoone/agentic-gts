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


def _points_in_radius(points: np.ndarray, c: tuple[float, float],
                      r: float) -> np.ndarray:
    """2D circular crop of points around (c[0], c[1]) within radius r."""
    if len(points) == 0:
        return points
    d = np.hypot(points[:, 0] - c[0], points[:, 1] - c[1])
    return points[d <= r]


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
        trust = bool(self.opts.get("trust_input_boxes"))
        for b in scene.boxes:
            sup = geo.support_fraction(scene, b)
            # a single-view back-projection fragment has an empty interior but
            # a fully-backed face; rescue it the same way the rule layer does,
            # and for trusted input never flag low interior support alone as a
            # false positive -- the detector vouched for it.
            if sup < 0.12:
                sup = max(sup, geo.face_support_fraction(scene, b))
            if sup < 0.12 and not trust:
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
        # candidate boxes that may be faces of the SAME rack (no intersection,
        # aligned along the row, close together) -- the geometry cannot decide
        # (rules can't be enumerated), so hand to the VLM to reason about.
        for aid, bid in self._find_merge_candidates(scene):
            a, b = scene.get_box(aid), scene.get_box(bid)
            if a is None or b is None:
                continue
            x0 = min(a.center[0], b.center[0])
            y0 = min(a.center[1], b.center[1])
            x1 = max(a.center[0], b.center[0])
            y1 = max(a.center[1], b.center[1])
            issues.append(Issue(IssueType.MERGED_NEIGHBORS, [aid, bid],
                                (x0 - 0.5, y0 - 0.5, x1 + 0.5, y1 + 0.5),
                                detail="adjacent, may be same rack faces",
                                severity=0.65))
        return issues

    def _find_merge_candidates(self, scene: Scene,
                               gap_ratio: float = 2.0) -> list[tuple[str, str]]:
        """Pairs of boxes likely to be faces of ONE rack.

        A single rack's front / back / side surfaces come out as separate
        detector boxes with NO intersection: they overlap along the ROW axis
        (same along-position) but separate along the CROSS axis (front vs back
        face, or two side fragments). This is not enumerable by rules -- the
        VLM decides. We only scan for plausible candidates: same row, near
        along-position, cross-positions close (within a rack footprint).
        """
        yaw = float(scene.meta.get("yaw", 0.0))
        axis = np.array([math.cos(yaw), math.sin(yaw)])
        cross = np.array([-math.sin(yaw), math.cos(yaw)])
        res = []
        for i, a in enumerate(scene.boxes):
            for b in scene.boxes[i + 1:]:
                ca, cb = np.asarray(a.center[:2]), np.asarray(b.center[:2])
                if np.linalg.norm(ca - cb) > 3.5:
                    continue
                along_a, along_b = ca @ axis, cb @ axis
                cross_a, cross_b = ca @ cross, cb @ cross
                # same row: cross positions must be close (< 1.2 rack depth)
                if abs(cross_a - cross_b) > 1.2:
                    continue
                # near same along position (a rack is ~one unit long): centers
                # within ~1.5 rack widths along the row
                if abs(along_a - along_b) > 1.5:
                    continue
                # either the cross gap is small (front/back of ONE rack) or
                # the along gap is small (left/right fragments of ONE rack)
                cross_gap = abs(cross_a - cross_b)
                along_gap = abs(along_a - along_b)
                small_along = min(a.size[0], b.size[0]) * gap_ratio
                small_cross = min(a.size[1], b.size[1]) * gap_ratio
                near = (cross_gap <= small_cross and along_gap <= small_along)
                if near:
                    res.append((a.box_id, b.box_id))
        return res

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
                    f.write(render_godview_png(scene.points, scene.boxes,
                                               gs_ply=scene.meta.get("gs_ply")))
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
        # One readable per-issue diagnostic line so the user can see WHY the
        # agent decided what it did (geometry evidence + VLM/mock verdict).
        self._log_issue(scene, issue)
        before = copy.deepcopy(scene.boxes)
        for attempt in range(self.max_retries + 1):
            snapshot = copy.deepcopy(scene.boxes)
            action = self._decide(scene, issue)
            applied = self._execute(scene, issue, action)
            if not applied:
                print(f"[diag][C]   attempt {attempt}: action '{action.action}' not applied"
                      f" (rollback)")
                scene.boxes = snapshot
                continue
            if self._verify(scene, issue):
                print(f"[diag][C]   attempt {attempt}: '{action.action}' ok -> "
                      f"{len(scene.boxes)} boxes after")
                report.actions_taken.append({
                    "issue_id": issue.issue_id, "action": action.action,
                    "params": action.params, "attempt": attempt,
                })
                self._render_fix(scene, issue, before, out_tag="after")
                return True
            print(f"[diag][C]   attempt {attempt}: '{action.action}' FAILED verify "
                  f"(rollback)")
            scene.boxes = snapshot  # rollback
        # leave as-is; mark involved boxes low confidence
        for bid in issue.box_ids:
            b = scene.get_box(bid)
            if b:
                b.confidence = Confidence.LOW
        self._render_fix(scene, issue, before, out_tag="unresolved")
        return False

    def _log_issue(self, scene: Scene, issue: Issue) -> None:
        """Print the evidence behind an issue decision so it is auditable."""
        b = scene.get_box(issue.box_ids[0]) if issue.box_ids else None
        if b is None:
            print(f"[diag][C] issue {issue.issue_type.value} "
                  f"(no box for {issue.box_ids[:3]}) @ detail={issue.detail[:60]}")
            return
        sup = geo.support_fraction(scene, b)
        sup = max(sup, geo.face_support_fraction(scene, b)) if sup < 0.12 else sup
        if issue.issue_type == IssueType.MERGED_ROW:
            nc, dom, _ = geo.center_field_clusters(scene, b)
            print(f"[diag][C] issue MERGED_ROW  box={b.box_id[:6]} "
                  f"@({b.center[0]:.1f},{b.center[1]:.1f}) "
                  f"size=({b.size[0]:.2f}x{b.size[1]:.2f}x{b.size[2]:.2f}) "
                  f"n_clusters={nc} support={sup:.2f}")
        elif issue.issue_type == IssueType.FALSE_POSITIVE:
            print(f"[diag][C] issue FALSE_POS  box={b.box_id[:6]} "
                  f"@({b.center[0]:.1f},{b.center[1]:.1f}) "
                  f"size=({b.size[0]:.2f}x{b.size[1]:.2f}) support={sup:.2f}")
        else:
            print(f"[diag][C] issue {issue.issue_type.value} box={b.box_id[:6]} "
                  f"@({b.center[0]:.1f},{b.center[1]:.1f}) "
                  f"size=({b.size[0]:.2f}x{b.size[1]:.2f}) detail={issue.detail[:60]}")

    def _render_fix(self, scene: Scene, issue: Issue, before_boxes,
                    out_tag: str = "after") -> None:
        """Render a before/after crop around the issue's boxes so the fix is
        visible, not just logged. Uses the issue region (not get_box, which
        returns None after a split/delete removed the original box).
        Best-effort; never breaks the loop."""
        if not self.out_dir:
            return
        try:
            import os as _os
            from agentic_gts.output.visualize import overlay_topdown
            _os.makedirs(self.out_dir, exist_ok=True)
            # bounding region: prefer the issue region, else a surviving box
            cx = cy = ext = None
            if issue.region:
                x0, y0, x1, y1 = issue.region
                cx, cy = (x0 + x1) / 2.0, (y0 + y1) / 2.0
                ext = max(x1 - x0, y1 - y0) / 2.0 + 0.5
            else:
                surv = [b for b in scene.boxes if b.box_id in issue.box_ids]
                if not surv and before_boxes:
                    surv = [b for b in before_boxes if b.box_id in issue.box_ids]
                if surv:
                    b = surv[0]
                    cx, cy = b.center[0], b.center[1]
                    ext = max(b.size[0], b.size[1]) * 1.5 + 0.5
            if cx is None:
                return
            r_pts = _points_in_radius(scene.points, (cx, cy), ext)
            tmp_before = Scene(points=r_pts, boxes=before_boxes)
            tmp_after = Scene(points=r_pts, boxes=scene.boxes)
            with open(_os.path.join(self.out_dir,
                      f"fix_{issue.issue_type.value}_{out_tag}_before.png"), "wb") as f:
                f.write(overlay_topdown(tmp_before, title=f"{issue.issue_type.value} before"))
            with open(_os.path.join(self.out_dir,
                      f"fix_{issue.issue_type.value}_{out_tag}_after.png"), "wb") as f:
                f.write(overlay_topdown(tmp_after, title=f"{issue.issue_type.value} after"))
            print(f"[diag][C]   fix render -> {self.out_dir}/fix_{issue.issue_type.value}_"
                  f"{out_tag}_[before|after].png")
        except Exception as e:
            print(f"[diag][C]   fix render failed ({type(e).__name__}: {e})")

    # ---------------- decision ----------------
    def _save_local_evidence(self, scene: Scene, box: OrientedBox,
                             issue: Issue) -> None:
        """Save the per-box local crop the VLM adjudicates on (PNG).

        Best-effort: never lets an I/O problem break the repair loop.
        """
        if not self.out_dir:
            return
        try:
            from agentic_gts.agent.judge import render_topdown_image
            import matplotlib
            matplotlib.use("Agg")
            import matplotlib.pyplot as plt
            import os as _os
            _os.makedirs(self.out_dir, exist_ok=True)
            # MUST match what adjudicate_box() feeds the VLM: pass gs_ply so
            # the saved evidence is a true Gaussian render (same image the
            # judge based its verdict on), not a scatter view.
            img = render_topdown_image(scene.points, [box],
                                       gs_ply=scene.meta.get("gs_ply"))
            path = _os.path.join(self.out_dir,
                                 f"evidence_{box.box_id[:8]}.png")
            plt.imsave(path, img)
            print(f"[diag][C] local evidence -> {path} ({issue.detail[:40]})")
        except Exception as e:
            print(f"[diag][C] local evidence save failed ({type(e).__name__})")

    def _decide(self, scene: Scene, issue: Issue) -> Verdict:
        box = scene.get_box(issue.box_ids[0]) if issue.box_ids else None
        if issue.issue_type == IssueType.MERGED_ROW and box is not None:
            n_clusters, dom, _ = geo.center_field_clusters(scene, box)
            # Geometry is the hard signal: a box whose center-field splits
            # into >=2 coherent slabs IS multiple racks -- split it. The VLM
            # is a second opinion only: it can confirm the count / spot a
            # genuinely single tall unit, but it does not veto a clear
            # geometric multi-cluster. Trusted boxes are roughly right; the
            # merging of adjacent racks is exactly the refinement we want.
            split = n_clusters >= 2
            if split:
                # use the VLM to sanity-check the count where available, but
                # never to overrule the geometry
                try:
                    verdict = self.judge.adjudicate_box(
                        scene, box,
                        question=("Does the red box contain one rack or multiple racks? "
                                  "If multiple, they should be split."),
                        options=["one rack", "multiple racks"],
                    )
                    choice = (verdict.params or {}).get("choice", "")
                    # if VLM saw multiple, trust it over the peak count too
                    if choice == "multiple racks":
                        split = True
                except Exception:
                    pass
            if split:
                width_unit = float(self.opts.get("width_unit", 0.6))
                # split into the number of racks the geometry actually found
                # (n_clusters), not a width_unit guess -- a 2.6m box with two
                # clusters should become TWO racks, not round(2.6/0.6)=4.
                n = max(2, min(int(n_clusters), int(round(box.size[0] / width_unit))))
                return Verdict(action="split", params={"n": n, "width_unit": width_unit})
            return Verdict(action="keep")
        if issue.issue_type == IssueType.MERGED_NEIGHBORS and len(issue.box_ids) >= 2:
            # Two boxes that do NOT intersect but sit close / aligned: they may
            # be different faces of ONE rack (front/back/side) -- the kind of
            # merge that cannot be enumerated with rules and needs reasoning.
            # Ask the VLM over a crop that shows BOTH boxes.
            a = scene.get_box(issue.box_ids[0])
            b = scene.get_box(issue.box_ids[1])
            if a is None or b is None:
                return Verdict(action="keep")
            verdict = self.judge.adjudicate_pair(
                scene, a, b,
                question=("Do the two red boxes belong to the SAME rack (different "
                          "faces / a split rack) or are they TWO SEPARATE racks? "
                          "Merge them only if they are the same device."),
                options=["same rack, merge", "two separate racks"],
            )
            choice = (verdict.params or {}).get("choice", "")
            if choice == "same rack, merge" or verdict.action == "merge":
                return Verdict(action="merge", params={"box_ids": [a.box_id, b.box_id]})
            return Verdict(action="keep")
        if issue.issue_type == IssueType.FALSE_POSITIVE and box is not None:
            # persist the exact per-box evidence image the VLM sees: when
            # the global pass nominates a box, this local crop is what the
            # confirm/refuse decision was made on -- the key artifact for
            # auditing (and tuning) that decision
            self._save_local_evidence(scene, box, issue)
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
        if verdict.action == "merge":
            ids = (verdict.params or {}).get("box_ids") or issue.box_ids
            boxes = [scene.get_box(i) for i in ids]
            boxes = [b for b in boxes if b is not None]
            if len(boxes) < 2:
                return False
            merged = geo.merge_box_pair(scene, boxes[0], boxes[1])
            if merged is None:
                return False
            for b in boxes:
                scene.remove_box(b.box_id)
            merged.source = BoxSource.AGENT_FIX
            merged.confidence = Confidence.HIGH
            scene.boxes.append(merged)
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
        # merged-row: after split, no single piece should remain as wide as
        # (or wider than) the original fused box -- i.e. the split actually
        # took. A piece is fine if it is clearly narrower than the parent.
        if issue.issue_type == IssueType.MERGED_ROW:
            parent = scene.get_box(issue.box_ids[0])
            parent_w = parent.size[0] if parent is not None else None
            pieces = [b for b in scene.boxes
                      if b.meta.get("split_from") in issue.box_ids or b.box_id in issue.box_ids]
            for p in pieces:
                if parent_w is not None and p.size[0] >= parent_w * 0.95:
                    return False
                if p.size[0] > width_unit * 3.0:  # sanity: absurdly wide piece
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
