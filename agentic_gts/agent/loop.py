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
                continue
            # width-grid misfit: the box spans a non-integer number of rack
            # units (e.g. one whole device + half of the next). Sparse
            # half-devices rarely form a density cluster, so the cluster
            # check above misses them -- the GRID does not (a 0.9 m box is
            # neither 1 nor 2 units of 0.6 m). Only nominates; the profile
            # cliffs / VLM arbitration decide what to do.
            L = float(b.size[0])
            if L > width_unit * 1.2:
                nearest = max(1, round(L / width_unit))
                if abs(L - nearest * width_unit) > 0.25 * width_unit:
                    issues.append(Issue(
                        IssueType.WIDTH_MISFIT, [b.box_id], self._region(b),
                        detail=f"width={L:.2f} not on {width_unit:.1f} grid",
                        severity=0.6))
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
        # NOTE: same-device fragment pairs (old MERGED_NEIGHBORS) are no
        # longer detected here: the per-box VLM refine + deterministic
        # geometric merge in run() handle them before this loop runs.
        return issues

    # ---------------- god-view global audit ----------------
    def _save_godview_png(self, scene: Scene, name: str = "godview.png") -> None:
        """Persist the exact top-down image the VLM audits on (best-effort)."""
        if not self.out_dir:
            return
        try:
            from agentic_gts.agent.judge import render_godview_png
            import os as _os
            _os.makedirs(self.out_dir, exist_ok=True)
            path = _os.path.join(self.out_dir, name)
            with open(path, "wb") as f:
                f.write(render_godview_png(scene.points, scene.boxes,
                                           gs_ply=scene.meta.get("gs_ply")))
            print(f"[diag][C] godview render -> {path}")
        except Exception as e:
            print(f"[diag][C] godview render save failed ({type(e).__name__})")

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
        self._save_godview_png(scene, "godview.png")
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
        # 1. fine-grained per-box pose refinement FIRST: the VLM proposes
        # quantized size/yaw corrections (z-rotation only) from the
        # axes-annotated local view; geometry tools re-fit the box to
        # point support. Doing this BEFORE merging is the whole point:
        # fragments of one rack become co-axial (same corrected yaw), so
        # the geometric merge below can pair them reliably.
        self._vlm_refine(scene, report)
        # 2. geometric merge of the now co-axial fragments: with a common
        # yaw, same-device fragments satisfy the deterministic front/back
        # + side complement rules, and the VLM merge adjudication is no
        # longer needed (the old MERGED_NEIGHBORS VLM pass is retired).
        self._geometric_merge(scene, report)
        # 2.5 row-depth completion: thin single-face fragments (rows
        # scanned only from their facades) are expanded to the full row
        # thickness from the cross-axis surface-band profile BEFORE the
        # repair loop audits anything -- no VLM stage can grow a depth
        # (dw clamp + shrink-only refit), so this must be geometry.
        self._depth_completion(scene, report)
        # 3. repair loop: false positives, fused rows, overlaps.
        # fixes cascade: a split creates fragments that may need merging, a
        # merge may overlap a neighbour. One detection pass cannot see the
        # problems the fixes THEMSELVES introduce -- so re-detect after each
        # round (rules only; the god-view runs once up front) until a
        # fixpoint or the round cap. `handled` prevents retrying an issue
        # that failed verification (its key is stable while geometry is).
        handled: set = set()
        max_rounds = int(self.opts.get("agent_rounds", 3))
        issues = self.godview_pass(scene) + self.detect_issues(scene)
        for rnd in range(max_rounds):
            from collections import Counter
            cnt = Counter(i.issue_type.value for i in issues)
            print(f"[diag][C] round {rnd}: issues detected: "
                  f"{dict(cnt) if cnt else 'none'}")
            fresh = 0
            for issue in issues:
                key = (issue.issue_type.value, tuple(issue.box_ids),
                       issue.detail)
                if key in handled:
                    continue
                handled.add(key)
                fresh += 1
                ok = self._handle_issue(scene, issue, report)
                entry = {"issue": issue.to_dict(), "ok": ok}
                (report.resolved if ok else report.unresolved).append(entry)
            if fresh == 0:
                break
            issues = self.detect_issues(scene)
        # final edge refinement: snap every box to its point support
        self._refine_edges(scene)
        # confidence tagging (BEFORE the final QA: the QA's LOW marks must
        # survive as the last word, not be overwritten back to HIGH by a
        # good support fraction)
        for b in scene.boxes:
            sup = geo.support_fraction(scene, b)
            if b.meta.get("depth_completed"):
                # a completed box spans the hollow cabinet interior by
                # design -- interior occupancy is structurally low, the
                # faces (both observed) are the honest support signal
                sup = max(sup, geo.face_support_fraction(scene, b))
            if sup > 0.3 and b.source != BoxSource.ROW_COMPLETION:
                b.confidence = Confidence.HIGH
            elif sup > 0.15:
                b.confidence = Confidence.MID
            else:
                b.confidence = Confidence.LOW
        # final global QA over the REPAIRED state: the repair loop can
        # itself introduce global anomalies, and the first god-view pass
        # ran before any fix so it never saw them
        self._final_godview_qa(scene, report)
        return report

    def _final_godview_qa(self, scene: Scene, report: AgentReport) -> None:
        """One last god-view audit over the repaired state.

        Late flags are NOT executed (repairs had their chance) -- a
        deletion here could not be re-examined. Instead the box is marked
        LOW confidence and surfaces as an unresolved entry for human
        review. Mock/failure -> silently skipped.
        """
        self._save_godview_png(scene, "godview_final.png")
        try:
            flagged = self.judge.adjudicate_godview(scene, scene.boxes)
        except Exception as e:
            print(f"[diag][C] final godview QA error ({type(e).__name__}) -> skipped")
            return
        if not flagged:
            print("[diag][C] final godview QA: clean")
            return
        for f in flagged:
            if not (0 <= f["index"] < len(scene.boxes)):
                continue
            b = scene.boxes[f["index"]]
            b.confidence = Confidence.LOW
            print(f"[diag][C] final godview flagged #{f['index']} "
                  f"@({b.center[0]:.1f},{b.center[1]:.1f}): {f['reason']} "
                  f"-> LOW confidence, human review")
            issue = Issue(IssueType.FALSE_POSITIVE, [b.box_id],
                          self._region(b),
                          detail=f"final godview: {f['reason']}", severity=0.7)
            report.unresolved.append({"issue": issue.to_dict(), "ok": False})

    def _geometric_merge(self, scene: Scene, report: AgentReport) -> None:
        """Deterministic fragment merge AFTER the per-box pose refinement.

        The VLM merge adjudication (old MERGED_NEIGHBORS pass) is retired:
        once refine has corrected each fragment's yaw/size to the device,
        fragments of ONE rack are co-axial, and the B0 geometry rules
        (front/back + side complement) pair them reliably -- no reasoning
        needed. Reuses rules.fuse_fragments verbatim.
        """
        try:
            from agentic_gts.rules.rules import fuse_fragments
            yaw = float(scene.meta.get("yaw", 0.0))
            trusted = bool(self.opts.get("trust_input_boxes"))
            before = len(scene.boxes)
            boxes, n_absorbed = fuse_fragments(scene, yaw=yaw,
                                               trusted=trusted)
            if n_absorbed > 0:
                scene.boxes = boxes
                report.actions_taken.append(
                    {"issue_id": "geometric_merge", "action": "merge",
                     "params": {"absorbed": n_absorbed}})
                print(f"[diag][C] geometric merge absorbed {n_absorbed} "
                      f"fragments ({before} -> {len(scene.boxes)} boxes)")
        except Exception as e:
            print(f"[diag][C] geometric merge failed ({type(e).__name__}: "
                  f"{e}) -> skipped")

    @staticmethod
    def _row_mates(scene: Scene, box: OrientedBox) -> list:
        """Other boxes of the SAME row: yaw aligned (mod 180 deg -- a
        back-view fragment's yaw is flipped), near the row line. Facing
        rows across the aisle sit one row-pitch away and are excluded by
        the cross-offset cap."""
        mates = []
        for b in scene.boxes:
            if b is box:
                continue
            dyaw = abs(math.atan2(math.sin(b.yaw - box.yaw),
                                  math.cos(b.yaw - box.yaw)))
            dyaw = min(dyaw, abs(math.pi - dyaw))
            if dyaw > math.radians(12):
                continue
            if abs(b.center[2] - box.center[2]) > 1.2:
                continue
            loc = box.world_to_local(np.asarray([b.center], dtype=float))[0]
            if abs(loc[1]) > 1.7 or abs(loc[0]) > 5.0:
                continue
            mates.append(b)
        return mates

    def _depth_completion(self, scene: Scene, report: AgentReport) -> None:
        """Expand thin single-face fragments to the full row depth.

        A row scanned only from its facades leaves middle-of-row devices
        as thin boxes (each initial box hugs the one face its view
        observed). The oblique top-down view can SHOW the VLM the
        mismatch, but nothing downstream can act on it: the VLM's dw is
        clamped to +/-0.5m and the deliberately shrink-only refit
        collapses any growth back to the observed face shell. The row's
        own two surface bands in the cross-axis density profile are the
        ground truth -- complete the depth geometrically BEFORE the
        repair loop audits anything (see geo.complete_row_depth).

        Opposite-face fragments of the same device (the expansion target
        band is their observed face) are absorbed instead of left to
        collide as OVERLAP issues. Completed boxes are marked
        meta['depth_completed'] so the final edge refinement and the
        confidence tagging treat the hollow interior correctly.
        """
        try:
            n_done, n_absorbed = 0, 0
            live = {x.box_id for x in scene.boxes}
            for b in list(scene.boxes):
                if b.box_id not in live:    # absorbed by an earlier completion
                    live.discard(b.box_id)
                    continue
                # the VLM fit no longer judges depth at all (dw retired:
                # pseudo-precision), so geometry owns EVERY under-deep
                # box, thin fragment or not. complete_row_depth itself
                # requires the completed depth to exceed the current one
                # by >= 0.2m and passes the wall/overlap guards.
                if b.size[1] >= 0.9:
                    continue
                mates = self._row_mates(scene, b)
                new = None
                try:
                    new = geo.complete_row_depth(scene, b, mates)
                except Exception as e:
                    print(f"[diag][C] depth completion error on "
                          f"{b.box_id[:6]} ({type(e).__name__}: {e})")
                    continue
                if new is None:
                    continue
                # absorb opposite-face fragments of the same device: thin
                # boxes whose centre falls inside the completed footprint
                absorbed = []
                for o in list(scene.boxes):
                    if o is b or o.size[1] >= 0.5:
                        continue
                    if new.contains(np.asarray([o.center], dtype=float),
                                   margin=0.15):
                        absorbed.append(o)
                # overlap guard against everything that survives
                rest = [o for o in scene.boxes
                        if o is not b and o not in absorbed]
                if any(new.iou_2d(o) > 0.25 for o in rest):
                    continue
                old_d = b.size[1]
                b.center = new.center
                b.size = new.size
                b.meta["depth_completed"] = True
                for o in absorbed:
                    scene.remove_box(o.box_id)
                n_done += 1
                n_absorbed += len(absorbed)
                print(f"[diag][C] depth completion: box {b.box_id[:6]} "
                      f"{old_d:.2f} -> {b.size[1]:.2f} m"
                      + (f" (absorbed {len(absorbed)})" if absorbed else ""))
            if n_done:
                report.actions_taken.append(
                    {"issue_id": "depth_completion", "action": "expand_depth",
                     "params": {"completed": n_done, "absorbed": n_absorbed}})
        except Exception as e:
            print(f"[diag][C] depth completion failed ({type(e).__name__}: "
                  f"{e}) -> skipped")

    def _vlm_refine(self, scene: Scene, report: AgentReport) -> None:
        """Two-phase per-box refinement, SEQUENTIAL by dependency:

          phase 1 -- ORIENTATION from the oblique near-top-down view:
          the row direction reads best with little perspective
          foreshortening, and a skewed box makes every horizontal-view
          extent judgment unreliable (its wireframe edges no longer run
          parallel to the device faces);
          phase 2 -- LENGTH ends from the front/side views, RE-RENDERED
          on the corrected box so the extent question sees honest
          geometry.

        The VLM nominates DIRECTIONS only, geometry measures magnitudes
        (metres/degrees from an image are pseudo-precision):
          - phase 1: sweep the nominated direction (opposite direction
            as fallback -- point evidence overrules a wrong nomination)
            to the support peak (geo.sweep_yaw);
          - phase 2: 'short' end -> growth re-fit -- the seed's x is
            enlarged (max 0.25m: edge-level misalignment; a full
            grid-unit shortfall is WIDTH_MISFIT's split/merge domain,
            not fine refine) so the density span can extend to the
            device's true edge, stopping at the gap to the neighbour;
            'over' end -> plain re-fit -- the shrink-only span snaps
            back to the point support.
        Guards: IoU < 0.3 with the original rolls back; height (and
        completed depth) trusted throughout. A hallucinated nomination
        is inert by construction: 'short' with no points beyond the edge
        re-fits to the same span, 'over' with full support keeps it.
        """
        for b in list(scene.boxes):
            cur = b
            # ---- phase 1: yaw from the oblique near-top-down view ----
            try:
                v1 = self.judge.adjudicate_yaw(scene, cur)
            except Exception as e:
                print(f"[diag][C] yaw refine error on {b.box_id[:6]} "
                      f"({type(e).__name__}) -> skipped")
                v1 = None
            if (v1 is not None and v1.action == "refine"
                    and v1.params.get("yaw_dir") in ("cw", "ccw")):
                direction = 1.0 if v1.params["yaw_dir"] == "ccw" else -1.0
                swept = None
                try:
                    swept = geo.sweep_yaw(scene, cur, direction=direction)
                except Exception as e:
                    print(f"[diag][C] yaw sweep error on {b.box_id[:6]} "
                          f"({type(e).__name__}: {e})")
                if swept is not None:
                    yaw, refit = swept
                    if refit.iou_2d(b) >= 0.3:
                        print(f"[diag][C] refine {b.box_id[:6]}: yaw "
                              f"sweep -> {math.degrees(yaw - b.yaw):+.1f}deg "
                              f"(nominated {v1.params['yaw_dir']})")
                        cur = self._adopt_refit(scene, cur, refit)
                        report.actions_taken.append(
                            {"issue_id": "vlm_refine", "action": "refine",
                             "params": {"yaw_dir": v1.params["yaw_dir"],
                                        "applied_deg":
                                        round(math.degrees(yaw - b.yaw), 1)}})
            # ---- phase 2: x-ends on the CORRECTED box ----
            try:
                verdict = self.judge.adjudicate_extent(scene, cur)
            except Exception as e:
                print(f"[diag][C] extent refine error on {b.box_id[:6]} "
                      f"({type(e).__name__}) -> skipped")
                continue
            if verdict.action != "refine" or not verdict.params:
                continue
            p = verdict.params
            ends = (p.get("x_minus"), p.get("x_plus"))
            if "short" in ends or "over" in ends:
                keep_depth = bool(cur.meta.get("depth_completed"))
                # one-sided 'short': shift the seed centre toward the
                # nominated end -- a symmetric growth alone cannot reach
                # an edge further than half(growth) from the centre, and
                # fit_box_to_points re-centres on the trimmed point span
                fwdx = np.array([math.cos(cur.yaw), math.sin(cur.yaw)])
                cx, cy = cur.center[:2]
                if p.get("x_plus") == "short" and p.get("x_minus") != "short":
                    cx, cy = (cx + fwdx[0] * 0.125, cy + fwdx[1] * 0.125)
                elif p.get("x_minus") == "short" and p.get("x_plus") != "short":
                    cx, cy = (cx - fwdx[0] * 0.125, cy - fwdx[1] * 0.125)
                seed = (cur.size[0] + 0.25 if "short" in ends else cur.size[0],
                        cur.size[1], cur.size[2])
                refit = geo.fit_box_to_points(scene, (cx, cy), seed,
                                              cur.yaw, keep_height=True,
                                              keep_depth=keep_depth)
                if refit is not None and refit.iou_2d(b) >= 0.3:
                    if abs(refit.size[0] - cur.size[0]) > 0.02:
                        print(f"[diag][C] refine {b.box_id[:6]}: x re-fit "
                              f"{cur.size[0]:.2f} -> {refit.size[0]:.2f} "
                              f"({'+'.join(e for e in ends if e != 'ok')})")
                        cur = self._adopt_refit(scene, cur, refit)
                        report.actions_taken.append(
                            {"issue_id": "vlm_refine", "action": "refine",
                             "params": {"ends": {"x_minus": p.get("x_minus"),
                                                 "x_plus": p.get("x_plus")},
                                        "length":
                                        round(float(refit.size[0]), 2)}})
                else:
                    print(f"[diag][C] refine {b.box_id[:6]}: x nomination "
                          f"inert (no support change) -> keep")

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

    def _refine_edges(self, scene: Scene) -> None:
        """Snap box edges to point support for the final layout accuracy.

        Expansion is tiny (2cm): adjacent racks are only mm apart, so any
        larger search window would absorb the neighbour's surface points.
        Heights are trusted input -- never re-derived from points here.
        """
        refined: list[OrientedBox] = []
        for b in scene.boxes:
            refit = geo.fit_box_to_points(scene, b.center[:2],
                                          (b.size[0] + 0.02, b.size[1] + 0.02, b.size[2]),
                                          b.yaw, keep_height=True,
                                          keep_depth=bool(b.meta.get("depth_completed")))
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

    def _decide_width_misfit(self, scene: Scene, box: OrientedBox) -> Verdict:
        """Width-grid misfit: decide split / truncate / keep.

        Priority of evidence:
        1. profile GAPS (interior empty runs) -- hard geometry: split at
           the cliff positions, no VLM needed.
        2. profile TAILS (fading ends) -- hard geometry: truncate the box
           to the strong-density span (the observed half-device or noise
           tail loses its claim on the box).
        3. neither -- ambiguous (flush devices, a genuinely wide unit):
           ask the VLM for the device count. "one device" (or a failed /
           mock call) -> keep: a wide single device is legal and the grid
           prior alone must never butcher it. "multiple" -> split at the
           grid positions from the box edge (devices tile the row).
        """
        prof = geo.profile_cuts(scene, box)
        if prof["gaps"]:
            return Verdict(action="split",
                           params={"n": len(prof["gaps"]) + 1,
                                   "cuts": prof["gaps"]},
                           detail=f"gaps at {[f'{c:.2f}' for c in prof['gaps']]}")
        lo, hi = prof["tails"]
        if lo is not None or hi is not None:
            return Verdict(action="truncate",
                           params={"lo": lo, "hi": hi},
                           detail=f"tails lo={lo} hi={hi}")
        try:
            verdict = self.judge.adjudicate_box(
                scene, box,
                question=("Does the red wireframe cover exactly ONE device, "
                          "or does it extend past a device boundary and "
                          "cover more than one (possibly partial) device?"),
                options=["one device", "multiple devices"],
            )
            choice = (verdict.params or {}).get("choice", "")
        except Exception:
            choice = ""
        if choice == "multiple devices":
            width_unit = float(self.opts.get("width_unit", 0.6))
            L = float(box.size[0])
            n = max(2, int(round(L / width_unit)))
            # grid-aligned cuts from the box edge: devices tile the row, so
            # boundaries fall on unit multiples when no cliff is visible
            cuts = [-L / 2 + width_unit * k for k in range(1, n)]
            cuts = [c for c in cuts if -L / 2 + 0.1 < c < L / 2 - 0.1]
            if not cuts:
                return Verdict(action="keep", detail="no valid grid cut")
            return Verdict(action="split", params={"n": len(cuts) + 1,
                                                    "cuts": cuts},
                           detail=f"vlm: multiple, grid cuts")
        return Verdict(action="keep", detail="no cliff, vlm/default: one device")

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
        if issue.issue_type == IssueType.WIDTH_MISFIT and box is not None:
            return self._decide_width_misfit(scene, box)
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
                # the density profile is the authority on WHERE devices
                # end. A sparse tail can masquerade as a second "cluster"
                # to center_field_clusters (the 1.5-device box: one whole
                # rack + a fading half-observed neighbour) -- equal-splitting
                # that produces two wrong halves. Consult the profile:
                # gaps -> split at the cliffs; tails -> truncate the fading
                # end; only a clean profile falls back to equal division.
                prof = geo.profile_cuts(scene, box)
                if prof["gaps"]:
                    return Verdict(action="split",
                                   params={"n": len(prof["gaps"]) + 1,
                                           "cuts": prof["gaps"]})
                lo, hi = prof["tails"]
                if lo is not None or hi is not None:
                    return Verdict(action="truncate",
                                   params={"lo": lo, "hi": hi},
                                   detail="second cluster is a fading tail")
                # split into the number of racks the geometry actually found
                # (n_clusters), not a width_unit guess -- a 2.6m box with two
                # clusters should become TWO racks, not round(2.6/0.6)=4.
                n = max(2, min(int(n_clusters), int(round(box.size[0] / width_unit))))
                return Verdict(action="split", params={"n": n, "width_unit": width_unit})
            return Verdict(action="keep")
        if issue.issue_type == IssueType.FALSE_POSITIVE and box is not None:
            # The evidence image (three-view composite) is rendered AND
            # persisted inside the judge (evidence_{id}.png) -- no
            # second render here.
            verdict = self.judge.adjudicate_box(
                scene, box,
                question="Is there truly a device at the red box, or is it empty space?",
                options=["real device", "empty space"],
            )
            choice = (verdict.params or {}).get("choice", "")
            sup = geo.support_fraction(scene, box)
            # Deletion is irreversible and the dangerous direction: require
            # BOTH the VLM's positive identification and a confident reply.
            # A low-confidence "empty space" degrades to a shrink attempt /
            # unresolved + LOW confidence (human review), never a delete.
            vlm_says_empty = (choice == "empty space"
                              and verdict.confidence >= 0.6)
            if vlm_says_empty or sup < 0.08:
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
                                 params.get("width_unit"),
                                 params.get("cuts"))
            if len(subs) < 2:
                return False
            scene.remove_box(box.box_id)
            for s in subs:
                s.source = BoxSource.AGENT_FIX
                scene.boxes.append(s)
            return True
        if verdict.action == "truncate":
            # cut the box at the profile tail bounds, then refit: the seed
            # keeps a small margin on the NON-cut side so the refit can
            # re-capture the device's true edge there, while the cut side
            # stays put (the tail points are outside the seed and cannot
            # drag the percentile back out).
            box = scene.get_box(issue.box_ids[0])
            if box is None:
                return False
            params = verdict.params or {}
            half_x = box.size[0] / 2.0
            lo = -half_x if params.get("lo") is None else float(params["lo"])
            hi = half_x if params.get("hi") is None else float(params["hi"])
            if hi - lo < 0.25:
                return False
            margin = 0.15
            # margin on the NON-cut side only: lets the refit re-capture
            # the device's true edge where the box under-covered it; the
            # cut side stays fixed so tail points cannot stretch it back
            seed_lo = lo - (margin if params.get("lo") is None else 0.0)
            seed_hi = hi + (margin if params.get("hi") is None else 0.0)
            axis = box.rotation[:, 0]
            mid = (seed_lo + seed_hi) / 2.0
            new_c = np.asarray(box.center) + axis * mid
            seed = (seed_hi - seed_lo, box.size[1], box.size[2])
            refit = geo.fit_box_to_points(scene, new_c[:2], seed, box.yaw)
            if refit is None or refit.size[0] >= box.size[0] - 0.03:
                return False  # no real reduction -> rollback to original
            refit.source = BoxSource.AGENT_FIX
            refit.row_id = box.row_id
            refit.device_type = box.device_type
            scene.remove_box(box.box_id)
            scene.boxes.append(refit)
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
