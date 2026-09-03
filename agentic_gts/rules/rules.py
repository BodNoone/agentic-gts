"""Stage B: deterministic rule-based post-processing.

Uses domain priors + deterministic geometry only (no ML). This is the
cheapest and highest-leverage stage; it should absorb 40-60% of errors
before the agent is even consulted.

Rules:
  0. fragment merging (B0)                 -> fuses per-view back-projection
                                             fragments of the same device
  1. size/large-aspect filter              -> removes false positives / oversized
  2. row alignment                         -> fixes slight misalignment
  3. point-support check                   -> removes empty boxes
  4. row-structure gap completion          -> recovers missing racks
  5. merged-row splitting                  -> splits fused adjacent racks
"""
from __future__ import annotations

import math

import numpy as np

from agentic_gts.core.models import (
    BoxSource,
    Confidence,
    DeviceType,
    OrientedBox,
    Issue,
    IssueType,
    Scene,
)
from agentic_gts.tools import geometry as geo


def apply_rules(scene: Scene, opts: dict | None = None) -> tuple[list[OrientedBox], list[Issue]]:
    """Run the deterministic rule pipeline on scene.boxes.

    Mutates scene.boxes in place to the refined set and returns
    (refined_boxes, issues_found).
    """
    opts = opts or {}
    yaw = float(opts.get("yaw", scene.meta.get("yaw", 0.0)))
    width_unit = float(opts.get("width_unit", 0.6))
    depth = float(opts.get("depth", 1.1))
    height = float(opts.get("height", 2.0))
    min_support = float(opts.get("min_support", 0.15))
    max_aspect = float(opts.get("max_aspect", 2.2))
    issues: list[Issue] = []

    boxes = list(scene.boxes)
    _sz = [tuple(round(s, 2) for s in b.size) for b in boxes[:10]]
    _more = "..." if len(boxes) > 10 else ""
    print(f"[diag][B] input boxes: {len(boxes)}  (sizes={_sz}{_more})")

    # 0) fragment merge (B0): per-view back-projection fragments of the
    # same device are fused before any filtering -- a fragment alone often
    # fails the wall/support tests below and would be wrongly deleted.
    # Deliberately NOT recorded as an Issue: issues feed the agent, which
    # would try to "fix" (split) exactly what we just merged.
    # A box WIDER than a single rack unit is already multi-device (merged
    # racks) or a deliberate group, not a fragment -- it must NOT be absorbed
    # by fragment merging. It is held out and re-added after.
    if opts.get("trust_input_boxes"):
        wide = [b for b in boxes if b.size[0] > width_unit * 1.2]
        frag = [b for b in boxes if b.size[0] <= width_unit * 1.2]
        scene.boxes = frag
        boxes, n_absorbed = geo.merge_fragments(scene, yaw=yaw, trusted=True)
        boxes = boxes + wide
        print(f"[diag][B] fragment merge (B0): absorbed {n_absorbed} -> "
              f"{len(boxes)} boxes ({len(wide)} wide/multi-device held out)")
    else:
        boxes, n_absorbed = geo.merge_fragments(scene, yaw=yaw)
        print(f"[diag][B] fragment merge (B0): absorbed {n_absorbed} -> "
              f"{len(boxes)} boxes")
    scene.boxes = boxes

    # 0) wall filter: a box whose points form a single thin sheet in the
    # cross (depth) direction is a wall fragment, not a device. Devices have
    # front+back faces (bimodal depth) or full depth >= ~0.5m.
    # SKIPPED for trusted external input (trust_input_boxes): the wall prior
    # exists to clean Stage A geometric candidates. Mask-detector boxes are
    # already semantic ("this is a device"), and a single-view fragment is
    # legitimately a thin sheet -- running the wall test on fragments kills
    # exactly the boxes the user vouched for.
    if opts.get("trust_input_boxes"):
        print("[diag][B] wall filter: skipped (trusted external input boxes)")
    else:
        non_wall: list[OrientedBox] = []
        drop_wall_detail: list[str] = []
        for b in boxes:
            if _is_wall_sheet(scene, b):
                drop_wall_detail.append(
                    f"{b.box_id[:4]}@({b.center[0]:.1f},{b.center[1]:.1f}) "
                    f"{b.size[0]:.2f}x{b.size[1]:.2f}x{b.size[2]:.2f}")
                issues.append(Issue(IssueType.FALSE_POSITIVE, [b.box_id], _box_region(b),
                                    detail="wall sheet"))
                continue
            non_wall.append(b)
        print(f"[diag][B] wall filter: dropped {len(boxes) - len(non_wall)}, "
              f"kept {len(non_wall)}")
        for d in drop_wall_detail[:15]:
            print(f"[diag][B]   wall-dropped: {d}")
        boxes = non_wall
        scene.boxes = non_wall

    # 1) point-support + aspect filter (drop empty / implausible).
    # SKIPPED for trusted external input: the detector boxes are already
    # semantic ('this is a device'), and the rules here are geometric priors
    # tuned for Stage-A candidates. Single-view back-projection fragments are
    # legitimately small / low-interior-support, and the aspect/support tests
    # would delete exactly the fine pieces the user vouched for. Fragment
    # merging (B0) already handled the 'one rack split into many' case; the
    # remaining size/shape problems are for the agent, not the rule filter.
    if opts.get("trust_input_boxes"):
        kept = boxes
        print("[diag][B] support/aspect filter: skipped (trusted external input boxes)")
    else:
        kept: list[OrientedBox] = []
        n_low, n_aspect = 0, 0
        drop_sup_detail: list[str] = []
        for b in boxes:
            # interior-volume support, rescued by best-face coverage: a
            # single-view back-projection fragment observes only one face, so
            # its interior is mostly empty and volume occupancy is structurally
            # low -- but its observed face is fully backed by points. A floating
            # false positive has neither.
            sup = geo.support_fraction(scene, b)
            if sup < min_support:
                sup = max(sup, geo.face_support_fraction(scene, b))
            aspect = max(b.size[0], b.size[1]) / max(min(b.size[0], b.size[1]), 1e-3)
            if sup < min_support:
                n_low += 1
                drop_sup_detail.append(
                    f"{b.box_id[:4]}@({b.center[0]:.1f},{b.center[1]:.1f}) "
                    f"{b.size[0]:.2f}x{b.size[1]:.2f}x{b.size[2]:.2f} sup={sup:.2f}")
                issue_region = _box_region(b)
                issues.append(Issue(IssueType.LOW_SUPPORT, [b.box_id], issue_region,
                                    detail=f"support={sup:.2f}"))
                continue
            if aspect > max_aspect and b.size[0] > 0.3:
                issues.append(Issue(IssueType.OVERSIZED, [b.box_id], _box_region(b),
                                    detail=f"aspect={aspect:.2f}"))
                # split oversized-aspect boxes by standard width
                n = max(1, int(round(b.size[0] / width_unit)))
                if n > 1:
                    kept.extend(geo.split_box(scene, b, n, width_unit))
                    continue
                # otherwise refit
                refit = geo.fit_box_to_points(scene, b.center[:2], b.size, b.yaw)
                if refit is not None:
                    refit.source = BoxSource.RULE_FIX
                    refit.confidence = Confidence.MID
                    kept.append(refit)
                continue
            kept.append(b)
        if n_low:
            print(f"[diag][B] support filter: dropped {n_low} "
                  f"(supports={drop_sup_detail[:15]}{'...' if len(drop_sup_detail) > 15 else ''}, "
                  f"thr={min_support})")
        if n_aspect:
            print(f"[diag][B] aspect filter: split/refit {n_aspect} oversized boxes")
        print(f"[diag][B] after support/aspect filter: kept {len(kept)}/{len(boxes)}")
    scene.boxes = kept
    boxes = kept

    # 2) row alignment: snap each box's yaw to the dominant row direction.
    # SKIPPED for trusted input (do not force-rotate boxes the user vouched
    # for; if a box is really misoriented the agent can fix it).
    if opts.get("trust_input_boxes"):
        print("[diag][B] row alignment: skipped (trusted external input boxes)")
    else:
        rows = geo.row_structure(scene, yaw=yaw)
        print(f"[diag][B] row structure: {len(rows)} rows (yaw={math.degrees(yaw):.1f} deg)")
        for r in rows:
            axis = np.asarray(r["axis"])
            fixed_yaw = math.atan2(axis[1], axis[0])
            for b in r["boxes"]:
                if not geo.is_aligned(b, axis, tol_deg=12.0):
                    b.yaw = fixed_yaw
                    b.source = BoxSource.RULE_FIX
                    issues.append(Issue(IssueType.MISALIGNED, [b.box_id], _box_region(b)))

    # 3) row-structure gap completion (recover missing racks).
    # SKIPPED for trusted input: adding newly-synthesized boxes here would
    # inject boxes the detector never saw -- the 'strange extra things' the
    # user is complaining about. Missing racks are the agent's job.
    if opts.get("trust_input_boxes"):
        filled: list[OrientedBox] = []
        print("[diag][B] gap completion: skipped (trusted external input boxes)")
    else:
        filled: list[OrientedBox] = []
        rows = geo.row_structure(scene, yaw=yaw)
        for r in rows:
            gaps = geo.find_gaps(scene, r, max_gap_racks=2, width_unit=width_unit)
            for g in gaps:
                nb = geo.add_box_at(scene, r, g, width_unit, depth, height)
                if nb is not None:
                    filled.append(nb)
                    issues.append(Issue(IssueType.MISSING, [], _row_region(r, yaw),
                                        detail=f"gap filled at row={r['id']}"))
        print(f"[diag][B] gap completion: filled {len(filled)} boxes")

    # 4) merged-row detection & split via center-field completeness.
    # For trusted input, do NOT hard-split here: the agent's per-box VLM
    # adjudication (with local evidence) decides whether a box is really
    # several racks and how many to cut -- a rule that splits by width_unit
    # can wrongly divide a single legitimate box or guess the wrong count.
    # The rule split stays for Stage-A (untrusted) candidates.
    if opts.get("trust_input_boxes"):
        print("[diag][B] merged-row split: handed to agent (trusted external input boxes)")
    else:
        split_new: list[OrientedBox] = []
        n_split = 0
        for b in scene.boxes:
            n_clusters, dom, _ = geo.center_field_clusters(scene, b)
            if n_clusters >= 2:
                # more than one coherent slab -> likely multiple racks fused
                n_split += 1
                k = min(n_clusters, int(round(b.size[0] / width_unit)) or n_clusters)
                k = max(k, 2)
                split_new.extend(geo.split_box(scene, b, k, width_unit))
                issues.append(Issue(IssueType.MERGED_ROW, [b.box_id], _box_region(b),
                                    detail=f"n_clusters={n_clusters}"))
            else:
                split_new.append(b)
        scene.boxes = split_new
        print(f"[diag][B] merged-row split: {n_split} boxes split")

    final = scene.boxes + filled
    scene.boxes = final
    print(f"[diag][B] final boxes: {len(final)}")
    return final, issues


def refine_boxes(scene: Scene, opts: dict | None = None) -> tuple[list[OrientedBox], list[Issue]]:
    return apply_rules(scene, opts)


def _box_region(b: OrientedBox) -> tuple[float, float, float, float]:
    c = np.asarray(b.center[:2])
    half = np.asarray(b.size[:2]) / 2.0
    r = b.rotation[:2, :2]
    corners = np.array([[-1, -1], [1, -1], [1, 1], [-1, 1]]) * half
    world = corners @ r.T + c
    return (float(world[:, 0].min()), float(world[:, 1].min()),
            float(world[:, 0].max()), float(world[:, 1].max()))


def _row_region(row: dict, yaw: float) -> tuple[float, float, float, float]:
    along = [np.asarray(b.center[:2]) @ np.asarray(row["axis"]) for b in row["boxes"]] or [0.0]
    cross = row["cross_axis_coord"]
    axis = np.asarray(row["axis"])
    lo_a, hi_a = min(along), max(along)
    pts = axis * lo_a + np.array([-axis[1], axis[0]]) * cross
    pts2 = axis * hi_a + np.array([-axis[1], axis[0]]) * cross
    return (min(pts[0], pts2[0]) - 0.5, min(pts[1], pts2[1]) - 0.5,
            max(pts[0], pts2[0]) + 0.5, max(pts[1], pts2[1]) + 0.5)


def _is_wall_sheet(scene: Scene, box: OrientedBox, depth_thr: float = 0.35) -> bool:
    """Wall fragments are thin sheets: points occupy a narrow cross-axis slab.

    A real rack shows front+back faces ~deep apart, so its point distribution
    along the cross (depth) axis is either wide (>0.5m) or bimodal. A wall
    gives a single narrow slab. We additionally require the box to be flat
    (depth << along length) to avoid killing short AC units.

    The tall-column test catches what the thin-sheet test misses: thick /
    noisy 3DGS walls whose depth spread exceeds depth_thr. We voxelize the
    box footprint and apply the shared wall_column_mask test (structure
    top above TALL_Z AND the high part itself spans >=0.5m -- the cable-tray
    guard, see wall_column_mask). A wall box has nearly all columns tall;
    a rack with an overhead tray or next to a wall does not.
    """
    from agentic_gts.segment.coarse import wall_column_mask
    region = _box_region(box)
    pts = scene.points_in_region(region)
    big = box.size[0] >= 2.0 or box.size[1] >= 2.0
    if len(pts) < 20:
        return big  # no support -> prune big empty boxes, keep small ones
    # tall-column test (applies to all sizes; robust to wall thickness)
    p = pts[pts[:, 2] > 0.3]  # drop floor points
    if len(p) >= 20:
        key = np.floor(p[:, :2] / 0.15).astype(np.int64)
        uniq, inv = np.unique(key, axis=0, return_inverse=True)
        wall = wall_column_mask(p[:, 2], inv, len(uniq))
        if len(wall) and float(np.mean(wall)) > 0.5:
            return True
    if not big:
        return False  # small boxes are never thin-sheet walls
    local = box.world_to_local(pts)
    # cross axis = local y (the depth axis)
    ydepth = local[:, 1]
    q10, q90 = np.percentile(ydepth, [10, 90])
    spread = q90 - q10
    # width of the box along depth
    depth = box.size[1]
    flat = depth <= depth_thr or (spread <= depth_thr and box.size[0] >= max(box.size[1] * 3, 0.8))
    return flat and (spread <= depth_thr)
