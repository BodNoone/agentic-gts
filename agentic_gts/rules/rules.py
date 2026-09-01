"""Stage B: deterministic rule-based post-processing.

Uses domain priors + deterministic geometry only (no ML). This is the
cheapest and highest-leverage stage; it should absorb 40-60% of errors
before the agent is even consulted.

Rules:
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

    # 0) wall filter: a box whose points form a single thin sheet in the
    # cross (depth) direction is a wall fragment, not a device. Devices have
    # front+back faces (bimodal depth) or full depth >= ~0.5m.
    non_wall: list[OrientedBox] = []
    for b in boxes:
        if _is_wall_sheet(scene, b):
            issues.append(Issue(IssueType.FALSE_POSITIVE, [b.box_id], _box_region(b),
                                detail="wall sheet"))
            continue
        non_wall.append(b)
    print(f"[diag][B] wall filter: dropped {len(boxes) - len(non_wall)}, kept {len(non_wall)}")
    scene.boxes = non_wall
    boxes = non_wall

    # 1) point-support + aspect filter (drop empty / implausible)
    kept: list[OrientedBox] = []
    n_low, n_aspect = 0, 0
    drop_sup_detail: list[str] = []
    for b in boxes:
        sup = geo.support_fraction(scene, b)
        aspect = max(b.size[0], b.size[1]) / max(min(b.size[0], b.size[1]), 1e-3)
        if sup < min_support:
            n_low += 1
            drop_sup_detail.append(f"{sup:.2f}")
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

    # 2) row alignment: snap each box's yaw to the dominant row direction
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

    # 3) row-structure gap completion (recover missing racks)
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

    # 4) merged-row detection & split via center-field completeness
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

    final = scene.boxes + filled
    scene.boxes = final
    print(f"[diag][B] merged-row split: {n_split} boxes split; final boxes: {len(final)}")
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
    box footprint and compare each column's STRUCTURE TOP (max z of the
    lowest contiguous vertical run, cut at the first >1m z-gap) against
    TALL_Z: walls are continuous floor->ceiling, devices stop at ~2.4m
    with a void below the ceiling. A wall box has nearly all columns tall;
    a rack next to a wall only has the thin wall-slice columns tall.
    """
    from agentic_gts.segment.coarse import TALL_Z, structure_top_per_label
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
        top = structure_top_per_label(p[:, 2], inv, len(uniq))
        if len(top) and float(np.mean(top > TALL_Z)) > 0.5:
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
