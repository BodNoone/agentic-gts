"""Deterministic geometry tools used by both the rule layer and the agent.

All functions return concrete coordinates (the agent never alters geometry
itself; it only *selects* which tool to call and with what discrete args).
"""
from __future__ import annotations

import math

import numpy as np

from agentic_gts.core.models import BoxSource, Confidence, DeviceType, OrientedBox, Scene


def fit_box_to_points(scene: Scene, seed_center: tuple[float, float],
                      seed_size: tuple[float, float, float], yaw: float,
                      inlier_frac: float = 0.9,
                      keep_height: bool = False,
                      keep_depth: bool = False) -> OrientedBox | None:
    """Refit an oriented box to the local point support.

    Boundary estimation via 1D occupancy histograms per axis: find the
    contiguous occupied span containing the center. Robust to sparse noise
    while keeping edges tight to the true surface.

    keep_height=True: the input box heights are TRUSTED (detector
    output) -- keep the seed's z-extent untouched instead of re-deriving
    it from point percentiles (surface fragments / ceiling cuts make
    point-based z unreliable).

    keep_depth=True: same trust for the cross-row DEPTH (size[1]). A
    joined-row box split into cabinets (ground.py split_stage) spans the
    hollow interior of a closed cabinet whose 3DGS interior is empty --
    the point-percentile span would collapse it back to a thin face shell.
    """

    seed = OrientedBox(center=(seed_center[0], seed_center[1], seed_size[2] / 2),
                       size=seed_size, yaw=yaw)
    region = _region_of_box(seed, expand=0.3)
    pts = scene.points_in_region(region)
    if len(pts) < 20:
        return None
    local = seed.world_to_local(pts)
    half = np.asarray(seed_size) / 2.0
    m = np.all(np.abs(local) <= half, axis=1)
    inside = local[m]
    if len(inside) < 20:
        return None

    # Robust per-axis span via mildly-trimmed percentiles of the surface
    # points strictly inside the seed (shrink-only: adjacent racks sit
    # millimetres away, any expansion absorbs the neighbour's face).
    qlo, qhi = np.percentile(inside, [0.5, 99.5], axis=0)
    xmin, xmax = float(qlo[0]), float(qhi[0])
    ymin, ymax = float(qlo[1]), float(qhi[1])
    if keep_height:
        zmin, zmax = -half[2], half[2]
    else:
        zmin, zmax = float(qlo[2]), float(qhi[2])
    if keep_depth:
        ymin, ymax = -half[1], half[1]

    new_size = (max(xmax - xmin, 0.15), max(ymax - ymin, 0.15), max(zmax - zmin, 0.2))
    local_center = np.array([(xmin + xmax) / 2, (ymin + ymax) / 2, (zmin + zmax) / 2])
    center = seed.local_to_world(local_center.reshape(1, 3))[0]
    box = OrientedBox(center=tuple(center), size=new_size, yaw=yaw,
                      device_type=DeviceType.RACK)
    coverage = support_fraction(scene, box)
    # keep_depth boxes span the hollow cabinet interior BY DESIGN -- their
    # interior occupancy is structurally low (one observed face band), so
    # the trust flag also relaxes the coverage floor.
    if coverage < (0.05 if keep_depth else 0.12):
        return None
    return box


def support_fraction(scene: Scene, box: OrientedBox, expand: float = 0.0) -> float:
    """Fraction of the box interior volume the point cloud actually fills.

    Uses a 3D occupancy grid; returns density of occupied voxels within the box.
    """
    region = _region_of_box(box, expand)
    pts = scene.points_in_region(region)
    if len(pts) == 0:
        return 0.0
    local = box.world_to_local(pts)
    half = np.asarray(box.size) / 2.0
    m = np.all(np.abs(local) <= half, axis=1)
    inside = local[m]
    if len(inside) < 5:
        return 0.0
    cell = 0.1
    nb = np.maximum((np.asarray(box.size) / cell).astype(int), 1)
    idx = np.clip(((inside + half) / cell).astype(int), 0, nb - 1)
    occupied = np.zeros(nb, dtype=bool)
    occupied[idx[:, 0], idx[:, 1], idx[:, 2]] = True
    return float(occupied.sum() / max(nb.prod(), 1))


def profile_cuts(scene: Scene, box: OrientedBox, cell: float = 0.05,
                 min_points: int = 40) -> dict:
    """Analyze the along-axis (local x) point-density profile of a box.

    Two hard geometric signals for the width audit:
      gaps  -- interior near-empty runs flanked by occupied bins on both
               sides: a device boundary. The cut position is the middle
               of the empty run. Both sides must still span >= 0.3 m
               (a plausible device) -- a boundary against a 0.15 m sliver
               is a TAIL, not a gap.
      tails -- leading/trailing weak runs (density < 20% of peak, at least
               0.15 m long): the box overhangs its point support with a
               fading fragment (e.g. half an observed device). Truncation
               bounds are the first/last strong bin edges.

    Returns {"gaps": [local-x cuts], "tails": (lo | None, hi | None)} in
    box-LOCAL coordinates. All-empty / too-sparse profiles return no cuts.
    """
    out = {"gaps": [], "tails": (None, None)}
    region = _region_of_box(box, expand=0.03)
    pts = scene.points_in_region(region)
    if len(pts) < min_points:
        return out
    local = box.world_to_local(pts)
    half = np.asarray(box.size) / 2.0
    inside = local[np.all(np.abs(local) <= half, axis=1)]
    if len(inside) < min_points:
        return out
    x = inside[:, 0]
    lo, hi = float(-half[0]), float(half[0])
    edges = np.arange(lo, hi + cell / 2, cell)
    if len(edges) < 4:
        return out
    hist, _ = np.histogram(x, bins=edges)
    peak = float(hist.max())
    if peak < 5:
        return out

    # ---- gaps: interior empty runs between substantial dense runs ----
    empty = hist < max(peak * 0.1, 2.0)
    i = 0
    while i < len(hist):
        if not empty[i]:
            i += 1
            continue
        j = i
        while j < len(hist) and empty[j]:
            j += 1
        # interior only: occupied bins on BOTH sides, >= 2 bins wide
        if i > 0 and j < len(hist) and (j - i) >= 2:
            if (edges[i] - lo) >= 0.3 and (hi - edges[j]) >= 0.3:
                out["gaps"].append(float((edges[i] + edges[j]) / 2.0))
        i = j

    # ---- tails: fading ends below 20% of the peak ----
    thr = max(peak * 0.2, 3.0)
    strong = hist >= thr
    k0 = 0
    while k0 < len(hist) and not strong[k0]:
        k0 += 1
    if 0 < k0 < len(hist) and (edges[k0] - lo) >= 0.15:
        out["tails"] = (float(edges[k0]), None)
    k1 = len(hist) - 1
    while k1 >= 0 and not strong[k1]:
        k1 -= 1
    if 0 <= k1 < len(hist) - 1 and (hi - edges[k1 + 1]) >= 0.15:
        out["tails"] = (out["tails"][0], float(edges[k1 + 1]))

    # the surviving span must still be a plausible device
    lo_t = lo if out["tails"][0] is None else out["tails"][0]
    hi_t = hi if out["tails"][1] is None else out["tails"][1]
    if hi_t - lo_t < 0.3:
        out["tails"] = (None, None)
    return out


def split_box(scene: Scene, box: OrientedBox, n: int,
              width_unit: float | None = None,
              cuts: list[float] | None = None) -> list[OrientedBox]:
    """Split a (merged-row) box along the row axis.

    `cuts` (box-local x positions, e.g. density-profile gap middles) take
    priority: pieces land on the measured device boundaries instead of an
    equal division -- a 0.9 m "1.5-device" box must cut at 0.6, not 0.45.
    Without cuts, falls back to equal division into n (or width_unit-derived)
    pieces.
    """
    L, W, H = box.size
    row_axis = box.rotation[:, 0]  # local x direction in world
    half = L / 2.0
    if cuts:
        bounds = [-half] + sorted(float(c) for c in cuts) + [half]
    else:
        if n is not None and n >= 2:
            k = int(n)
        elif width_unit and width_unit > 0:
            k = max(2, int(round(L / width_unit)))
        else:
            k = 2
        if k < 2:
            return [box]
        bounds = list(np.linspace(-half, half, k + 1))
    center = np.asarray(box.center)
    result = []
    for a, b in zip(bounds[:-1], bounds[1:]):
        if b - a < 0.1:      # degenerate sliver segment: skip
            continue
        seg_c = center + row_axis * ((a + b) / 2.0)
        result.append(OrientedBox(
            center=tuple(seg_c), size=(b - a, W, H), yaw=box.yaw,
            device_type=DeviceType.RACK, source=box.source,
            confidence=box.confidence, row_id=box.row_id,
            meta={"split_from": box.box_id},
        ))
    if len(result) < 2:
        return [box]
    return result


def row_structure(scene: Scene, yaw: float = 0.0,
                  cluster_tol: float = 0.25) -> list[dict]:
    """Detect rows by clustering box centers' cross-axis coordinate.

    Rows are lines of roughly-constant cross-axis position. Returns a list of
    row dicts with id, cross-axis coordinate, and sorted member boxes.
    """
    if not scene.boxes:
        return []
    centers = np.asarray([b.center[:2] for b in scene.boxes])
    # project onto row direction and cross direction
    axis = np.array([math.cos(yaw), math.sin(yaw)])
    cross = np.array([-math.sin(yaw), math.cos(yaw)])
    cross_vals = centers @ cross
    # 1D clustering on cross coordinate
    order = np.argsort(cross_vals)
    rows: list[dict] = []
    cur = [order[0]]
    for idx in order[1:]:
        if abs(cross_vals[idx] - cross_vals[cur[-1]]) > cluster_tol:
            _finalize_row(scene, cur, cross_vals, axis, rows)
            cur = [idx]
        else:
            cur.append(idx)
    _finalize_row(scene, cur, cross_vals, axis, rows)
    for r in rows:
        r["id"] = len(rows) and rows.index(r) or 0
        r["cross_axis_coord"] = float(cross_vals[r["members_idx"][0]])
    for i, r in enumerate(rows):
        r["id"] = i
    return rows


def _finalize_row(scene, indices, cross_vals, axis, rows) -> None:
    idx = np.asarray(indices, dtype=int)
    cols = []
    for j in idx:
        b = scene.boxes[j]
        along = np.asarray(b.center[:2]) @ axis
        cols.append((along, b))
    cols.sort(key=lambda t: t[0])
    member_boxes = [b for _, b in cols]
    gap_list = []
    for a, b in zip(member_boxes[:-1], member_boxes[1:]):
        along_a = np.asarray(a.center[:2]) @ axis
        along_b = np.asarray(b.center[:2]) @ axis
        gap = along_b - along_a - (a.size[0] / 2 + b.size[0] / 2)
        gap_list.append(float(max(gap, 0.0)))
    rows.append({
        "members_idx": idx.tolist(),
        "boxes": member_boxes,
        "gaps": gap_list,
        "axis": axis.tolist(),
        "cross_axis_coord": float(cross_vals[idx[0]]),
    })


def find_gaps(scene: Scene, row: dict, max_gap_racks: int = 2,
              width_unit: float = 0.6) -> list[float]:
    """Return along-axis center positions where a rack is likely missing.

    Checks (a) interior gaps between adjacent boxes and (b) row *ends*:
    if point density continues beyond the first/last box, racks are missing
    at the row ends. All candidates are validated by point support before
    being returned (support is re-checked in add_box_at as well).
    """
    boxes = row["boxes"]
    if not boxes:
        return []
    axis = np.asarray(row["axis"])
    cross_axis_coord = row["cross_axis_coord"]
    cross = np.array([-axis[1], axis[0]])
    out: list[float] = []

    # --- interior gaps ---
    for a, b, gap in zip(boxes[:-1], boxes[1:], row["gaps"]):
        if gap <= 0.05 * width_unit:
            continue
        n_units = gap / width_unit
        if n_units < 0.6:
            continue
        n_units = min(int(round(n_units)), max_gap_racks)
        center_a = np.asarray(a.center[:2]) @ axis
        center_b = np.asarray(b.center[:2]) @ axis
        for k in range(1, n_units + 1):
            frac = k / (n_units + 1)
            out.append(float(center_a + (center_b - center_a) * frac))

    # --- row ends: walk outward while point support persists ---
    ref = boxes[0]
    depth, height = ref.size[1], ref.size[2]
    along_vals = [np.asarray(b.center[:2]) @ axis for b in boxes]
    lo_end = min(along_vals) - (boxes[0].size[0] / 2)
    hi_end = max(along_vals) + (boxes[-1].size[0] / 2)
    for direction, end in ((-1, lo_end), (1, hi_end)):
        for k in range(1, max_gap_racks + 1):
            cand_along = end + direction * (width_unit * (k - 0.5) + 0.01)
            world_xy = axis * cand_along + cross * cross_axis_coord
            probe = OrientedBox(
                center=(world_xy[0], world_xy[1], height / 2),
                size=(width_unit * 0.9, depth, height),
                yaw=math.atan2(axis[1], axis[0]))
            if support_fraction(scene, probe) >= 0.15:
                out.append(float(cand_along))
            else:
                break
    return out


def add_box_at(scene: Scene, row: dict, along: float, width_unit: float,
               depth: float, height: float) -> OrientedBox | None:
    """Create a box at the given row-axis coordinate."""
    axis = np.asarray(row["axis"])
    cross_axis_coord = row["cross_axis_coord"]
    cross = np.array([-axis[1], axis[0]])
    world_xy = axis * along + cross * cross_axis_coord
    box = OrientedBox(
        center=(world_xy[0], world_xy[1], height / 2),
        size=(width_unit, depth, height),
        yaw=math.atan2(axis[1], axis[0]),
        device_type=DeviceType.RACK, source=BoxSource.ROW_COMPLETION,
        confidence=Confidence.LOW,
        row_id=row["id"],
    )
    if support_fraction(scene, box) < 0.15:
        return None
    return box


def is_aligned(box: OrientedBox, row_axis: np.ndarray, tol_deg: float = 12.0) -> bool:
    """Check whether a box's long axis aligns with the row direction."""
    bx = box.rotation[:2, 0]
    ra = np.asarray(row_axis)[:2]
    denom = (np.linalg.norm(bx) * np.linalg.norm(ra)) or 1e-9
    ang = math.degrees(math.acos(float(np.clip(abs(np.dot(bx, ra)) / denom, -1, 1))))
    return ang <= tol_deg


# --------------------------------------------------------------- B0: fragments
def merge_fragments(scene: Scene, yaw: float = 0.0,
                    overlap_thr: float = 0.35,
                    max_depth: float = 2.2,
                    max_width: float = 0.9,
                    max_merge_size: float = 1.2,
                    trusted: bool = False) -> tuple[list[OrientedBox], int]:
    """Merge fragment boxes that observe the SAME device (Stage B0).

    Input boxes coming from per-view mask back-projection are fragments:
    the same device yields several small boxes, each covering the visible
    face from one viewpoint. Two boxes belong to the same device when,
    projected onto the row frame (yaw):

      a) their footprints overlap substantially -- same-spot fragments
         (intersection over the SMALLER footprint >= overlap_thr), or
      b) they cover complementary front/back halves -- the along extents
         overlap nearly fully while the cross extents are disjoint and
         their union stays within one plausible rack depth (<= max_depth), or
      c) they cover complementary left/right halves -- the cross extents
         overlap nearly fully while the along extents are disjoint and
         their union stays within one plausible rack width (<= max_width).

    Deliberately NOT merged: adjacent racks in a row (along union spans two
    devices > max_width) and back-to-back racks (cross union > max_depth)
    -- those stay separate and are handled by gap completion /
    merged-row split. Known limitation: two adjacent sub-0.45m devices
    sitting flush would satisfy (c) and wrongly merge.

    `trusted=True` disables rule (c): for detector boxes that ALREADY make
    per-device cuts, left/right complementary halves usually mean two DIFFERENT
    adjacent devices, not one split device -- merging them would undo the
    detector's own separation. (a)+(b) still merge true same-device fragments.

    Returns (boxes, n_merges). n_merges counts absorbed boxes.
    """
    axis = np.array([math.cos(yaw), math.sin(yaw)])
    cross = np.array([-math.sin(yaw), math.cos(yaw)])
    boxes = list(scene.boxes)

    def _extents(b: OrientedBox):
        cs = np.asarray(b.corners_2d())          # 4x2 world
        a = cs @ axis
        c = cs @ cross
        return float(a.min()), float(a.max()), float(c.min()), float(c.max())

    def _merge_pair(a: OrientedBox, b: OrientedBox) -> OrientedBox:
        # union extent in the row frame, snapped to the global yaw
        ea, eb = _extents(a), _extents(b)
        a0, a1 = min(ea[0], eb[0]), max(ea[1], eb[1])
        c0, c1 = min(ea[2], eb[2]), max(ea[3], eb[3])
        along = a1 - a0
        depth = c1 - c0
        height = max(a.size[2], b.size[2])
        ca = (a0 + a1) / 2.0
        cc = (c0 + c1) / 2.0
        ctr = axis * ca + cross * cc
        merged = OrientedBox(
            center=(float(ctr[0]), float(ctr[1]), height / 2),
            size=(max(along, 0.2), max(depth, 0.3), max(height, 0.4)),
            yaw=yaw,
            device_type=a.device_type if a.device_type != DeviceType.UNKNOWN else b.device_type,
            source=BoxSource.RULE_FIX, confidence=Confidence.MID,
            row_id=a.row_id,
            meta={"merged_from": [a.box_id, b.box_id]},
        )
        # refit to actual points so edges land on surfaces, not on the
        # union of noisy fragment bounds
        refit = fit_box_to_points(scene, merged.center[:2], merged.size, yaw)
        if refit is not None and max(refit.size[:2]) <= max_merge_size:
            refit.source = BoxSource.RULE_FIX
            refit.confidence = Confidence.MID
            refit.row_id = a.row_id
            refit.meta = merged.meta
            return refit
        return merged

    n_absorbed = 0
    changed = True
    while changed and len(boxes) > 1:
        changed = False
        exts = [_extents(b) for b in boxes]
        for i in range(len(boxes)):
            for j in range(i + 1, len(boxes)):
                a0, a1, c0, c1 = exts[i]
                b0, b1, d0, d1 = exts[j]
                area_i = max((a1 - a0) * (c1 - c0), 1e-6)
                area_j = max((b1 - b0) * (d1 - d0), 1e-6)
                a_ov = min(a1, b1) - max(a0, b0)          # along overlap
                c_ov = min(c1, d1) - max(c0, d0)          # cross overlap
                inter = max(a_ov, 0.0) * max(c_ov, 0.0)
                same_spot = inter / min(area_i, area_j) >= overlap_thr
                # front/back complementary halves: full along overlap, cross
                # disjoint but union within one rack depth
                along_frac = a_ov / max(min(a1 - a0, b1 - b0), 1e-6) \
                    if a_ov > 0 else 0.0
                cross_union = max(c1, d1) - min(c0, d0)
                # front/back complementary halves: largely full along overlap,
                # cross extents disjoint-ish (allow up to 35% overlap so
                # real back-projection faces with slight overlap still merge)
                # and the union within one plausible rack depth (<= 2.2m).
                fb_complementary = (along_frac >= 0.8 and c_ov <= 0.35 * cross_union
                                    and cross_union <= max_depth)
                # left/right complementary halves: full cross overlap, along
                # union within one rack width (small along overlap allowed --
                # half-fragments of one device often overlap slightly)
                cross_frac = c_ov / max(min(c1 - c0, d1 - d0), 1e-6) \
                    if c_ov > 0 else 0.0
                along_union = max(a1, b1) - min(a0, b0)
                lr_complementary = (cross_frac >= 0.8 and along_union <= max_width)
                # trusted: detector boxes already split per device, so
                # left/right complements usually mean adjacent devices, not a
                # split one -- only allow same-spot / front-back merges
                use_lr = lr_complementary and not trusted
                if same_spot or fb_complementary or use_lr:
                    boxes[i] = _merge_pair(boxes[i], boxes[j])
                    boxes.pop(j)
                    n_absorbed += 1
                    changed = True
                    break
            if changed:
                break
    return boxes, n_absorbed


def _region_of_box(box: OrientedBox, expand: float) -> tuple[float, float, float, float]:
    c = np.asarray(box.center[:2])
    half = np.asarray(box.size[:2]) / 2.0 + expand
    r = box.rotation[:2, :2]
    corners = np.array([[-1, -1], [1, -1], [1, 1], [-1, 1]]) * half
    world = corners @ r.T + c
    return (float(world[:, 0].min()), float(world[:, 1].min()),
            float(world[:, 0].max()), float(world[:, 1].max()))


def _dbscan_1d(x: np.ndarray, eps: float, min_pts: int) -> np.ndarray:
    """Minimal 1D DBSCAN: chain points via <=eps connectivity, prune small runs."""
    if len(x) == 0:
        return np.zeros(0, dtype=int)
    order = np.argsort(x)
    xs = x[order]
    labels = np.full(len(xs), -1, dtype=int)
    cid = 0
    start = 0
    while start < len(xs):
        end = start
        while end + 1 < len(xs) and xs[end + 1] - xs[end] <= eps:
            end += 1
        if end - start + 1 >= min_pts:
            labels[start:end + 1] = cid
            cid += 1
        start = end + 1
    out = np.full(len(x), -1, dtype=int)
    out[order] = labels
    return out
