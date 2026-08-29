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
                      inlier_frac: float = 0.9) -> OrientedBox | None:
    """Refit an oriented box to the local point support.

    Boundary estimation via 1D occupancy histograms per axis: find the
    contiguous occupied span containing the center. Robust to sparse noise
    while keeping edges tight to the true surface.
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
    zmin, zmax = float(qlo[2]), float(qhi[2])

    new_size = (max(xmax - xmin, 0.15), max(ymax - ymin, 0.15), max(zmax - zmin, 0.2))
    local_center = np.array([(xmin + xmax) / 2, (ymin + ymax) / 2, (zmin + zmax) / 2])
    center = seed.local_to_world(local_center.reshape(1, 3))[0]
    box = OrientedBox(center=tuple(center), size=new_size, yaw=yaw,
                      device_type=DeviceType.RACK)
    coverage = support_fraction(scene, box)
    if coverage < 0.12:
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


def center_field_clusters(scene: Scene, box: OrientedBox,
                          dbscan_eps: float = 0.05,
                          min_pts: int = 10) -> tuple[int, float, np.ndarray]:
    """FoundObj-inspired completeness check for (possibly merged) rack boxes.

    Real racks expose their *side faces* as high-density slabs along the row
    axis (local x). One rack -> 2 side-face peaks (its two ends). A merged
    box containing k racks -> k+1 peaks (shared boundaries also produce a
    density slab from the two adjacent side faces). We count interior
    density peaks of the x-histogram to estimate how many racks are inside.

    Returns (estimated_rack_count, dominant_fraction, labels_placeholder).
    """
    region = _region_of_box(box, expand=0.1)
    pts = scene.points_in_region(region)
    if len(pts) < min_pts:
        return 0, 0.0, np.zeros(0, dtype=int)
    local = box.world_to_local(pts)
    half = np.asarray(box.size) / 2.0
    inside = local[np.all(np.abs(local) <= half + 0.05, axis=1)]
    if len(inside) < min_pts:
        return 0, 0.0, np.zeros(0, dtype=int)
    x = inside[:, 0]
    cell = 0.02
    nb = max(int(box.size[0] / cell), 4)
    hist, edges = np.histogram(x, bins=nb, range=(-half[0], half[0]))
    if hist.max() < 5:
        return 1, 1.0, np.zeros(0, dtype=int)
    # peaks = bins clearly denser than the running background (front/back faces
    # give a uniform base level; side faces spike above it)
    base = np.median(hist)
    thr = max(base * 2.5, 8)
    peak_mask = hist >= thr
    # collapse contiguous peak bins into single peaks
    n_peaks = 0
    prev = False
    for p in peak_mask:
        if p and not prev:
            n_peaks += 1
        prev = p
    # k peaks (including both outer ends) -> k-1 racks
    est_racks = max(n_peaks - 1, 1)
    return est_racks, 1.0 / est_racks, np.zeros(0, dtype=int)


def split_box(scene: Scene, box: OrientedBox, n: int, width_unit: float | None = None) -> list[OrientedBox]:
    """Split a (merged-row) box into n sub-boxes along the row axis."""
    if n < 2:
        return [box]
    L, W, H = box.size
    row_axis = box.rotation[:, 0]  # local x direction in world
    # choose split width
    total = L
    widths = np.full(n, total / n)
    if width_unit and width_unit > 0:
        k = max(1, int(round(total / width_unit)))
        widths = np.full(k, total / k)
        n = k
    center = np.asarray(box.center)
    result = []
    acc = 0.0
    for i in range(n):
        seg_c = center - row_axis * total / 2 + row_axis * (acc + widths[i] / 2)
        result.append(OrientedBox(
            center=tuple(seg_c), size=(widths[i], W, H), yaw=box.yaw,
            device_type=DeviceType.RACK, source=box.source,
            confidence=box.confidence, row_id=box.row_id,
            meta={"split_from": box.box_id},
        ))
        acc += widths[i]
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
