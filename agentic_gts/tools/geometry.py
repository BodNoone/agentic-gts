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


def face_support_fraction(scene: Scene, box: OrientedBox, cell: float = 0.1,
                          dilate: int = 1) -> float:
    """Coverage of the box's BEST-covered face by nearby points.

    Interior-volume support (support_fraction) is structurally biased
    against single-view fragments: a fragment box observes one face, so
    most of its interior is empty and occupancy lands below any reasonable
    threshold even though the observation is perfectly real. Real devices
    always have at least one face backed by points; a floating false
    positive has none.

    Returns the max over the 6 faces of the fraction of that face's cells
    having a point within `dilate` cells (holes in sparse clouds are
    bridged by the dilation).
    """
    region = _region_of_box(box, 0.0)
    pts = scene.points_in_region(region)
    if len(pts) == 0:
        return 0.0
    local = box.world_to_local(pts)
    half = np.asarray(box.size) / 2.0
    m = np.all(np.abs(local) <= half + cell, axis=1)
    inside = local[m]
    if len(inside) < 5:
        return 0.0
    nb = np.maximum((np.asarray(box.size) / cell).astype(int), 1)
    idx = np.clip(((inside + half) / cell).astype(int), 0, nb - 1)
    occ = np.zeros(nb, dtype=bool)
    occ[idx[:, 0], idx[:, 1], idx[:, 2]] = True
    from scipy.ndimage import binary_dilation
    occ_d = binary_dilation(occ, iterations=dilate)
    best = 0.0
    # skip the bottom 0.2m of SIDE faces: floor points sit under every
    # floor-touching box and would otherwise rescue false positives
    z_keep = 2
    for axis in range(3):
        if axis < 2:                      # side faces: +x, -x, +y, -y
            for side in (0, -1):
                face = np.take(occ_d, side, axis=axis)   # (other_axis, z)
                sel = face[:, z_keep:] if face.shape[1] > z_keep else face
                if sel.size:
                    best = max(best, float(sel.mean()))
        else:                            # top face only (bottom excluded)
            face = np.take(occ_d, -1, axis=axis)
            if face.size:
                best = max(best, float(face.mean()))
    return best


def center_field_clusters(scene: Scene, box: OrientedBox,
                          dbscan_eps: float = 0.05,
                          min_pts: int = 10,
                          empty_frac: float = 0.25) -> tuple[int, float, np.ndarray]:
    """How many racks does a (possibly merged) box contain, from the point
    density along the row axis (local x).

    A single rack fills the box's x-extent with a contiguous high-density
    run (its front/back/side surfaces) -- 1 cluster. A box that fused two
    racks with a gap between them shows two dense runs separated by a
    low-density gap -> 2 clusters. Two flush racks with no point gap still
    produce a contiguous run, which the side-face peaks below can catch.

    We first segment the x-histogram into dense runs separated by near-empty
    gaps (the straightforward 'two racks with an aisle between' case). If
    that gives one run (flush racks), fall back to counting side-face density
    peaks. Returns (estimated_rack_count, dominant_fraction, labels_placeholder).
    """
    region = _region_of_box(box, expand=0.03)
    pts = scene.points_in_region(region)
    if len(pts) < min_pts:
        return 0, 0.0, np.zeros(0, dtype=int)
    local = box.world_to_local(pts)
    half = np.asarray(box.size) / 2.0
    # keep points strictly inside the box (a small margin, no relax): an
    # expanded region would sweep in neighbouring racks and create a fake
    # gap in the x histogram, misreading a single rack as two.
    inside = local[np.all(np.abs(local) <= half, axis=1)]
    if len(inside) < min_pts:
        return 0, 0.0, np.zeros(0, dtype=int)
    x = inside[:, 0]
    cell = 0.02
    nb = max(int(box.size[0] / cell), 4)
    hist, edges = np.histogram(x, bins=nb, range=(-half[0], half[0]))
    peak = float(hist.max())
    if peak < 5:
        return 1, 1.0, np.zeros(0, dtype=int)

    # ---- dense-run segmentation: separate by a REAL empty gap ----
    # Two racks fused into one box usually keep an aisle gap between them:
    # a contiguous run of essentially EMPTY bins (near-zero points). A single
    # rack is contiguous in x even if its surface density fluctuates, so we
    # require the gap to be truly empty (each bin < ~4% of the peak) and at
    # least a couple of bins wide to count as a separator.
    peak_f = float(hist.max())
    # A bin is EMPTY only if it has essentially no points (below an absolute
    # small count). A rack's side surface is thin but contiguous -- its sparse
    # bins (~20-45 pts) must stay occupied. A genuine aisle gap between two
    # racks is near zero (<~8 pts) across a couple of bins.
    nonempty = hist > 8.0
    # count contiguous NON-EMPTY runs (a run of >0 nonempty bins)
    runs = 0
    in_run = False
    for v in nonempty:
        if v and not in_run:
            runs += 1
            in_run = True
        elif not v:
            in_run = False
    # longest empty gap (>=2 bins, i.e. >=4cm) confirms a separator
    max_gap = 0
    gap = 0
    for v in nonempty:
        if not v:
            gap += 1
            max_gap = max(max_gap, gap)
        else:
            gap = 0
    if runs >= 2 and max_gap >= 2:
        return min(runs, max(2, int(round(box.size[0] / 0.6)))), \
            1.0 / runs, np.zeros(0, dtype=int)

    # ---- flush racks: fall back to side-face density peaks ----
    base = float(np.median(hist))
    thr = max(base * 2.5, 8)
    peak_mask = hist >= thr
    n_peaks = 0
    prev = False
    for p in peak_mask:
        if p and not prev:
            n_peaks += 1
        prev = p
    est_racks = max(n_peaks - 1, 1)
    return est_racks, 1.0 / est_racks, np.zeros(0, dtype=int)


def split_box(scene: Scene, box: OrientedBox, n: int, width_unit: float | None = None) -> list[OrientedBox]:
    """Split a (merged-row) box into n sub-boxes along the row axis."""
    if n < 2:
        return [box]
    L, W, H = box.size
    row_axis = box.rotation[:, 0]  # local x direction in world
    # choose split width: use the caller's n (the count the geometry actually
    # found) unless it is invalid; width_unit is only a fallback when n<2.
    total = L
    if n is not None and n >= 2:
        k = int(n)
    elif width_unit and width_unit > 0:
        k = max(2, int(round(total / width_unit)))
    else:
        k = 2
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


def merge_box_pair(scene: Scene, a: OrientedBox, b: OrientedBox,
                   snap_to_points: bool = True) -> OrientedBox | None:
    """Merge two boxes into one bounding rack box (faces of the same device).

    Joins the two along the shared row axis: union of their row extents,
    depth/height = the larger of the two, yaw follows the dominant one. If
    both snap to points, refit the union so the edges land on surfaces.
    Returns None if the pair cannot be merged (e.g. no point support).
    """
    # row axis from the larger / lower-index box; align the other to it
    yaw = a.yaw
    axis = np.array([math.cos(yaw), math.sin(yaw)])
    cross = np.array([-math.sin(yaw), math.cos(yaw)])

    def _ext(b: OrientedBox):
        cs = np.asarray(b.corners_2d())
        pa = cs @ axis
        pc = cs @ cross
        return (float(pa.min()), float(pa.max()),
                float(pc.min()), float(pc.max()))
    ea, eb = _ext(a), _ext(b)
    a0 = min(ea[0], eb[0]); a1 = max(ea[1], eb[1])
    c0 = min(ea[2], eb[2]); c1 = max(ea[3], eb[3])
    L = a1 - a0
    W = c1 - c0
    H = max(a.size[2], b.size[2])
    cx, cy = a0 + L / 2, c0 + W / 2
    world_xy = axis * cx + cross * cy
    merged = OrientedBox(
        center=(float(world_xy[0]), float(world_xy[1]), H / 2),
        size=(max(L, 0.2), max(W, 0.3), max(H, 0.4)),
        yaw=yaw, device_type=DeviceType.RACK,
        source=BoxSource.RULE_FIX, confidence=Confidence.MID,
        meta={"merged_from": [a.box_id, b.box_id]},
    )
    if snap_to_points:
        refit = fit_box_to_points(scene, merged.center[:2], merged.size, yaw)
        if refit is not None and max(refit.size[:2]) >= max(merged.size[:2]) * 0.6:
            refit.meta = merged.meta
            return refit
    # fall back to plain union if refinement produced something degenerate
    if support_fraction(scene, merged) < 0.1:
        return None
    return merged


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
