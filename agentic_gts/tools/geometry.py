"""Deterministic geometry tools used by the agent.

All functions return concrete coordinates (the agent never alters geometry
itself; it only *selects* which tool to call and with what discrete args).
"""
from __future__ import annotations

import math

import numpy as np

from agentic_gts.core.models import (BoxSource, Confidence, DeviceType,
                                     OrientedBox, Scene)


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


def _region_of_box(box: OrientedBox, expand: float) -> tuple[float, float, float, float]:
    c = np.asarray(box.center[:2])
    half = np.asarray(box.size[:2]) / 2.0 + expand
    r = box.rotation[:2, :2]
    corners = np.array([[-1, -1], [1, -1], [1, 1], [-1, 1]]) * half
    world = corners @ r.T + c
    return (float(world[:, 0].min()), float(world[:, 1].min()),
            float(world[:, 0].max()), float(world[:, 1].max()))


# ---------- row completion (geometry-only recall fallback) ----------


def _probe_filled(scene: Scene, box: OrientedBox,
                  min_dev_pts: int = 40, min_h: float = 0.5,
                  span_frac: float = 0.4) -> bool:
    """Is there a real DEVICE inside the probe box?

    Three guards, mirroring the grounding fit's:
      - device-band point count (floor texture excluded by the z cut)
      - height: something standing at least min_h tall
      - FOOTPRINT FILL in BOTH horizontal axes: a cabinet fills the
        probe's along AND cross extents (it is a solid footprint);
        a WALL slice -- the classic false positive at row ends, where
        a partition runs past the row's last cabinet -- is thin in
        one axis and fails the span check no matter its orientation.
    """
    region = _region_of_box(box, 0.0)
    pts = scene.points_in_region(region)
    if len(pts) == 0:
        return False
    local = box.world_to_local(pts)
    half = np.asarray(box.size) / 2.0
    inside = local[np.all(np.abs(local) <= half, axis=1)]
    if len(inside) == 0:
        return False
    dev = inside[inside[:, 2] > 0.30]
    if len(dev) < min_dev_pts:
        return False
    if float(dev[:, 2].max()) < min_h:
        return False
    sx = float(np.percentile(dev[:, 0], 95) - np.percentile(dev[:, 0], 5))
    sy = float(np.percentile(dev[:, 1], 95) - np.percentile(dev[:, 1], 5))
    return sx >= span_frac * box.size[0] and sy >= span_frac * box.size[1]


def complete_row_gaps(scene: Scene, width_unit: float = 0.6,
                       cluster_tol: float = 0.35, max_gap_racks: int = 2,
                       min_dev_pts: int = 40) -> list[OrientedBox]:
    """Geometry-only recall fallback: fill missed cabinets along rows.

    In the no-hint flow the VLM grounding is the ONLY box producer, so
    a cabinet it missed (occluded from the nadir view, dim, dropped
    with a poor-quality view) is lost for good -- no downstream stage
    can recover a range that was never grounded. This pass walks the
    FITTED rows' interiors and ends with point-support probes and adds
    a box wherever a real device stands unclaimed: the same job the
    old rules' find_gaps/add_box_at did, now seeded from grounded
    boxes instead of hint input.

    Rows are read off the boxes themselves: the long side (size[0])
    always rides its row's axis (_fit_region_box contract), so members
    are bucketed parallel / perpendicular to the frame yaw and clustered
    on the cross-axis coordinate.

    Returns the added boxes (also appended to scene.boxes). Every fill
    is Confidence.LOW + source ROW_COMPLETION: it is geometric
    evidence, not VLM-confirmed, so it surfaces for human review.
    """
    if not scene.boxes:
        return []
    yaw = float(scene.meta.get("yaw", 0.0) or 0.0)
    d_main = np.array([math.cos(yaw), math.sin(yaw)])
    d_perp = np.array([-d_main[1], d_main[0]])

    def _bucket_axis(b: OrientedBox) -> np.ndarray:
        # long axis (local x) parallel to the frame yaw -> main bucket
        long = b.rotation[:2, 0]
        return d_main if abs(float(long @ d_main)) > 0.9 else d_perp

    # ---- cluster into rows: (axis bucket) then 1D clustering on cross ----
    groups: dict[int, list[int]] = {0: [], 1: []}
    for i, b in enumerate(scene.boxes):
        groups[0 if _bucket_axis(b) is d_main else 1].append(i)

    added: list[OrientedBox] = []

    def _try_fill(axis: np.ndarray, cross: np.ndarray,
                  along_c: float, cross_coord: float, w: float,
                  depth: float, height: float, row_id) -> None:
        world_xy = axis * along_c + cross * cross_coord
        # never duplicate: skip if any existing (or just-added) box
        # already covers the spot
        probe_center = np.array([[world_xy[0], world_xy[1], height / 2.0]])
        for b in scene.boxes:
            if b.contains(probe_center, margin=0.10)[0]:
                return
        cand = OrientedBox(
            center=(float(world_xy[0]), float(world_xy[1]), height / 2.0),
            size=(w * 0.9, depth, height),
            yaw=math.atan2(float(axis[1]), float(axis[0])),
            device_type=DeviceType.RACK, source=BoxSource.ROW_COMPLETION,
            confidence=Confidence.LOW, row_id=row_id,
            meta={"row_completion": True})
        if _probe_filled(scene, cand, min_dev_pts=min_dev_pts):
            scene.boxes.append(cand)
            added.append(cand)

    for bucket in (0, 1):
        idxs = groups[bucket]
        if not idxs:
            continue
        axis = d_main if bucket == 0 else d_perp
        cross = np.array([-axis[1], axis[0]])
        centers = np.asarray([scene.boxes[i].center[:2] for i in idxs])
        cross_vals = centers @ cross
        order = np.argsort(cross_vals)
        rows: list[list[int]] = []
        cur = [order[0]]
        for j in order[1:]:
            if abs(cross_vals[j] - cross_vals[cur[-1]]) > cluster_tol:
                rows.append(cur)
                cur = [j]
            else:
                cur.append(j)
        rows.append(cur)

        for members in rows:
            boxes_m = [scene.boxes[i] for i in members]
            along_vals = [float(np.asarray(b.center[:2]) @ axis)
                          for b in boxes_m]
            mem = sorted(zip(along_vals, boxes_m, members),
                         key=lambda t: t[0])
            depth = float(np.median([b.size[1] for b in boxes_m]))
            height = float(np.median([b.size[2] for b in boxes_m]))
            # per-cabinet width: median of member lengths, clamped --
            # an UNSPLIT row box has one huge size[0] and would walk
            # the ends in giant steps
            w = float(np.clip(np.median([b.size[0] for b in boxes_m]),
                              0.4, 1.2))
            cross_coord = float(np.median(
                [np.asarray(b.center[:2]) @ cross for b in boxes_m]))
            row_ids = {b.row_id for b in boxes_m}
            row_id = row_ids.pop() if len(row_ids) == 1 else None

            # ---- interior gaps between adjacent members ----
            for (la, a, _), (lb, b, _) in zip(mem[:-1], mem[1:]):
                gap = (lb - b.size[0] / 2.0) - (la + a.size[0] / 2.0)
                if gap < 0.5 * w:
                    continue
                n_units = min(int(round(gap / w)), max_gap_racks)
                e_a = la + a.size[0] / 2.0
                for k in range(1, n_units + 1):
                    _try_fill(axis, cross, e_a + gap * k / (n_units + 1),
                              cross_coord, w, depth, height, row_id)

            # ---- row ends: walk outward while devices persist ----
            lo_end = min(la for la, _, _ in mem) - mem[0][1].size[0] / 2.0
            hi_end = max(la for la, _, _ in mem) + max(
                b.size[0] for _, b, _ in mem) / 2.0
            for direction, end in ((-1.0, lo_end), (1.0, hi_end)):
                for k in range(1, max_gap_racks + 1):
                    cand_along = end + direction * (w * (k - 0.5) + 0.01)
                    before = len(added)
                    _try_fill(axis, cross, cand_along, cross_coord,
                              w, depth, height, row_id)
                    if len(added) == before:
                        break   # no support -> stop walking this end
    return added


# ---------- split-seam regularisation ----------


def snap_row_seams(boxes: list[OrientedBox], yaw: float,
                   seam_tol: float = 0.12, vertex_tol: float = 0.12,
                   height_tol: float = 0.05) -> int:
    """Snap the facing edges of adjacent cabinets in a row together.

    A joined row split into single cabinets gets each piece's along-row
    extent from its OWN measured span, so facing edges can sit a few
    centimetres apart (or overlap) and the row reads as DISCONNECTED
    boxes. Wherever two facing edges sit within `seam_tol`, both are
    set to their average along coordinate; the shared edge's CROSS
    vertices merge only where both facing corner pairs are within
    `vertex_tol` (otherwise each side keeps its own cross extent).

    Guards: NEVER seam across a HEIGHT STEP > `height_tol` (the split
    separated different-height cabinets on purpose -- user directive;
    heights never enter the merge beyond this gate), and facing edges
    must overlap laterally by >= half the thinner body (two different
    sub-rows are never snapped). Boxes are assumed to share the row
    frame (`yaw`). Mutates in place; returns the number of seams
    snapped.
    """
    if len(boxes) < 2:
        return 0
    axis = np.array([math.cos(yaw), math.sin(yaw)])
    cross = np.array([-math.sin(yaw), math.cos(yaw)])

    def _span(b: OrientedBox):
        c = np.asarray(b.center, dtype=float)
        a = float(c[:2] @ axis)
        x = float(c[:2] @ cross)
        half = float(b.size[0]) / 2.0
        return a - half, a + half, x, float(b.size[1])

    def _rebuild(b: OrientedBox, lo: float, hi: float,
                 cross_c: float, depth: float) -> None:
        mid = 0.5 * (lo + hi)
        xy = axis * mid + cross * cross_c
        b.center = (float(xy[0]), float(xy[1]), float(b.center[2]))
        b.size = (float(hi - lo), float(depth), float(b.size[2]))

    order = sorted(range(len(boxes)), key=lambda i: _span(boxes[i])[0])
    snapped = 0
    for i, j in zip(order[:-1], order[1:]):
        a, b = boxes[i], boxes[j]
        a_lo, a_hi, a_x, a_d = _span(a)
        b_lo, b_hi, b_x, b_d = _span(b)
        gap = b_lo - a_hi
        # height-step gate: the split separated them on purpose
        a_top = float(a.center[2]) + float(a.size[2]) / 2.0
        b_top = float(b.center[2]) + float(b.size[2]) / 2.0
        if abs(a_top - b_top) > height_tol:
            continue
        if abs(gap) > seam_tol:
            continue
        # the facing edges must overlap laterally, or they are two
        # different sub-rows rather than neighbours
        a_clo, a_chi = a_x - a_d / 2.0, a_x + a_d / 2.0
        b_clo, b_chi = b_x - b_d / 2.0, b_x + b_d / 2.0
        if min(a_chi, b_chi) - max(a_clo, b_clo) < 0.5 * min(a_d, b_d):
            continue
        seam = 0.5 * (a_hi + b_lo)
        # merge the shared edge's cross vertices only when both facing
        # corner pairs are close (see docstring)
        if (abs(a_clo - b_clo) <= vertex_tol
                and abs(a_chi - b_chi) <= vertex_tol):
            new_clo = 0.5 * (a_clo + b_clo)
            new_chi = 0.5 * (a_chi + b_chi)
            cx, d = 0.5 * (new_clo + new_chi), new_chi - new_clo
            _rebuild(a, a_lo, seam, cx, d)
            _rebuild(b, seam, b_hi, cx, d)
        else:
            _rebuild(a, a_lo, seam, a_x, a_d)
            _rebuild(b, seam, b_hi, b_x, b_d)
        snapped += 1
    return snapped


# ---------- final face polish (stageF: mesh-driven thickness snap) ----------

def aabb_gap_xy(a: "OrientedBox", b: "OrientedBox") -> float:
    """Footprint-AABB distance between two boxes (metres; 0 when the
    AABBs overlap)."""
    ca, cb = a.corners_2d(), b.corners_2d()
    dx = max(cb[:, 0].min() - ca[:, 0].max(),
             ca[:, 0].min() - cb[:, 0].max(), 0.0)
    dy = max(cb[:, 1].min() - ca[:, 1].max(),
             ca[:, 1].min() - cb[:, 1].max(), 0.0)
    return float(math.hypot(dx, dy))

# cross-profile histogram resolution
_FACE_BIN = 0.03
# a mesh-sampled face sheet is 2-5cm thick: aggregate this much cross
# extent into one "face strength" sample (single bins split a sheet)
_FACE_BAND = 0.06
# per-face search window, ASYMMETRIC: generous INWARD (the open-door
# error direction is outward inflation -- the true sheet sits inward),
# tight OUTWARD (a small under-measure allowance; a tight outward
# window also keeps a flush wall or a neighbour's sheet from pulling
# the face out)
_FACE_SNAP_IN = 0.35
_FACE_SNAP_OUT = 0.15
# a candidate peak must reach this fraction of the window's strongest
# peak AND of the box profile's global max. The window bar alone would
# let a door-only window snap onto its own plateau; the global bar
# (relative to the box's own strongest sheet) rejects that: an open
# door's cross profile is a low wide plateau, an order of magnitude
# under the face sheets
_FACE_SNAP_TAU = 0.50
_FACE_SNAP_ABS = 0.25
# the END rule's bar is LOWER: it separates the plateau from mask
# bleed (a ~20:1 contrast), while the cross rule separates competing
# peaks (2:1) -- 50% of a fluctuating window max sits inside the
# plateau's own counting noise (an edge bin a couple of sigma low
# vs a bin a couple high) and drops the TRUE edge bin, landing the
# snap a bin inside
_FACE_END_TAU = 0.30
# moves below this are noise -> no-op
_FACE_SNAP_MIN_MOVE = 0.02
# device depth bounds for the snapped result
_FACE_SNAP_MIN_D, _FACE_SNAP_MAX_D = 0.30, 2.50


def snap_box_faces(box: OrientedBox,
                   pts: np.ndarray) -> tuple[OrientedBox, dict]:
    """Snap ONE box's front/back (cross-axis) faces onto the densest
    mesh sheets in its own column (user direction: a final thickness
    polish -- the side-view chain has real holes, wall-masked /
    ladder-dominated / dim renders leave the depth uncorrected, and the
    mesh fallback measures whatever is in the slice, an open door
    included, since door subtraction only exists in the local VIEW
    path).

    The box's cross profile: a face sheet is a tall narrow peak (all
    its points share one cross coordinate), an open door a low wide
    plateau (the swung panel spreads along the cross axis), a wall
    behind a strong but FARTHER peak. Rule: per face, snap to the
    NEAREST peak within an inward-biased window that clears both the
    window-relative and the profile-global strength bars -- nearest
    beats strongest so a flush wall never steals the back face and the
    door plateau never qualifies. Idempotent: a face already on its
    sheet has its peak at distance ~0 and does not move. Returns
    (box, info); the box is unchanged unless info["moved"]."""
    yaw = float(box.yaw)
    axis = np.array([math.cos(yaw), math.sin(yaw)])
    cross = np.array([-math.sin(yaw), math.cos(yaw)])
    c = np.asarray(box.center, dtype=float)
    along_c = float(c[:2] @ axis)
    cross_c = float(c[:2] @ cross)
    half_len = float(box.size[0]) / 2.0
    half_d = float(box.size[1]) / 2.0
    bottom = float(c[2]) - float(box.size[2]) / 2.0
    top = float(c[2]) + float(box.size[2]) / 2.0
    pts = np.asarray(pts, dtype=float)
    if len(pts) < 50:
        return box, {"moved": False, "reason": "too few points"}
    # the box's own column: along-slice + a floor-excluding z band
    # (floor points span the whole cross range and would flatten the
    # face peaks into a uniform background)
    m = ((np.abs(pts[:, :2] @ axis - along_c) <= half_len + 0.05)
         & (pts[:, 2] > bottom + 0.30) & (pts[:, 2] <= top))
    if int(m.sum()) < 50:
        return box, {"moved": False, "reason": "too few profile points"}
    cc = pts[m, :2] @ cross
    prof_lo = cross_c - half_d - _FACE_SNAP_OUT
    prof_hi = cross_c + half_d + _FACE_SNAP_OUT
    # explicit bin COUNT, not arange(stop): arange's ceil drifts in fp
    # and an edge landing a hair below the sheet drops its points
    nb = int(np.floor((prof_hi - prof_lo) / _FACE_BIN)) + 2
    edges = prof_lo + _FACE_BIN * np.arange(nb + 1)
    hist, _ = np.histogram(cc, bins=edges)
    centers = 0.5 * (edges[:-1] + edges[1:])
    k = max(1, int(round(_FACE_BAND / _FACE_BIN)))
    band = np.convolve(hist, np.ones(k, dtype=int), mode="same")
    gmax = int(band.max())
    if gmax <= 0:
        return box, {"moved": False, "reason": "empty profile"}

    def _snap(face_pos: float, win_lo: float, win_hi: float) -> float:
        in_win = (centers >= win_lo) & (centers <= win_hi)
        if not in_win.any():
            return face_pos
        thr = max(_FACE_SNAP_TAU * float(band[in_win].max()),
                  _FACE_SNAP_ABS * gmax)
        is_peak = np.ones(len(band), dtype=bool)
        is_peak[1:-1] = ((band[1:-1] >= band[:-2])
                         & (band[1:-1] >= band[2:]))
        cand = np.where(in_win & is_peak & (band >= thr))[0]
        if not len(cand):
            return face_pos
        d = np.abs(centers[cand] - face_pos)
        return float(centers[cand[int(np.argmin(d))]])

    front, back = cross_c + half_d, cross_c - half_d
    f1 = _snap(front, front - _FACE_SNAP_IN, front + _FACE_SNAP_OUT)
    b1 = _snap(back, back - _FACE_SNAP_OUT, back + _FACE_SNAP_IN)
    mf = abs(f1 - front) >= _FACE_SNAP_MIN_MOVE
    mb = abs(b1 - back) >= _FACE_SNAP_MIN_MOVE
    if not mf:
        f1 = front
    if not mb:
        b1 = back
    if not (mf or mb):
        return box, {"moved": False, "reason": "already on the sheets"}
    depth, old_depth = f1 - b1, 2.0 * half_d
    if not (_FACE_SNAP_MIN_D <= depth <= _FACE_SNAP_MAX_D):
        return box, {"moved": False,
                     "reason": f"snapped depth {depth:.2f} out of bounds"}
    new_cross_c = 0.5 * (f1 + b1)
    dxy = cross * (new_cross_c - cross_c)
    new_box = OrientedBox(
        center=(float(c[0] + dxy[0]), float(c[1] + dxy[1]), float(c[2])),
        size=(float(box.size[0]), float(depth), float(box.size[2])),
        yaw=box.yaw, box_id=box.box_id, device_type=box.device_type,
        source=box.source, confidence=box.confidence, row_id=box.row_id,
        meta=box.meta)
    info = {"moved": True,
            "depth": [round(old_depth, 3), round(depth, 3)],
            "front": [round(front, 3), round(f1, 3)],
            "back": [round(back, 3), round(b1, 3)]}
    return new_box, info


def _row_continues(boxes, box, side: int) -> bool:
    """True when another box continues the row past `box`'s `side` end
    (+1 = right along the row axis, -1 = left): similar direction,
    overlapping cross range, along-adjacent (a seam or a small gap).
    User direction: joined rows are fine-tuned ONLY at their outmost
    ends -- internal seams stay where snap_row_seams put them."""
    yaw = float(box.yaw)
    axis = np.array([math.cos(yaw), math.sin(yaw)])
    cross = np.array([-math.sin(yaw), math.cos(yaw)])
    c = np.asarray(box.center, dtype=float)[:2]
    along_c = float(c @ axis)
    half_len = float(box.size[0]) / 2.0
    half_d = float(box.size[1]) / 2.0
    for b in boxes:
        if b.box_id == box.box_id:
            continue
        if abs(math.remainder(float(b.yaw) - yaw, math.pi)) \
                > math.radians(20.0):
            continue
        cs = b.corners_2d()
        ba = cs @ axis
        bc = cs @ cross
        # cross ranges must overlap (same row line, not the facing row
        # across the aisle): at least half the thinner extent
        if (min(bc.max(), half_d) - max(bc.min(), -half_d)
                <= 0.5 * min(bc.max() - bc.min(), 2.0 * half_d)):
            continue
        gap = ((ba.min() - (along_c + half_len)) if side > 0
               else ((along_c - half_len) - ba.max()))
        if -0.20 <= gap <= 0.40:
            return True
    return False


def snap_box_ends(box: OrientedBox, pts: np.ndarray,
                   snap_left: bool = True,
                   snap_right: bool = True) -> tuple[OrientedBox, dict]:
    """Snap the FREE row ends (along-axis faces) of one box onto the
    plateau edges of its own column (user direction: joined rows are
    fine-tuned only at their outmost ends -- the caller gates the
    faces; internal seams belong to snap_row_seams).

    The along profile of a row is a PLATEAU, not twin peaks: the
    front/back sheets run its whole length, so there is no peak to
    snap to -- the end face snaps to the OUTER EDGE of the qualifying
    run nearest the current face. The run CONTAINING the face when it
    sits on mass (extend/trim within the connected structure); when
    the face sits past the mass (bleed, or a gap between structures)
    the run farthest from the box centre wins -- an over-inflated or
    gap-straddling face trims/extends to the OUTER structure and never
    amputates an inner one (a union box spanning row + gap + clump
    keeps its clump; splitting it is stageC's job). The edge must be
    VISIBLE: a run reaching the window's outer boundary means the
    structure continues past the window -- the true end is beyond
    reach and the face does not move."""
    if not (snap_left or snap_right):
        return box, {"moved": False, "reason": "no free ends"}
    yaw = float(box.yaw)
    axis = np.array([math.cos(yaw), math.sin(yaw)])
    cross = np.array([-math.sin(yaw), math.cos(yaw)])
    c = np.asarray(box.center, dtype=float)
    along_c = float(c[:2] @ axis)
    cross_c = float(c[:2] @ cross)
    half_len = float(box.size[0]) / 2.0
    half_d = float(box.size[1]) / 2.0
    bottom = float(c[2]) - float(box.size[2]) / 2.0
    top = float(c[2]) + float(box.size[2]) / 2.0
    pts = np.asarray(pts, dtype=float)
    if len(pts) < 50:
        return box, {"moved": False, "reason": "too few points"}
    # the box's own row line: cross-slice + the same floor-excluding
    # z band as the face snap
    m = ((np.abs(pts[:, :2] @ cross - cross_c) <= half_d + 0.05)
         & (pts[:, 2] > bottom + 0.30) & (pts[:, 2] <= top))
    if int(m.sum()) < 50:
        return box, {"moved": False, "reason": "too few profile points"}
    aa = pts[m, :2] @ axis
    prof_lo = along_c - half_len - _FACE_SNAP_OUT
    prof_hi = along_c + half_len + _FACE_SNAP_OUT
    nb = int(np.floor((prof_hi - prof_lo) / _FACE_BIN)) + 2
    edges = prof_lo + _FACE_BIN * np.arange(nb + 1)
    hist, _ = np.histogram(aa, bins=edges)
    centers = 0.5 * (edges[:-1] + edges[1:])
    # RAW counts, not the sliding band: the end rule detects a
    # PLATEAU EDGE (where density drops), and the band convolve HALVES
    # the edge bin's strength (it only sees the interior side) -- at
    # the 50% window bar the true edge bin is borderline and the snap
    # lands a bin inside. The band exists for the cross faces' thin
    # sheets; the plateau needs no aggregation.
    gmax = int(hist.max())
    if gmax <= 0:
        return box, {"moved": False, "reason": "empty profile"}

    def _runs(q):
        idx = np.where(q)[0]
        if not len(idx):
            return []
        runs, start, prev = [], idx[0], idx[0]
        for j in idx[1:]:
            if j == prev + 1:
                prev = j
            else:
                runs.append((start, prev))
                start = prev = j
        runs.append((start, prev))
        return runs

    def _snap_end(face_pos: float, outer_sign: int) -> float:
        # outer_sign=+1: the right face (outward = +along); -1: left
        if outer_sign > 0:
            win_lo = face_pos - _FACE_SNAP_IN
            win_hi = face_pos + _FACE_SNAP_OUT
        else:
            win_lo = face_pos - _FACE_SNAP_OUT
            win_hi = face_pos + _FACE_SNAP_IN
        in_win = (centers >= win_lo) & (centers <= win_hi)
        if not in_win.any():
            return face_pos
        thr = max(_FACE_END_TAU * float(hist[in_win].max()),
                  _FACE_SNAP_ABS * gmax)
        runs = _runs(in_win & (hist >= thr))
        if not runs:
            return face_pos
        fi = int(np.argmin(np.abs(centers - face_pos)))
        run = next((r for r in runs if r[0] <= fi <= r[1]), None)
        if run is None:
            edge_i = 1 if outer_sign > 0 else 0
            run = max(runs, key=lambda r: abs(centers[r[edge_i]] - along_c))
        edge = run[1] if outer_sign > 0 else run[0]
        # the edge must be VISIBLE within the window: a run reaching
        # the outer boundary means the structure continues past it
        if outer_sign > 0 and centers[edge] >= win_hi - _FACE_BIN:
            return face_pos
        if outer_sign < 0 and centers[edge] <= win_lo + _FACE_BIN:
            return face_pos
        # SUB-BIN edge: the outermost actual point in the run's edge
        # bin. A bin-centre target quantises to +/-3cm, and the
        # PARTIAL edge bin (the true edge lands mid-bin) can fail the
        # threshold, which would land the snap a full bin early -- the
        # point-level edge keeps both cases sub-centimetre
        sel = (aa >= edges[edge]) & (aa <= edges[edge + 1])
        if not sel.any():
            return face_pos
        return float(aa[sel].max() if outer_sign > 0 else aa[sel].min())

    right, left = along_c + half_len, along_c - half_len
    r1 = _snap_end(right, +1) if snap_right else right
    l1 = _snap_end(left, -1) if snap_left else left
    mr = abs(r1 - right) >= _FACE_SNAP_MIN_MOVE
    ml = abs(l1 - left) >= _FACE_SNAP_MIN_MOVE
    if not mr:
        r1 = right
    if not ml:
        l1 = left
    if not (mr or ml):
        return box, {"moved": False,
                     "reason": "already on the plateau edges"}
    length, old_length = r1 - l1, 2.0 * half_len
    if length < 0.30:
        return box, {"moved": False,
                     "reason": f"snapped length {length:.2f} too short"}
    new_along_c = 0.5 * (r1 + l1)
    dxy = axis * (new_along_c - along_c)
    new_box = OrientedBox(
        center=(float(c[0] + dxy[0]), float(c[1] + dxy[1]), float(c[2])),
        size=(float(length), float(box.size[1]), float(box.size[2])),
        yaw=box.yaw, box_id=box.box_id, device_type=box.device_type,
        source=box.source, confidence=box.confidence, row_id=box.row_id,
        meta=box.meta)
    info = {"moved": True,
            "length": [round(old_length, 3), round(length, 3)],
            "left": [round(left, 3), round(l1, 3)],
            "right": [round(right, 3), round(r1, 3)]}
    return new_box, info


def _own_points(pts: np.ndarray, boxes, self_box,
                reach: float = 1.0) -> np.ndarray:
    """Mask over pts: True = the point is NOT inside any OTHER near
    box. Each box's stageF profile is built from its OWN mass only
    (user report: a big box over-covering a small device must not
    measure the device's points -- with them in the profile the face
    snaps onto the DEVICE instead of the row's own sheet; this also
    closes the cross-box peak-steal hole noted at the face snap's
    introduction). Boxes farther than `reach` cannot own points in
    this box's column and are skipped for cost."""
    near = [b for b in boxes
            if b.box_id != self_box.box_id
            and aabb_gap_xy(self_box, b) <= reach]
    if not near:
        return np.ones(len(pts), dtype=bool)
    inside = np.zeros(len(pts), dtype=bool)
    for b in near:
        inside |= b.contains(pts)
    return ~inside


def _nested_trim(box: "OrientedBox", boxes, pts: np.ndarray):
    """Cut an over-extended face back to a nested smaller box's near
    boundary (user report: ONE VLM rect covering a row + the small
    device beside it -- the fit/split left a big box still covering the
    device while the device's own small box also survived, two
    detections at the device). GUARD: the cut region's points must be
    >= 80% the SMALL box's own -- a phantom small box inside a correct
    big box fails this (the cut would remove the big box's own mass)
    and is left alone. Returns (box, info); unchanged unless
    info["moved"]."""
    pts = np.asarray(pts, dtype=float)
    if len(pts) < 50 or len(boxes) < 2:
        return box, None
    for s in boxes:
        if s.box_id == box.box_id:
            continue
        if float(s.containment_2d(box)) < 0.50:
            continue
        a_s = float(np.prod(np.asarray(s.size, dtype=float)[:2]))
        a_b = float(np.prod(np.asarray(box.size, dtype=float)[:2]))
        if a_s > 0.5 * a_b:
            continue            # only the clearly smaller box wins
        yaw = float(box.yaw)
        axis = np.array([math.cos(yaw), math.sin(yaw)])
        cross = np.array([-math.sin(yaw), math.cos(yaw)])
        c = np.asarray(box.center, dtype=float)
        sc = np.asarray(s.center, dtype=float)
        d2 = (sc - c)[:2]
        dalong, dcross = float(d2 @ axis), float(d2 @ cross)
        cs = s.corners_2d()
        sa, scross = cs @ axis, cs @ cross
        along_c = float(c[:2] @ axis)
        cross_c = float(c[:2] @ cross)
        half_len = float(box.size[0]) / 2.0
        half_d = float(box.size[1]) / 2.0
        bottom = float(c[2]) - float(box.size[2]) / 2.0
        top = float(c[2]) + float(box.size[2]) / 2.0
        along_all = pts[:, :2] @ axis
        cross_all = pts[:, :2] @ cross
        band = (pts[:, 2] > bottom + 0.30) & (pts[:, 2] <= top)
        if abs(dcross) >= abs(dalong):
            near_edge = (float(scross.min()) if dcross > 0
                         else float(scross.max()))
            old_face = cross_c + (half_d if dcross > 0 else -half_d)
            lo, hi = sorted((near_edge, old_face))
            cut = (band & (cross_all >= lo) & (cross_all <= hi)
                   & (np.abs(along_all - along_c) <= half_len + 0.05))
            side = "front" if dcross > 0 else "back"
        else:
            near_edge = (float(sa.min()) if dalong > 0
                         else float(sa.max()))
            old_face = along_c + (half_len if dalong > 0 else -half_len)
            lo, hi = sorted((near_edge, old_face))
            cut = (band & (along_all >= lo) & (along_all <= hi)
                   & (np.abs(cross_all - cross_c) <= half_d + 0.05))
            side = "right" if dalong > 0 else "left"
        if int(cut.sum()) < 20:
            continue
        if float(s.contains(pts[cut]).mean()) < 0.80:
            continue            # the cut region is not the small box's
        if side in ("front", "back"):
            front, back = cross_c + half_d, cross_c - half_d
            if side == "front":
                front = near_edge
            else:
                back = near_edge
            new_depth = front - back
            if not (0.30 <= new_depth <= 2.50):
                continue
            dxy = cross * (0.5 * (front + back) - cross_c)
            new_box = OrientedBox(
                center=(float(c[0] + dxy[0]), float(c[1] + dxy[1]),
                        float(c[2])),
                size=(float(box.size[0]), float(new_depth),
                      float(box.size[2])),
                yaw=box.yaw, box_id=box.box_id,
                device_type=box.device_type, source=box.source,
                confidence=box.confidence, row_id=box.row_id,
                meta=box.meta)
        else:
            right, left = along_c + half_len, along_c - half_len
            if side == "right":
                right = near_edge
            else:
                left = near_edge
            new_len = right - left
            if new_len < 0.30:
                continue
            dxy = axis * (0.5 * (right + left) - along_c)
            new_box = OrientedBox(
                center=(float(c[0] + dxy[0]), float(c[1] + dxy[1]),
                        float(c[2])),
                size=(float(new_len), float(box.size[1]),
                      float(box.size[2])),
                yaw=box.yaw, box_id=box.box_id,
                device_type=box.device_type, source=box.source,
                confidence=box.confidence, row_id=box.row_id,
                meta=box.meta)
        info = {"moved": True, "side": side,
                "cut": [round(old_face, 3), round(near_edge, 3)],
                "small_box": s.box_id}
        return new_box, info
    return box, None


def snap_faces_to_mesh(scene: Scene) -> int:
    """stageF: per box -- (1) NESTED TRIM, cutting an over-extended
    face back to a nested smaller box's near boundary (guarded: the cut
    region must be the small box's own mass); (2) the FREE row ends
    (the outmost ends of a joined row -- internal seams are left to
    snap_row_seams, user direction) and (3) the front/back faces snap
    onto the densest mesh sheets of the box's OWN column (other boxes'
    points never feed the profile). Identity and meta preserved;
    adjustments recorded in meta['nested_trim'] / meta['end_snap'] /
    meta['face_snap']. Returns the number of boxes adjusted."""
    n = 0
    snapshot = list(scene.boxes)
    pts = np.asarray(scene.points, dtype=float)
    for i, b in enumerate(snapshot):
        try:
            cur = b
            tb, tinfo = _nested_trim(cur, scene.boxes, pts)
            if tinfo is not None:
                cur = tb
                scene.boxes[i] = cur
                cur.meta["nested_trim"] = tinfo
                n += 1
                print(f"[stageF] {cur.box_id[:6]} nested trim: "
                      f"{tinfo['side']} face {tinfo['cut'][0]:.2f} -> "
                      f"{tinfo['cut'][1]:.2f} (cut back to "
                      f"{tinfo['small_box'][:6]})")
            own = _own_points(pts, scene.boxes, cur)
            nb, einfo = snap_box_ends(
                cur, pts[own],
                snap_left=not _row_continues(snapshot, b, -1),
                snap_right=not _row_continues(snapshot, b, +1))
            if einfo.get("moved"):
                nb.meta["end_snap"] = einfo
                print(f"[stageF] {nb.box_id[:6]} end snap: length "
                      f"{einfo['length'][0]:.2f} -> {einfo['length'][1]:.2f}m "
                      f"(left {einfo['left'][0]:.2f} -> {einfo['left'][1]:.2f}, "
                      f"right {einfo['right'][0]:.2f} -> "
                      f"{einfo['right'][1]:.2f})")
            nb2, finfo = snap_box_faces(nb, pts[own])
            if finfo.get("moved"):
                nb2.meta["face_snap"] = finfo
                print(f"[stageF] {nb2.box_id[:6]} face snap: depth "
                      f"{finfo['depth'][0]:.2f} -> {finfo['depth'][1]:.2f}m "
                      f"(front {finfo['front'][0]:.2f} -> "
                      f"{finfo['front'][1]:.2f}, back "
                      f"{finfo['back'][0]:.2f} -> {finfo['back'][1]:.2f})")
            if einfo.get("moved") or finfo.get("moved"):
                scene.boxes[i] = nb2
                n += 1
        except Exception as e:
            print(f"[stageF] {b.box_id[:6]} snap failed "
                  f"({type(e).__name__}: {e})")
    return n
