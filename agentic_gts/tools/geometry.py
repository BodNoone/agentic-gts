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


# ---------- structural geometry filter (wall/pillar post-processing) ----------

# device height ceiling: no server rack / IT cabinet / CRAC exceeds
# this; walls and pillars run to the ceiling (user observation: the
# heights are clearly different). 42U racks are ~2.0m, tall network
# racks ~2.4m, CRACs ~2.3m -- 2.8 is a generous ceiling
_STRUCT_MAX_DEVICE_H = 2.80
# the vertical band above the box top to scan for cable trays: real
# device rows in a datacenter ALWAYS have cable trays / ladders
# running above them (typically 0.3-0.5m above rack tops); walls and
# pillars have nothing but air up to the ceiling (user observation:
# local views of walls/pillars show no overhead cable infrastructure)
_STRUCT_TRAY_BAND = (0.20, 0.80)
# minimum mesh points in the tray band above the box to count as
# "has overhead infrastructure"
_STRUCT_TRAY_MIN_PTS = 30


def has_overhead_structure(scene: Scene, box: OrientedBox,
                           band: tuple = _STRUCT_TRAY_BAND,
                           min_pts: int = _STRUCT_TRAY_MIN_PTS) -> bool:
    """True when the box has cable-tray/ladder mesh points in the
    vertical band above its top, within its (slightly padded)
    footprint -- real device rows always have overhead cable
    infrastructure; walls and pillars do not."""
    pts = np.asarray(scene.points, dtype=float)
    if not len(pts):
        return False
    top = float(box.center[2]) + float(box.size[2]) / 2.0
    m = ((pts[:, 2] > top + band[0]) & (pts[:, 2] < top + band[1]))
    if int(m.sum()) < min_pts:
        return False
    sel = pts[m]
    # XY overlap with the box footprint (padded 0.3m: trays may sit
    # slightly offset from the rack row centre line)
    half = np.asarray(box.size, dtype=float)[:2] / 2.0 + 0.30
    local = box.world_to_local(sel)
    inside = np.all(np.abs(local[:, :2]) <= half, axis=1)
    return int(inside.sum()) >= min_pts


def filter_structural_by_geometry(scene: Scene) -> int:
    """Post-processing wall/pillar filter (user direction: the VLM
    structural class was reverted; walls and pillars are filtered by
    GEOMETRY instead). A box is marked LOW when BOTH signals fire:

    1. HEIGHT anomaly: the box is taller than any real device
       (> 2.80m -- walls and pillars run to the ceiling; server racks
       are 1.8-2.5m);
    2. NO overhead cable infrastructure: the vertical band above the
       box's top, within its footprint, has no cable-tray/ladder mesh
       points (real device rows ALWAYS have trays above them; walls
       and pillars have nothing but air).

    Both signals are required: a tall rack WITH trays above it passes
    (has infrastructure); a short box without trays passes (not
    tall). Only tall boxes with nothing above them are marked LOW --
    the conservative direction. Returns the number of boxes flagged."""
    n = 0
    for b in scene.boxes:
        h = float(b.size[2])
        tall = h > _STRUCT_MAX_DEVICE_H
        if not tall:
            continue
        if has_overhead_structure(scene, b):
            continue
        b.confidence = Confidence.LOW
        b.meta["structural_geom"] = {
            "height": round(h, 2), "overhead_pts": 0,
            "reason": "tall box with no overhead cable infrastructure "
                      "(wall/pillar geometric signature)"}
        n += 1
        print(f"[stageF] {b.box_id[:6]} structural geometry filter: "
              f"height {h:.2f}m > {_STRUCT_MAX_DEVICE_H}m, no overhead "
              f"tray points -> LOW (wall/pillar)")
    return n


_LOW_FURNITURE_MAX_H = 1.20
_LOW_FURNITURE_MIN_FOOTPRINT = 0.45
_LOW_FURNITURE_SOLID_SUPPORT = 0.25


def filter_low_furniture_by_geometry(scene: Scene) -> int:
    """Mark likely tables/benches/chairs for removal.

    This conservative filter only targets low, unrefined boxes without
    overhead device infrastructure. SAM/VLM-adopted pieces are retained for
    review because they have stronger semantic evidence than a raw global
    grounding box.
    """
    n = 0
    for b in scene.boxes:
        if b.source == BoxSource.AGENT_FIX:
            continue
        h = float(b.size[2])
        w, d = float(b.size[0]), float(b.size[1])
        if (h > _LOW_FURNITURE_MAX_H
                or min(w, d) < _LOW_FURNITURE_MIN_FOOTPRINT):
            continue
        if has_overhead_structure(scene, b):
            continue
        # Low battery cabinets and similar equipment can lack overhead
        # trays. Dense 3D surface support distinguishes their cabinet-like
        # body from sparse-legged furniture such as tables and chairs.
        if support_fraction(scene, b) >= _LOW_FURNITURE_SOLID_SUPPORT:
            continue
        b.confidence = Confidence.LOW
        b.meta["low_furniture_geom"] = {
            "height": round(h, 2),
            "footprint": [round(w, 2), round(d, 2)],
            "reason": "low box without overhead device infrastructure "
                      "(table/bench/chair signature)",
        }
        n += 1
    return n


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
# per-face search windows, ASYMMETRIC. Inward search scales with current
# depth (2/3 of size[1]); the strongest qualifying sheet wins, with
# distance as a tie-break. OUTWARD stays tight (a small under-measure
# allowance; also keeps a flush wall or neighbour's sheet from pulling
# the face out). END faces:
# the inflation source is mask BLEED, not doors -- the window stays
# bleed-sized
_FACE_SNAP_IN_FRAC = 2.0 / 3.0
_END_SNAP_IN = 0.35
_FACE_SNAP_OUT = 0.15
# a candidate peak must reach this fraction of the window's strongest
# peak AND of the box profile's global max. The window bar alone would
# let a door-only window snap onto its own plateau; the global bar
# (relative to the box's own strongest sheet) rejects that: an open
# door's cross profile is a low wide plateau, an order of magnitude
# under the face sheets
_FACE_SNAP_TAU = 0.50
_FACE_SNAP_ABS = 0.25
_INWARD_SNAP_WINDOW_FRAC = 0.55
_INWARD_SNAP_GLOBAL_FRAC = 0.25
_FACE_EMPTY_FACE_FRAC = 0.12
# the END rule's bar is LOWER: it separates the plateau from mask
# bleed (a ~20:1 contrast), while the cross rule separates competing
# peaks (2:1) -- 50% of a fluctuating window max sits inside the
# plateau's own counting noise (an edge bin a couple of sigma low
# vs one a couple high) and drops the TRUE edge bin, landing the
# snap a bin inside
_FACE_END_TAU = 0.30
# END inward search range: 1/3 of the box's along extent, capped at 3m.
_END_SNAP_IN_FRAC = 1.0 / 3.0
_END_SNAP_IN_MAX = 3.0
# moves below this are noise -> no-op
_FACE_SNAP_MIN_MOVE = 0.02
# device depth bounds for the snapped result
_FACE_SNAP_MIN_D, _FACE_SNAP_MAX_D = 0.30, 2.50


def _height_boundary_ok(box: OrientedBox, pts: np.ndarray,
                        axis: np.ndarray, cross: np.ndarray,
                        along: float, cross_pos: float,
                        end_profile: bool = False) -> bool:
    """Reject candidate sheets dominated by overhead ladder geometry."""
    c = np.asarray(box.center, dtype=float)
    half = np.asarray(box.size, dtype=float) / 2.0
    p = np.asarray(pts, dtype=float)
    along_p = p[:, :2] @ axis
    cross_p = p[:, :2] @ cross
    m = ((np.abs(along_p - along) <= (max(_FACE_BAND, 0.06)
                                     if end_profile
                                     else max(0.30, min(0.80,
                                                        0.25 * box.size[0]))))
         & (np.abs(cross_p - cross_pos) <= (half[1] + 0.05
                                            if end_profile
                                            else max(_FACE_BAND, 0.06)))
         & (p[:, 2] > c[2] - half[2] + 0.30))
    z = p[m, 2]
    if len(z) < 12:
        return False
    top = float(c[2] + half[2])
    body = int(np.count_nonzero(z <= top + 0.05))
    high = z[z > top + 0.10]
    overhead = int(len(high))
    if body < 12:
        return False
    # A long ladder can occupy only a small fraction of a low device's
    # candidate slice, so a majority-only test is too weak. A coherent
    # group of high points is enough to reject the candidate, even when the
    # overall box height would not change.
    if overhead >= 8 and float(np.percentile(high, 95)) > top + 0.15:
        return False
    # A candidate whose evidence is mostly above the current device top is
    # also more likely a cable ladder/tray than a device boundary.
    return overhead <= max(8, int(0.50 * body))


def _trim_sparse_top(box: OrientedBox, pts: np.ndarray):
    """Conservatively trim a low-density over-height tail downward only."""
    c = np.asarray(box.center, dtype=float)
    half = np.asarray(box.size, dtype=float) / 2.0
    local = box.world_to_local(np.asarray(pts, dtype=float))
    m = (np.all(np.abs(local[:, :2]) <= half[:2] + 0.05, axis=1)
         & (local[:, 2] > -half[2] + 0.30))
    z = local[m, 2] + c[2]
    if len(z) < 80:
        return box, {"moved": False, "reason": "too few top points"}
    lo, hi = float(c[2] - half[2] + 0.30), float(c[2] + half[2])
    edges = np.arange(lo, hi + 0.05, 0.05)
    if len(edges) < 4:
        return box, {"moved": False, "reason": "short height range"}
    hist, _ = np.histogram(z, bins=edges)
    peak = int(hist.max())
    if peak <= 0:
        return box, {"moved": False, "reason": "empty top profile"}
    # Find the highest density-supported body bin. A very sparse tail above
    # it is treated as reconstruction haze/overhead structure, never as a
    # reason to grow the box.
    supported = np.where(hist >= max(5, 0.15 * peak))[0]
    # Point counts alone let a dense ladder on one side define the top.
    # Require upper slices to cover the same footprint cells as the body.
    column = local[m]
    cells = np.clip(((column[:, :2] + half[:2]) /
                     (2.0 * half[:2]) * 4).astype(int), 0, 3)
    keys = cells[:, 0] * 4 + cells[:, 1]
    reference = ((z >= lo + 0.20 * (hi - lo))
                 & (z <= lo + 0.55 * (hi - lo)))
    reference_counts = np.bincount(keys[reference], minlength=16)
    body_cells = reference_counts >= 3
    if int(body_cells.sum()) < 4:
        return box, {"moved": False, "reason": "insufficient body coverage"}
    coverage = []
    for i in supported:
        near = (z >= edges[i] - 0.05) & (z <= edges[i + 1] + 0.05)
        counts = np.bincount(keys[near], minlength=16)
        coverage.append(float(((counts >= 3) & body_cells).sum())
                        / int(body_cells.sum()))
    supported = supported[np.asarray(coverage) >= 0.65]
    if not len(supported):
        return box, {"moved": False, "reason": "no supported top"}
    new_top = float(edges[int(supported[-1]) + 1])
    old_top = float(c[2] + half[2])
    if old_top - new_top < 0.15:
        return box, {"moved": False, "reason": "top tail not significant"}
    new_h = new_top - float(c[2] - half[2])
    if new_h < 0.30:
        return box, {"moved": False, "reason": "trimmed height too short"}
    new_box = OrientedBox(
        center=(float(c[0]), float(c[1]),
                float(c[2] - (old_top - new_top) / 2.0)),
        size=(float(box.size[0]), float(box.size[1]), float(new_h)),
        yaw=box.yaw, box_id=box.box_id, device_type=box.device_type,
        source=box.source, confidence=box.confidence, row_id=box.row_id,
        meta=box.meta)
    return new_box, {"moved": True,
                     "height": [round(float(box.size[2]), 3),
                                round(float(new_h), 3)],
                     "top": [round(old_top, 3), round(new_top, 3)]}


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
    behind a strong but FARTHER peak. Rule: per face, search the
    inward window FIRST and accept its nearest qualifying peak; only
    when no inward peak qualifies is the tight outward window tried.
    This shrink-first order prevents a nearby weak exterior structure
    from stealing a face that has a valid device sheet inside. The
    outward fallback is retained for genuinely under-measured boxes.
    Idempotent: a face already on its sheet has its peak at distance ~0
    and does not move. Returns
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

    debug = []

    def _peaks(win_lo: float, win_hi: float, side: int,
               inward: bool = False) -> np.ndarray:
        # Keep each face on its own side of the box centre. Without this
        # guard, an under-measured front face can see the back sheet in its
        # large inward window and produce an invalid negative depth.
        in_win = ((centers >= win_lo) & (centers <= win_hi)
                  & ((centers >= cross_c) if side > 0
                     else (centers <= cross_c)))
        if not in_win.any():
            return np.empty(0, dtype=np.int64)
        if inward:
            thr = max(_INWARD_SNAP_WINDOW_FRAC * float(band[in_win].max()),
                      _INWARD_SNAP_GLOBAL_FRAC * gmax)
        else:
            thr = max(_FACE_SNAP_TAU * float(band[in_win].max()),
                      _FACE_SNAP_ABS * gmax)
        is_peak = np.ones(len(band), dtype=bool)
        is_peak[1:-1] = ((band[1:-1] >= band[:-2])
                         & (band[1:-1] >= band[2:]))
        return np.where(in_win & is_peak & (band >= thr))[0]

    def _snap(face_pos: float, inward: tuple[float, float],
              outward: tuple[float, float], side: int) -> float:
        # Strict shrink-first: a valid inward sheet wins even if an
        # exterior peak is closer to the current face.
        face_zone = np.abs(centers - face_pos) <= _FACE_BAND
        current_face_density = (float(band[face_zone].max())
                                if face_zone.any() else 0.0)
        empty_face = current_face_density <= _FACE_EMPTY_FACE_FRAC * gmax
        cand = _peaks(*inward, side, inward=True)
        exterior = _peaks(*outward, side)
        near_surface = np.concatenate([cand, exterior])
        side_peak = (float(band[near_surface].max())
                     if len(near_surface) else 0.0)
        near_surface = near_surface[
            (np.abs(centers[near_surface] - face_pos) <= _FACE_BIN)
            & (band[near_surface] >= _INWARD_SNAP_WINDOW_FRAC * side_peak)]
        is_inward = bool(len(cand))
        searched = "inward"
        direction_reason = "inward_surface"
        if len(near_surface):
            # A supported surface already at the old face takes precedence
            # over denser internal shelves/panels.
            cand = near_surface
            is_inward = False
            searched = "keep"
            direction_reason = "existing_surface_supported"
        elif len(exterior):
            exterior = exterior[
                (centers[exterior] - face_pos) * side > _FACE_SNAP_MIN_MOVE]
            inner_peak = float(band[cand].max()) if len(cand) else 0.0
            if (len(exterior) and (not len(cand)
                    or float(band[exterior].max()) >= 1.25 * inner_peak)):
                cand = exterior
                is_inward = False
                searched = "outward"
                direction_reason = "stronger_exterior_surface"
        if not len(cand):
            cand = exterior
            is_inward = False
            searched = "outward"
        rec = {
            "face": "front" if side > 0 else "back",
            "old": round(float(face_pos), 3),
            "inward_range": [round(float(inward[0]), 3),
                             round(float(inward[1]), 3)],
            "outward_range": [round(float(outward[0]), 3),
                              round(float(outward[1]), 3)],
            "profile_points": int(m.sum()),
            "gmax": int(gmax),
            "face_density": round(current_face_density, 1),
            "empty_face": bool(empty_face),
            "searched": searched,
            "direction_reason": direction_reason,
            "candidate_count": int(len(cand)),
        }
        if len(cand):
            d = np.abs(centers[cand] - face_pos)
            if is_inward:
                peak = band[cand]
                strongest = np.flatnonzero(peak == peak.max())
                chosen = int(cand[strongest[int(np.argmin(d[strongest]))]])
            else:
                chosen = int(cand[int(np.argmin(d))])
            rec["candidate_peak"] = int(band[chosen])
            rec["candidate"] = round(float(centers[chosen]), 3)
            if searched == "keep":
                rec["accepted"] = False
                rec["reason"] = "existing_surface_supported"
                debug.append(rec)
                return face_pos
            # Inward candidates are already inside the current box and
            # identify its device surface. Overhead geometry must not veto
            # this recovery path; the height guard is for outward growth.
            if not is_inward and not _height_boundary_ok(
                    box, pts, axis, cross, along_c,
                    float(centers[chosen])):
                rec["accepted"] = False
                rec["reason"] = "outward_height_guard"
                debug.append(rec)
                return face_pos
            rec["accepted"] = True
            debug.append(rec)
            return float(centers[chosen])
        rec["accepted"] = False
        rec["reason"] = "no_qualifying_peak"
        debug.append(rec)
        return face_pos

    front, back = cross_c + half_d, cross_c - half_d
    inward_range = float(box.size[1]) * _FACE_SNAP_IN_FRAC
    f1 = _snap(front,
               (front - inward_range, front),
               (front, front + _FACE_SNAP_OUT), +1)
    b1 = _snap(back,
               (back, back + inward_range),
               (back - _FACE_SNAP_OUT, back), -1)
    mf = abs(f1 - front) >= _FACE_SNAP_MIN_MOVE
    mb = abs(b1 - back) >= _FACE_SNAP_MIN_MOVE
    if not mf:
        f1 = front
    if not mb:
        b1 = back
    if not (mf or mb):
        return box, {"moved": False, "reason": "already on the sheets",
                     "debug": debug}
    depth, old_depth = f1 - b1, 2.0 * half_d
    if not (_FACE_SNAP_MIN_D <= depth <= _FACE_SNAP_MAX_D):
        return box, {"moved": False,
                     "reason": f"snapped depth {depth:.2f} out of bounds",
                     "debug": debug}
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
            "back": [round(back, 3), round(b1, 3)],
            "debug": debug}
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


def _same_split_continues(boxes, box: OrientedBox, side: int) -> bool:
    """Whether this side is an internal seam from the same SAM split."""
    split_seed = box.meta.get("sam_split_seed")
    if not split_seed:
        return False
    for other in boxes:
        if (other.box_id == box.box_id
                or other.meta.get("sam_split_seed") != split_seed):
            continue
        if _row_continues([box, other], box, side):
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
    run nearest the current face, with an INWARD PRIORITY (user
    report: the face extended onto the cable ladder beside the row's
    end -- outward seeking finds the ladder; in practice end faces
    need trimming far more often than extending): (1) a face ON
    qualifying mass extends/trims within its own connected run (an
    under-measured end reaches the true edge; a union box's face sits
    ON the outer structure and does not move -- the amputation
    protection); (2) a face PAST its mass (bleed, a gap) trims to the
    nearest qualifying run INWARD -- outward seeking is removed
    entirely, an outward structure in the window (a ladder, a separate
    device) is a neighbour, never this row's continuation. The edge
    must be VISIBLE: a run reaching the window's outer boundary means
    the structure continues past the window -- the true end is beyond
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
        in_range = min(float(box.size[0]) * _END_SNAP_IN_FRAC,
                       _END_SNAP_IN_MAX)
        if outer_sign > 0:
            win_lo = face_pos - in_range
            win_hi = face_pos + _FACE_SNAP_OUT
        else:
            win_lo = face_pos - _FACE_SNAP_OUT
            win_hi = face_pos + in_range
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
        inward_window = in_win & ((centers <= face_pos) if outer_sign > 0
                                 else (centers >= face_pos))
        inward_threshold = max(
            _INWARD_SNAP_WINDOW_FRAC * float(hist[inward_window].max()),
            _INWARD_SNAP_GLOBAL_FRAC * gmax)
        if (run is not None and hist[fi] >= 0.5 * gmax):
            # genuinely ON the main structure (the face bin's density
            # is at least half the PROFILE's peak -- the row's own
            # sheet density, not just the containing run's: a face on
            # a sparse bridge between row and ladder has a low absolute
            # density even though it's ≥50% of the BRIDGE run's peak,
            # and must NOT count as on-mass)
            edge = run[1] if outer_sign > 0 else run[0]
        else:
            # INWARD PRIORITY (user direction: shrink is the default,
            # outward is the LAST resort -- scan inward up to 1/3 of
            # the box length first; only when NOTHING qualifies inward
            # look outward with the tight 0.15m window, and even then
            # only to a run whose density is >= the on-mass bar (a
            # sparse bridge/ladder run never attracts the face)
            if outer_sign > 0:
                inward = [r for r in runs if centers[r[1]] < face_pos
                          and hist[r[0]:r[1] + 1].max() >= inward_threshold]
                if inward:
                    run = max(inward, key=lambda r: centers[r[1]])
                    edge = run[1]
                else:
                    # last resort: OUTWARD, but only to a run whose
                    # peak density is on-par with the row (a bridge or
                    # ladder at <50% of gmax must never attract)
                    outward = [r for r in runs if centers[r[0]] > face_pos
                               and float(hist[r[0]:r[1] + 1].max())
                               >= 0.5 * gmax]
                    if not outward:
                        return face_pos
                    run = min(outward, key=lambda r: centers[r[0]])
                    edge = run[0]
            else:
                inward = [r for r in runs if centers[r[0]] > face_pos
                          and hist[r[0]:r[1] + 1].max() >= inward_threshold]
                if inward:
                    run = min(inward, key=lambda r: centers[r[0]])
                    edge = run[0]
                else:
                    outward = [r for r in runs if centers[r[1]] < face_pos
                               and float(hist[r[0]:r[1] + 1].max())
                               >= 0.5 * gmax]
                    if not outward:
                        return face_pos
                    run = max(outward, key=lambda r: centers[r[1]])
                    edge = run[1]
        # the edge must be VISIBLE within the window: a run reaching
        # the outer boundary means the structure continues past it
        if outer_sign > 0 and centers[edge] >= win_hi - _FACE_BIN:
            return face_pos
        if outer_sign < 0 and centers[edge] <= win_lo + _FACE_BIN:
            return face_pos
        inward_snap = (centers[edge] - face_pos) * outer_sign < 0.0
        if inward_snap:
            # A plateau still supported at and beyond the old end means
            # this face is inside the device, not an oversized empty end.
            beyond = ((centers - face_pos) * outer_sign > _FACE_SNAP_MIN_MOVE)
            beyond &= np.abs(centers - face_pos) <= _FACE_SNAP_OUT
            if (hist[fi] >= inward_threshold and beyond.any()
                    and float(hist[beyond].max()) >= inward_threshold):
                return face_pos
        if inward_snap:
            # Run detection locates plateau edges; its peak must pass the
            # same inward candidate-strength gate used by front/back sheets.
            # Include the immediately adjacent body bins: a partial edge bin
            # can form a short run due to sampling noise, without representing
            # a separate weak structure. This does not change edge selection.
            body_peak = float(hist[max(0, run[0] - 2):
                                   min(len(hist), run[1] + 3)].max())
            if body_peak < inward_threshold:
                return face_pos
        if (not inward_snap
                and not _height_boundary_ok(box, pts, axis, cross,
                                            float(centers[edge]), cross_c,
                                            end_profile=True)):
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
        # Pieces marked by the same local SAM refinement are independent
        # cabinets from one joined seed, not a large-box/small-box overlap.
        # Let their seam and own profiles remain independent.
        split_group = box.meta.get("sam_split_seed")
        if (split_group and split_group == s.meta.get("sam_split_seed")):
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


def trim_nested_boxes(scene: Scene) -> int:
    """Trim obvious global nested boxes before local VLM/SAM refinement.

    This is a seed-cleaning operation: a large Stage G region that covers a
    clearly smaller grounded device is cut back before local views and SAM
    masks are generated. Returns the number of boxes changed.
    """
    n = 0
    snapshot = list(scene.boxes)
    pts = np.asarray(scene.points, dtype=float)
    for i, b in enumerate(snapshot):
        try:
            trimmed, info = _nested_trim(b, scene.boxes, pts)
        except Exception as e:
            print(f"[preC] nested trim failed ({type(e).__name__})")
            continue
        if info is None:
            continue
        scene.boxes[i] = trimmed
        trimmed.meta["nested_trim"] = info
        n += 1
        print(f"[preC] {trimmed.box_id[:6]} nested trim: "
              f"{info['side']} face {info['cut'][0]:.2f} -> "
              f"{info['cut'][1]:.2f} (cut back to "
              f"{info['small_box'][:6]})")
    return n


def snap_faces_to_mesh(scene: Scene) -> int:
    """stageF: per box -- (1) the FREE row ends (the outmost ends of a
    joined row -- internal seams are left to snap_row_seams, user direction)
    and (2) the front/back faces snap
    onto the densest mesh sheets of the box's OWN column (other boxes'
    points never feed the profile). Nested trimming is performed before
    Stage C by trim_nested_boxes(); Stage F only polishes geometry.
    Identity and meta are preserved; adjustments are recorded in
    meta['end_snap'] / meta['face_snap']. Returns the number of boxes
    adjusted."""
    n = 0
    snapshot = list(scene.boxes)
    pts = np.asarray(scene.points, dtype=float)
    for i, b in enumerate(snapshot):
        try:
            cur = b
            tb, tinfo = _trim_sparse_top(cur, pts)
            if tinfo.get("moved"):
                cur = tb
                scene.boxes[i] = cur
                cur.meta["top_trim"] = tinfo
                n += 1
                print(f"[stageF] {cur.box_id[:6]} top trim: height "
                      f"{tinfo['height'][0]:.2f} -> "
                      f"{tinfo['height'][1]:.2f}m")
            own = _own_points(pts, scene.boxes, cur)
            nb, einfo = snap_box_ends(
                cur, pts[own],
                # Every independent grounding box gets both side faces
                # checked. Only a seam between pieces from the SAME SAM
                # split is reserved for Stage C's seam regularisation.
                snap_left=not _same_split_continues(snapshot, b, -1),
                snap_right=not _same_split_continues(snapshot, b, +1))
            if einfo.get("moved"):
                nb.meta["end_snap"] = einfo
                print(f"[stageF] {nb.box_id[:6]} end snap: length "
                      f"{einfo['length'][0]:.2f} -> {einfo['length'][1]:.2f}m "
                      f"(left {einfo['left'][0]:.2f} -> {einfo['left'][1]:.2f}, "
                      f"right {einfo['right'][0]:.2f} -> "
                      f"{einfo['right'][1]:.2f})")
            nb2, finfo = snap_box_faces(nb, pts[own])
            if finfo.get("debug"):
                nb2.meta["face_snap_debug"] = finfo["debug"]
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
