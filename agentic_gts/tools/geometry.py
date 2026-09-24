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
        # 1D clustering on the cross coordinate
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
    extent from its OWN measured span, so the seam between two
    neighbours can come out a few centimetres apart (mask bleed cut at
    slightly different places per view, the cross-view union, the
    per-piece side thickness correction). The result reads as a row of
    DISCONNECTED boxes. This walks the pieces sorted along the row axis
    and, wherever two facing edges sit within `seam_tol` (a small gap
    OR overlap), sets both to their average along coordinate -- the
    "average the vertices" rule. The shared edge's CROSS vertices are
    merged only where the two facing corner pairs are EACH within
    `vertex_tol` ("merge vertices only when close"); otherwise only the
    along seam is normalised and each side keeps its own cross extent.

    HEIGHT STEP (user directive): devices whose TOPS sit more than
    `height_tol` apart are NOT on one plane -- the split separated them
    on purpose (different-height cabinets), so they are NEVER seamed,
    however small the gap. Heights never enter the merge itself beyond
    this gate -- the top vertices may sit far apart in z, the footprint
    edge is normalised regardless.

    Boxes are assumed to share the row frame (`yaw`); the split pieces
    of one seed always do (they copy the seed's yaw and carry the along
    extent in size[0]). Facing edges must also overlap laterally by at
    least half the thinner body -- two different sub-rows are never
    snapped. Mutates in place; returns the number of seams snapped.
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
        # devices on DIFFERENT height planes (> height_tol between their
        # tops) are distinct devices the split separated on purpose --
        # NEVER seam across a height step, however small the gap (user
        # directive)
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
        # merge the shared edge's CROSS vertices only where the two
        # facing corner pairs are EACH within vertex_tol ("merge
        # vertices only when close"); otherwise only the along seam is
        # normalised and each side keeps its own cross extent. Heights
        # never enter -- the top vertices may sit far apart in z.
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
