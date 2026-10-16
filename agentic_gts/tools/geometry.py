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


def flag_insufficient_face_support(scene: Scene, tolerance: float = 0.03,
                                   cell: float = 0.10,
                                   min_coverage: float = 0.70,
                                   min_points: int = 10,
                                   required_faces: int = 2) -> int:
    """Adjust confidence one level using five non-bottom fitted faces.

    Density is occupied face-grid area, rather than raw point count, so a
    dense ladder fragment cannot qualify an otherwise empty face.
    
    Required faces defaults to 2 to accommodate middle cabinets in a row
    (only front/back + top visible; left/right occluded by neighbours).
    """
    flagged = 0
    for box in scene.boxes:
        region = _region_of_box(box, tolerance)
        local = box.world_to_local(scene.points_in_region(region))
        half = np.asarray(box.size, dtype=float) / 2.0
        faces = {}
        for name, axis, sign in (("front", 1, 1), ("back", 1, -1),
                                 ("left", 0, -1), ("right", 0, 1),
                                 ("top", 2, 1)):
            tangent = [i for i in range(3) if i != axis]
            lengths = 2.0 * half[tangent]
            mask = np.abs(local[:, axis] - sign * half[axis]) <= tolerance
            mask &= np.all(np.abs(local[:, tangent]) <= half[tangent], axis=1)
            selected = local[mask][:, tangent]
            bins = np.maximum(np.ceil(lengths / cell).astype(int), 1)
            idx = np.clip(((selected + half[tangent]) / cell).astype(int),
                          0, bins - 1)
            occupied = len(np.unique(idx, axis=0))
            coverage = float(occupied / int(np.prod(bins)))
            faces[name] = {"points": int(len(selected)),
                           "coverage": round(coverage, 4),
                           "passed": bool(len(selected) >= min_points
                                          and coverage >= min_coverage)}
        matched = sum(face["passed"] for face in faces.values())
        audit = box.meta.get("face_support", {})
        base_name = audit.get("base_confidence", box.confidence.value)
        base_conf = Confidence(base_name)
        ranks = {Confidence.LOW: 0, Confidence.MID: 1, Confidence.HIGH: 2}
        delta = 1 if matched >= required_faces else -1
        levels = (Confidence.LOW, Confidence.MID, Confidence.HIGH)
        adjusted = levels[int(np.clip(ranks[base_conf] + delta, 0, 2))]
        # Semantic/structural rejection flags and single-view support are
        # stronger evidence than geometric face support and must never be
        # promoted by this check.
        protected_low = any(box.meta.get(k) for k in
                            ("type_suspect", "structural_geom",
                             "low_furniture_geom", "single_view_support"))
        if protected_low:
            # Debug: show which flag triggered the protection
            active_flags = [k for k in ("type_suspect", "structural_geom",
                                        "low_furniture_geom", "single_view_support")
                           if box.meta.get(k)]
            print(f"[face-support] {box.box_id[:6]} protected LOW by: {', '.join(active_flags)}")
            adjusted = Confidence.LOW
        box.confidence = adjusted
        box.meta["face_support"] = {
            "base_confidence": base_conf.value,
            "confidence_before": base_conf.value,
            "confidence_after": adjusted.value,
            "faces": faces, "matched_faces": matched, "required_faces": required_faces,
            "tolerance_m": tolerance, "grid_m": cell,
            "min_coverage": min_coverage, "min_points": min_points,
            "passed": matched >= required_faces,
        }
        if matched < required_faces:
            flagged += 1
        print(f"[face-support] {box.box_id} fitted={matched}/5 "
              f"confidence={base_conf.value}->{adjusted.value}")
    return flagged


def filter_outer_wall_boxes(scene: Scene, edge_margin: float = 0.35) -> int:
    """Remove wall-and-air boxes and long thin strips with no top support."""
    if not len(scene.points):
        return 0
    keep, removed = [], 0
    projection_bounds = {}
    for b in scene.boxes:
        support = b.meta.get("face_support", {})
        faces = support.get("faces", {})
        long_side = max(float(b.size[0]), float(b.size[1]))
        short_side = min(float(b.size[0]), float(b.size[1]))
        top_points = faces.get("top", {}).get("points")
        if (long_side >= 2.0 and 0 < short_side <= 0.45
                and long_side / short_side >= 5.0 and top_points == 0):
            b.meta["wall_false_positive"] = {
                "length": long_side, "thickness": short_side,
                "aspect_ratio": long_side / short_side,
                "top_points": 0,
                "reason": "long thin box with no points on its top face"}
            scene.meta.setdefault("wall_filter_audit", []).append(b.to_dict())
            print(f"[wall-filter] {b.box_id} removed: long thin box, empty top")
            removed += 1
            continue
        if support.get("matched_faces", 5) >= 3:
            keep.append(b)
            continue
        possible = [(axis, sign, name) for axis, pairs in (
            (0, ((-1, "left", "right"), (1, "right", "left"))),
            (1, ((-1, "back", "front"), (1, "front", "back"))))
            for sign, name, opposite in pairs
            if faces.get(name, {}).get("points", 0) >= 10
            and faces.get(name, {}).get("coverage", 0.0) >= 0.05
            and faces.get(opposite, {}).get("coverage", 1.0) < 0.05]
        if not possible:
            keep.append(b)
            continue
        tolerance = float(support.get("tolerance_m", 0.05))
        local = b.world_to_local(scene.points_in_region(_region_of_box(b, tolerance)))
        half = np.asarray(b.size, dtype=float) / 2.0
        body = local[np.all(np.abs(local) <= half + tolerance, axis=1)
                     & (local[:, 2] > -half[2] + 0.30)]
        wall_face = None
        for axis, sign, name in possible:
            concentration = (float((np.abs(body[:, axis] - sign * half[axis])
                                    <= tolerance).mean()) if len(body) else 0.0)
            if concentration < 0.90:
                continue
            direction = b.rotation[:2, axis]
            key = tuple(direction)
            if key not in projection_bounds:
                projected = scene.points[:, :2] @ direction
                projection_bounds[key] = (float(projected.min()),
                                          float(projected.max()))
            center = float(np.asarray(b.center[:2]) @ direction)
            face_pos = center + sign * float(b.size[axis]) / 2.0
            limit = projection_bounds[key][0 if sign < 0 else 1]
            if abs(face_pos - limit) <= edge_margin:
                wall_face = name
                break
        if wall_face:
            b.meta["wall_false_positive"] = {
                "outer_face": wall_face,
                "wall_point_fraction": round(concentration, 4),
                "reason": "outer-boundary box with wall-concentrated points "
                          "and empty room-facing face"}
            scene.meta.setdefault("wall_filter_audit", []).append(b.to_dict())
            print(f"[wall-filter] {b.box_id} removed: outer={wall_face}")
            removed += 1
            continue
        keep.append(b)
    scene.boxes = keep
    return removed


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
                    height_tol: float = 0.05, preserve_cross: bool = False) -> int:
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
        if (a.meta.get("surface_span_measured")
                and b.meta.get("surface_span_measured")):
            continue  # visible instance boundaries outrank seam averaging
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
        if (not preserve_cross and abs(a_clo - b_clo) <= vertex_tol
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
_SNAP_MIN_REMAINING = 0.30
_FACE_SNAP_OUT = 0.15
_FACE_SNAP_OUT_MAX = 0.30
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
_FACE_EMPTY_FACE_FRAC = 0.05
# the END rule's bar is LOWER: it separates the plateau from mask
# bleed (a ~20:1 contrast), while the cross rule separates competing
# peaks (2:1) -- 50% of a fluctuating window max sits inside the
# plateau's own counting noise (an edge bin a couple of sigma low
# vs one a couple high) and drops the TRUE edge bin, landing the
# snap a bin inside
_FACE_END_TAU = 0.30
# END inward search range: 1/3 of the box's along extent, capped at 3m.
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


def _extended_face_supported(box: OrientedBox, pts: np.ndarray,
                             axis: int, position: float) -> bool:
    """Require distributed surface evidence for growth beyond 15cm."""
    local = box.world_to_local(pts)
    tangent = [i for i in range(3) if i != axis]
    half = np.asarray(box.size) / 2.0
    mask = np.abs(local[:, axis] - position) <= 0.05
    mask &= np.all(np.abs(local[:, tangent]) <= half[tangent], axis=1)
    selected = local[mask][:, tangent]
    bins = np.maximum(np.ceil(2 * half[tangent] / 0.1).astype(int), 1)
    idx = np.clip(((selected + half[tangent]) / 0.1).astype(int), 0, bins - 1)
    return len(selected) >= 20 and len(np.unique(idx, axis=0)) / bins.prod() >= 0.30


def snap_box_faces(box: OrientedBox,
                   pts: np.ndarray, outward_limit: float = _FACE_SNAP_OUT,
                   _profile=None
                   ) -> tuple[OrientedBox, dict]:
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
    observed = box.meta.get("observed_faces", {})
    if "front" in observed and "back" in observed:
        return box, {"moved": False, "reason": "observed surfaces locked",
                     "debug": [{"face": name, "accepted": False,
                                "reason": "observed_surface_locked",
                                "searched": "keep"}
                               for name in ("front", "back")]}
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
    if _profile is None:
        m = ((np.abs(pts[:, :2] @ axis - along_c) <= half_len + 0.05)
             & (pts[:, 2] > bottom + 0.30) & (pts[:, 2] <= top))
        _profile = (int(m.sum()), pts[m, :2] @ cross)
    profile_points, cc = _profile
    if profile_points < 50:
        return box, {"moved": False, "reason": "too few profile points"}
    prof_lo = cross_c - half_d - outward_limit
    prof_hi = cross_c + half_d + outward_limit
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
        if observed:
            return box, {"moved": False, "reason": "empty unmeasured profile"}
        if outward_limit == _FACE_SNAP_OUT:
            fallback_box, fallback = snap_box_faces(
                box, pts, outward_limit=_FACE_SNAP_OUT_MAX, _profile=_profile)
            if fallback.get("moved"):
                front_pos, back_pos = cross_c + half_d, cross_c - half_d
                new_front, new_back = fallback["front"][1], fallback["back"][1]
                valid = all(
                    abs(new - old) < _FACE_SNAP_MIN_MOVE
                    or (0 < (new - old) * sign <= _FACE_SNAP_OUT_MAX
                        and _extended_face_supported(box, pts, 1, new - cross_c))
                    for old, new, sign in ((front_pos, new_front, 1),
                                           (back_pos, new_back, -1)))
                if valid:
                    for rec in fallback["debug"]:
                        rec["searched"] = "outward_extended"
                    return fallback_box, fallback
        return box, {"moved": False, "reason": "empty profile"}

    debug = []
    inward_options = {}

    def _peaks(win_lo: float, win_hi: float, side: int,
               inward: bool = False) -> np.ndarray:
        in_win = (centers >= win_lo) & (centers <= win_hi)
        if not in_win.any():
            return np.empty(0, dtype=np.int64)
        if inward:
            thr = max(_INWARD_SNAP_WINDOW_FRAC * float(band[in_win].max()),
                      _INWARD_SNAP_GLOBAL_FRAC * gmax)
        else:
            thr = max((0.65 if outward_limit > _FACE_SNAP_OUT else _FACE_SNAP_TAU)
                      * float(band[in_win].max()),
                      (0.35 if outward_limit > _FACE_SNAP_OUT else _FACE_SNAP_ABS)
                      * gmax)
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
        interior_peaks = _peaks(*inward, side, inward=True)
        cand = interior_peaks if empty_face else np.empty(0, dtype=np.int64)
        exterior = _peaks(*outward, side)
        near_surface = np.concatenate([interior_peaks, exterior])
        side_peak = (float(band[near_surface].max())
                     if len(near_surface) else 0.0)
        near_surface = near_surface[
            (np.abs(centers[near_surface] - face_pos) <= _FACE_BIN)
            & (band[near_surface] >= _INWARD_SNAP_WINDOW_FRAC * side_peak)]
        is_inward = bool(len(cand))
        searched = "inward"
        direction_reason = ("empty_face_inward_recovery" if empty_face
                            else "inward_blocked_face_has_support")
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
            inner_peak = float(band[interior_peaks].max()) if len(interior_peaks) else 0.0
            if (len(exterior) and (not len(interior_peaks)
                    or float(band[exterior].max()) >= 1.25 * inner_peak)):
                cand = exterior
                is_inward = False
                searched = "outward"
                direction_reason = "stronger_exterior_surface"
        if not len(cand):
            cand = exterior if empty_face or not len(interior_peaks) else cand
            is_inward = False
            searched = "outward"
        rec = {
            "face": "front" if side > 0 else "back",
            "old": round(float(face_pos), 3),
            "inward_range": [round(float(inward[0]), 3),
                             round(float(inward[1]), 3)],
            "outward_range": [round(float(outward[0]), 3),
                              round(float(outward[1]), 3)],
            "profile_points": profile_points,
            "gmax": int(gmax),
            "face_density": round(current_face_density, 1),
            "empty_face": bool(empty_face),
            "searched": searched,
            "direction_reason": direction_reason,
            "candidate_count": int(len(cand)),
        }
        if len(cand):
            if is_inward:
                inward_options[side] = cand
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
        rec["reason"] = ("no_qualifying_peak" if empty_face
                         else "inward_blocked_face_has_support")
        debug.append(rec)
        return face_pos

    front, back = cross_c + half_d, cross_c - half_d
    inward_range = max(float(box.size[1]) - _SNAP_MIN_REMAINING, 0.0)
    f1 = _snap(front,
               (front - inward_range, front),
               (front, front + outward_limit), +1)
    b1 = _snap(back,
                (back, back + inward_range),
                (back - outward_limit, back), -1)
    observed = box.meta.get("observed_faces", {})
    if "front" in observed:
        f1 = front
        inward_options.pop(1, None)
        debug[0].update(accepted=False, reason="observed_surface_locked",
                        searched="keep")
    if "back" in observed:
        b1 = back
        inward_options.pop(-1, None)
        debug[1].update(accepted=False, reason="observed_surface_locked",
                        searched="keep")
    if outward_limit == _FACE_SNAP_OUT and any(
            r.get("reason") in ("no_qualifying_peak", "outward_height_guard")
            for r in debug):
        _, fallback = snap_box_faces(box, pts, outward_limit=_FACE_SNAP_OUT_MAX,
                                     _profile=_profile)
        for index, original in enumerate((front, back)):
            if debug[index].get("reason") not in (
                    "no_qualifying_peak", "outward_height_guard"):
                continue
            record = fallback.get("debug", [{}, {}])[index]
            candidate = record.get("candidate", original)
            sign = 1 if index == 0 else -1
            if (record.get("accepted") and record.get("searched") == "outward"
                    and _FACE_SNAP_OUT < (candidate - original) * sign <= _FACE_SNAP_OUT_MAX
                    and _extended_face_supported(box, pts, 1, candidate - cross_c)):
                if index == 0:
                    f1 = candidate
                else:
                    b1 = candidate
                debug[index] = {**record, "searched": "outward_extended"}
    if 1 in inward_options and -1 in inward_options:
        # The expanded windows may overlap. Choose a supported pair jointly
        # so the two faces cannot collapse onto the same strongest sheet.
        pairs = [(f, b) for f in inward_options[1]
                 for b in inward_options[-1]
                 if _SNAP_MIN_REMAINING <= centers[f] - centers[b]
                 <= _FACE_SNAP_MAX_D]
        if pairs:
            f, b = max(pairs, key=lambda pair: (
                int(band[pair[0]]) + int(band[pair[1]]),
                -abs(centers[pair[0]] - front)
                - abs(centers[pair[1]] - back)))
            f1, b1 = float(centers[f]), float(centers[b])
            for rec, idx in zip(debug, (f, b)):
                rec["candidate"] = round(float(centers[idx]), 3)
                rec["candidate_peak"] = int(band[idx])
        else:
            # Keep one measured face if no two-sheet pair is wide enough.
            # This preserves the opposite original face instead of rejecting
            # both independent corrections.
            single = [(f1, back), (front, b1)]
            single = [p for p in single if _SNAP_MIN_REMAINING <= p[0] - p[1]
                      <= _FACE_SNAP_MAX_D]
            if single:
                f1, b1 = max(single, key=lambda p: p[0] - p[1])
                for rec, pos, original in zip(debug, (f1, b1), (front, back)):
                    rec["candidate"] = round(pos, 3)
                    if pos == original:
                        rec["accepted"] = False
                        rec["reason"] = "opposite_face_minimum_spacing"
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
                    snap_right: bool = True,
                    outward_limit: float = _FACE_SNAP_OUT,
                    _profile=None) -> tuple[OrientedBox, dict]:
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
    if _profile is None:
        m = ((np.abs(pts[:, :2] @ cross - cross_c) <= half_d + 0.05)
             & (pts[:, 2] > bottom + 0.30) & (pts[:, 2] <= top))
        _profile = (int(m.sum()), pts[m, :2] @ axis)
    profile_points, aa = _profile
    if profile_points < 50:
        return box, {"moved": False, "reason": "too few profile points"}
    prof_lo = along_c - half_len - outward_limit
    prof_hi = along_c + half_len + outward_limit
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
        if outward_limit == _FACE_SNAP_OUT:
            fallback_box, fallback = snap_box_ends(
                box, pts, snap_left, snap_right,
                outward_limit=_FACE_SNAP_OUT_MAX, _profile=_profile)
            if fallback.get("moved"):
                right_pos, left_pos = along_c + half_len, along_c - half_len
                valid = all(
                    abs(new - old) < _FACE_SNAP_MIN_MOVE
                    or (0 < (new - old) * sign <= _FACE_SNAP_OUT_MAX
                        and _extended_face_supported(box, pts, 0, new - along_c))
                    for old, new, sign in ((right_pos, fallback["right"][1], 1),
                                           (left_pos, fallback["left"][1], -1)))
                if valid:
                    return fallback_box, fallback
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
        in_range = max(float(box.size[0]) - _SNAP_MIN_REMAINING, 0.0)
        if outer_sign > 0:
            win_lo = face_pos - in_range
            win_hi = face_pos + outward_limit
        else:
            win_lo = face_pos - outward_limit
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
        face_zone = np.abs(centers - face_pos) <= _FACE_BAND
        face_density = float(hist[face_zone].max()) if face_zone.any() else 0.0
        if inward_snap and face_density > _FACE_EMPTY_FACE_FRAC * gmax:
            return face_pos
        if not inward_snap and outward_limit > _FACE_SNAP_OUT:
            if hist[run[0]:run[1] + 1].max() < max(
                    0.65 * hist[in_win].max(), 0.35 * gmax):
                return face_pos
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
    if outward_limit == _FACE_SNAP_OUT and (
            (snap_right and abs(r1 - right) < _FACE_SNAP_MIN_MOVE)
            or (snap_left and abs(l1 - left) < _FACE_SNAP_MIN_MOVE)):
        _, fallback = snap_box_ends(box, pts, snap_left, snap_right,
                                    outward_limit=_FACE_SNAP_OUT_MAX,
                                    _profile=_profile)
        if fallback.get("moved"):
            r2, l2 = fallback["right"][1], fallback["left"][1]
            if (snap_right and abs(r1 - right) < _FACE_SNAP_MIN_MOVE
                    and _FACE_SNAP_OUT < r2 - right <= _FACE_SNAP_OUT_MAX
                    and _extended_face_supported(box, pts, 0, r2 - along_c)):
                r1 = r2
            if (snap_left and abs(l1 - left) < _FACE_SNAP_MIN_MOVE
                    and _FACE_SNAP_OUT < left - l2 <= _FACE_SNAP_OUT_MAX
                    and _extended_face_supported(box, pts, 0, l2 - along_c)):
                l1 = l2
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
    if length < _SNAP_MIN_REMAINING:
        single = [(r1, left), (right, l1)]
        single = [p for p in single if p[0] - p[1] >= _SNAP_MIN_REMAINING]
        if single:
            r1, l1 = max(single, key=lambda p: p[0] - p[1])
            length = r1 - l1
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


def shrink_local_face_protrusions(box: OrientedBox, pts: np.ndarray):
    """Shrink a face only when its exterior is local and an inner body sheet
    covers both the box height and length. Never expand either face.
    """
    local = box.world_to_local(np.asarray(pts, dtype=float))
    half = np.asarray(box.size, dtype=float) / 2
    mask = ((np.abs(local[:, 0]) <= half[0])
            & (np.abs(local[:, 2]) <= half[2])
            & (np.abs(local[:, 1]) <= half[1] + 0.03))
    local = local[mask]
    audit = {"moved": False, "faces": {}}
    if len(local) < 50:
        return box, audit
    nx = max(1, int(np.ceil(box.size[0] / 0.10)))
    nz = max(1, int(np.ceil(box.size[2] / 0.10)))

    def coverage(points):
        if not len(points):
            return 0., 0., 0.
        x = np.clip(((points[:, 0] + half[0]) / .10).astype(int), 0, nx - 1)
        z = np.clip(((points[:, 2] + half[2]) / .10).astype(int), 0, nz - 1)
        return (len(np.unique(x)) / nx, len(np.unique(z)) / nz,
                len(np.unique(np.column_stack([x, z]), axis=0)) / (nx * nz))

    back, front = -half[1], half[1]
    for name, sign in (("front", 1.), ("back", -1.)):
        signed = sign * local[:, 1]
        # Search inward from the current boundary, stopping well before the
        # opposite face; a cabinet interior is not a replacement outer sheet.
        limit = min(0.30, box.size[1] * 0.25)
        candidates = np.arange(half[1] - .01, half[1] - limit, -.005)
        audit.setdefault("checks", {})[name] = {
            "reason": "no broad inner sheet", "max_area_coverage": 0.0}
        for candidate in candidates:
            sheet = local[np.abs(signed - candidate) <= .005]
            cx, cz, area = coverage(sheet)
            audit["checks"][name]["max_area_coverage"] = max(
                audit["checks"][name]["max_area_coverage"], area)
            if len(sheet) < 40 or cx < .60 or cz < .60 or area < .40:
                continue
            exterior = local[signed > candidate + .008]
            ex, ez, ea = coverage(exterior)
            # A second broad sheet is ambiguous (e.g. a door): keep the face.
            if len(exterior) and (ez >= .35 and ex >= .35):
                audit["checks"][name]["reason"] = "broad exterior structure"
                continue
            levels = np.floor((sheet[:, 2] + half[2]) / .10).astype(int)
            boundary = float(np.median([
                np.median(sign * sheet[levels == level, 1])
                for level in np.unique(levels)]))
            if half[1] - boundary < .01 - 1e-9:
                continue
            new_face = sign * boundary
            if name == "front":
                front = new_face
            else:
                back = new_face
            audit["faces"][name] = {
                "before": sign * float(half[1]), "after": new_face,
                "height_coverage": cz, "length_coverage": cx,
                "area_coverage": area, "exterior_height_coverage": ez,
                "exterior_length_coverage": ex, "exterior_area_coverage": ea}
            audit["checks"][name]["reason"] = "accepted"
            break
    if not audit["faces"] or front - back < .30:
        return box, {**audit, "moved": False}
    import copy
    result = copy.deepcopy(box)
    center = box.local_to_world(np.array([[0., (front + back) / 2, 0.]]))[0]
    result.center = tuple(float(v) for v in center)
    result.size = (box.size[0], float(front - back), box.size[2])
    audit["moved"] = True
    # Subsequent face snapping must respect these validated body boundaries.
    observed = result.meta.setdefault("observed_faces", {})
    for name in audit["faces"]:
        observed[name] = {"source": "stageF_body_shrink", **audit["faces"][name]}
    result.meta["body_face_shrink"] = audit
    return result, audit


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
            cur, shrink_info = shrink_local_face_protrusions(cur, pts[own])
            cur.meta["body_face_shrink_check"] = shrink_info
            if shrink_info["moved"]:
                scene.boxes[i] = cur
                print(f"[stageF] {cur.box_id[:6]} body face shrink: "
                      f"{shrink_info['faces']}")
            nb, einfo = snap_box_ends(
                cur, pts[own],
                # Every independent grounding box gets both side faces
                # checked. Only a seam between pieces from the SAME SAM
                # split is reserved for Stage C's seam regularisation.
                snap_left=not cur.meta.get("surface_span_measured")
                          and not _same_split_continues(snapshot, b, -1),
                snap_right=not cur.meta.get("surface_span_measured")
                           and not _same_split_continues(snapshot, b, +1))
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
            if shrink_info["moved"] or einfo.get("moved") or finfo.get("moved"):
                scene.boxes[i] = nb2
                n += 1
        except Exception as e:
            print(f"[stageF] {b.box_id[:6]} snap failed "
                  f"({type(e).__name__}: {e})")
    return n
