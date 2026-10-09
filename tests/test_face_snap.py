"""Tests for stageF -- the final mesh-driven face polish
(tools/geometry.snap_box_faces / snap_faces_to_mesh): front/back faces
snap onto the densest mesh sheets in the box's own column; an open
door's low wide cross-plateau never qualifies, a flush wall's strong
but farther peak never steals the face, and boxes already on their
sheets are untouched (idempotent)."""
import os
import sys

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from agentic_gts.core.models import OrientedBox, Scene
from agentic_gts.tools.geometry import (snap_box_ends, snap_box_faces,
                                        snap_faces_to_mesh,
                                        trim_nested_boxes)


def _box(half_d, half_len=1.0, z=1.05):
    return OrientedBox(center=(0.0, 0.0, z),
                       size=(2.0 * half_len, 2.0 * half_d, 2.10),
                       yaw=0.0)


def _sheet(rng, cross, n=1500, along=1.0, z_lo=0.35, z_hi=2.00):
    """A dense planar sheet at a fixed cross coordinate."""
    return np.column_stack([rng.uniform(-along, along, n),
                            rng.uniform(cross - 0.01, cross + 0.01, n),
                            rng.uniform(z_lo, z_hi, n)])


def _plateau(rng, c_lo, c_hi, n=400, along=0.6, z_lo=0.35, z_hi=2.00):
    """A low wide plateau across a cross range (an open door swung 90
    deg: the panel spreads along the cross axis)."""
    return np.column_stack([rng.uniform(-along, along, n),
                            rng.uniform(c_lo, c_hi, n),
                            rng.uniform(z_lo, z_hi, n)])


def _floor(rng, n=800):
    """Floor points spanning the whole cross range -- excluded by the
    z band, they would flatten the face peaks into a background."""
    return np.column_stack([rng.uniform(-1.0, 1.0, n),
                            rng.uniform(-1.2, 1.2, n),
                            rng.uniform(0.00, 0.20, n)])


def test_face_snap_pulls_in_open_door_inflation():
    """The primary case (user direction): a box inflated by an open
    door snaps both faces back onto the device's own sheets; the
    door's plateau never qualifies."""
    rng = np.random.default_rng(3)
    pts = np.vstack([
        _sheet(rng, 0.55), _sheet(rng, -0.55),
        _plateau(rng, 0.57, 1.25),      # the open door, front side
        _floor(rng)])
    b = _box(0.70)                       # inflated: faces at +/-0.70
    nb, info = snap_box_faces(b, pts)
    assert info["moved"], info
    assert abs(nb.size[1] - 1.11) < 0.06, \
        f"depth must return to the sheets (~1.11), got {nb.size[1]:.2f}"
    assert abs(nb.center[1]) < 0.03, "the box stays centred"
    assert nb.box_id == b.box_id and nb.meta is b.meta, \
        "identity and meta must survive"
    print(f"PASS face snap pulls in door inflation "
          f"(depth {info['depth'][0]:.2f} -> {info['depth'][1]:.2f})")


def test_face_snap_noop_on_clean_box():
    """A box already on its sheets does not move (idempotent)."""
    rng = np.random.default_rng(4)
    pts = np.vstack([_sheet(rng, 0.55), _sheet(rng, -0.55), _floor(rng)])
    b = _box(0.55)
    nb, info = snap_box_faces(b, pts)
    assert not info["moved"] and nb is b
    print("PASS face snap no-op on a clean box")


def test_face_snap_extends_under_measured():
    """An under-measured box (faces inside the device) extends out to
    its sheets through the tight outward window."""
    rng = np.random.default_rng(5)
    pts = np.vstack([_sheet(rng, 0.55), _sheet(rng, -0.55), _floor(rng)])
    b = _box(0.45)
    nb, info = snap_box_faces(b, pts)
    assert info["moved"], info
    assert abs(nb.size[1] - 1.11) < 0.06, \
        f"depth must reach the sheets, got {nb.size[1]:.2f}"
    print(f"PASS face snap extends under-measured "
          f"(depth {info['depth'][0]:.2f} -> {info['depth'][1]:.2f})")


def test_face_snap_ignores_wall_behind():
    """A flush wall behind the back face is a strong peak INSIDE the
    outward window but FARTHER than the device's own back sheet:
    nearest-strong wins, the face stays on the device."""
    rng = np.random.default_rng(6)
    wall = _sheet(rng, -0.68, n=2500, along=1.2)   # denser than the sheets
    pts = np.vstack([_sheet(rng, 0.55), _sheet(rng, -0.55), wall,
                     _floor(rng)])
    b = _box(0.55)
    nb, info = snap_box_faces(b, pts)
    assert not info["moved"], \
        f"a strong but farther wall must not steal the back face: {info}"
    print("PASS face snap ignores the wall behind (nearest-strong)")


def test_face_snap_wide_open_door_trims():
    """The user's case: a WIDE-open door (0.6m+ stick-out) pushes the
    face past the old 0.35m inward window -- the true sheet sat
    outside it and the strength bars went self-referential over the
    door's plateau, leaving the face stuck on the door. The door-sized
    window (1.2m) reaches the sheet through the door and trims."""
    rng = np.random.default_rng(7)
    pts = np.vstack([
        _sheet(rng, 0.55), _sheet(rng, -0.55),
        _plateau(rng, 0.56, 1.30, n=500),
        _floor(rng)])
    b = _box(1.00)                      # the face sits ON the open door
    nb, info = snap_box_faces(b, pts)
    assert info["moved"], \
        f"a wide-open door inflation must trim to the sheet: {info}"
    assert abs(nb.size[1] - 1.11) < 0.06, \
        f"depth must return to the sheets (~1.11), got {nb.size[1]:.2f}"
    print(f"PASS face snap wide-open door trims "
          f"(depth {info['depth'][0]:.2f} -> {info['depth'][1]:.2f})")


def test_face_snap_strictly_prefers_inward_sheet_over_near_outer_peak():
    """A valid inward device sheet wins over a closer exterior peak."""
    rng = np.random.default_rng(25)
    pts = np.vstack([
        _sheet(rng, 0.55, n=1500), _sheet(rng, -0.55, n=1500),
        _sheet(rng, 0.73, n=500),   # nearer exterior peak for front face
        _floor(rng)])
    b = _box(0.70)  # front face at 0.70; inward sheet is at 0.55
    nb, info = snap_box_faces(b, pts)
    assert info["moved"], info
    assert abs(info["front"][1] - 0.55) < 0.06, info
    assert abs(nb.size[1] - 1.11) < 0.08, nb.size
    print("PASS face snap strict inward priority")


def test_face_snap_outward_fallback_when_inward_sheet_missing():
    """An under-measured face may still extend to a strong outer sheet."""
    rng = np.random.default_rng(26)
    pts = np.vstack([
        _sheet(rng, 0.60, n=1500), _sheet(rng, -0.55, n=1500),
        _floor(rng)])
    b = _box(0.55)
    nb, info = snap_box_faces(b, pts)
    assert info["moved"], info
    assert abs(info["front"][1] - 0.60) < 0.06, info
    assert nb.size[1] > b.size[1], (b.size, nb.size)
    print("PASS face snap outward fallback")


def test_face_snap_uses_strongest_qualifying_inward_peak():
    """Inward search uses the strongest qualifying device sheet, with
    proximity only resolving equal-strength candidates."""
    rng = np.random.default_rng(27)
    pts = np.vstack([
        _sheet(rng, 0.55, n=1800), _sheet(rng, -0.55, n=1500),
        _sheet(rng, 0.65, n=900), _floor(rng)])
    b = _box(0.70)
    nb, info = snap_box_faces(b, pts)
    assert info["moved"], info
    assert abs(info["front"][1] - 0.55) < 0.06, info
    assert abs(nb.size[1] - 1.10) < 0.08, nb.size
    print("PASS face snap chooses strongest qualifying inward peak")


def test_face_snap_recovers_when_initial_face_has_no_points():
    """A grossly oversized seed face with no local mass must still find
    the strongest device sheet inside its thickness-scaled search range."""
    rng = np.random.default_rng(33)
    pts = np.vstack([
        _sheet(rng, 0.45, n=1200), _sheet(rng, -0.45, n=1200),
        _floor(rng)])
    # Initial faces are +/-0.90, with no points there; actual sheets are
    # at +/-0.45 and fall within 2/3 of the initial 1.8m depth.
    box = _box(0.90)
    nb, info = snap_box_faces(box, pts)
    assert info["moved"], info
    assert abs(info["front"][1] - 0.45) < 0.06, info
    assert abs(info["back"][1] + 0.45) < 0.06, info
    assert abs(nb.size[1] - 0.90) < 0.10, nb.size
    print("PASS face snap recovers empty oversized initial faces")


def test_low_device_does_not_snap_to_partial_tall_ladder():
    """A nearby tall ladder may be only a small part of the slice, but
    its high local z extent must still reject the face candidate."""
    rng = np.random.default_rng(29)
    body = np.vstack([
        _sheet(rng, 0.45, n=1400, z_lo=0.30, z_hi=0.95),
        _sheet(rng, -0.45, n=1400, z_lo=0.30, z_hi=0.95),
    ])
    ladder = np.column_stack([
        rng.uniform(-0.20, 0.20, 80),
        rng.uniform(0.53, 0.55, 80),
        rng.uniform(1.10, 2.60, 80),
    ])
    low = OrientedBox(center=(0.0, 0.0, 0.50),
                      size=(2.0, 1.10, 1.0), yaw=0.0)
    nb, info = snap_box_faces(low, np.vstack([body, ladder, _floor(rng)]))
    # The overhead guard may keep the original face, but it must never
    # accept the ladder as an outward candidate.
    assert info["moved"], info
    assert info["front"][1] <= 0.55 + 1e-9, info
    print("PASS low device rejects partial tall ladder")


def test_inward_face_snap_ignores_overhead_height_guard():
    """Overhead ladder points must not veto recovery to an inward device
    sheet; the height guard is reserved for outward candidate growth."""
    rng = np.random.default_rng(32)
    body = np.vstack([
        _sheet(rng, 0.45, n=1500, z_lo=0.30, z_hi=0.95),
        _sheet(rng, -0.45, n=1500, z_lo=0.30, z_hi=0.95),
    ])
    ladder = np.column_stack([
        rng.uniform(-0.4, 0.4, 300),
        rng.uniform(0.43, 0.47, 300),
        rng.uniform(1.10, 2.60, 300),
    ])
    low = OrientedBox(center=(0.0, 0.0, 0.50),
                      size=(2.0, 1.30, 1.0), yaw=0.0)
    nb, info = snap_box_faces(low, np.vstack([body, ladder, _floor(rng)]))
    assert info["moved"], info
    assert abs(info["front"][1] - 0.45) < 0.04, info
    assert nb.size[1] < low.size[1]
    print("PASS inward face snap ignores overhead height guard")


def test_top_trim_removes_sparse_overhead_tail():
    """A low-density tail above the body is trimmed, never extended."""
    rng = np.random.default_rng(28)
    body = np.vstack([_sheet(rng, 0.55, z_hi=2.0),
                      _sheet(rng, -0.55, z_hi=2.0)])
    tail = np.column_stack([rng.uniform(-0.8, 0.8, 40),
                            rng.uniform(-0.2, 0.2, 40),
                            rng.uniform(2.35, 3.0, 40)])
    box = _box(0.55, z=1.5)
    box.size = (box.size[0], box.size[1], 3.0)
    nb, info = _trim_sparse_top_for_test(box, np.vstack([body, tail]))
    assert info["moved"], info
    assert nb.size[2] < box.size[2]
    assert nb.center[2] < box.center[2]


def _trim_sparse_top_for_test(box, pts):
    from agentic_gts.tools.geometry import _trim_sparse_top
    return _trim_sparse_top(box, pts)


def test_face_snap_depth_bounds_guard():
    """Sheets that would snap the box below the minimum device depth
    are rejected -- the box is returned unchanged."""
    rng = np.random.default_rng(8)
    pts = np.vstack([_sheet(rng, 0.12), _sheet(rng, -0.12), _floor(rng)])
    b = _box(0.35)
    nb, info = snap_box_faces(b, pts)
    assert not info["moved"] and "bounds" in info.get("reason", ""), info
    print("PASS face snap depth bounds guard")


def test_snap_faces_to_mesh_scene_level():
    """Scene wrapper: the inflated box is adjusted with meta recorded,
    the clean box untouched, and the return count is right."""
    rng = np.random.default_rng(9)
    inflated_pts = np.vstack([_sheet(rng, 0.55), _sheet(rng, -0.55),
                              _plateau(rng, 0.57, 1.25), _floor(rng)])
    clean_pts = np.vstack([_sheet(rng, 0.55, along=0.5),
                           _sheet(rng, -0.55, along=0.5)])
    # two boxes in separate columns (x offset), one inflated one clean
    inf = _box(0.70)
    inf.center = (0.0, 5.0, 1.05)
    cln = _box(0.55, half_len=0.5)
    cln.center = (5.0, 0.0, 1.05)
    off_inf = inflated_pts + np.array([0.0, 5.0, 0.0])
    off_cln = clean_pts + np.array([5.0, 0.0, 0.0])
    scene = Scene(points=np.vstack([off_inf, off_cln]))
    scene.boxes = [inf, cln]
    n = snap_faces_to_mesh(scene)
    assert n == 1, f"exactly the inflated box adjusts, got {n}"
    adj = scene.boxes[0]
    assert abs(adj.size[1] - 1.11) < 0.06
    assert "face_snap" in adj.meta and adj.meta["face_snap"]["moved"]
    assert "face_snap" not in scene.boxes[1].meta
    assert abs(scene.boxes[1].size[1] - 1.10) < 1e-9, \
        "the clean box must be untouched"
    print("PASS snap_faces_to_mesh scene level (1 adjusted, 1 clean)")


# ---------- row-END snap tests ----------

def _row(rng, lo, hi, cross=0.55, n_per_m=900):
    """A row plateau: front+back sheets spanning [lo, hi] along."""
    n = max(400, int((hi - lo) * n_per_m))
    front = np.column_stack([rng.uniform(lo, hi, n),
                             rng.uniform(cross - 0.01, cross + 0.01, n),
                             rng.uniform(0.35, 2.00, n)])
    back = np.column_stack([rng.uniform(lo, hi, n),
                            rng.uniform(-cross - 0.01, -cross + 0.01, n),
                            rng.uniform(0.35, 2.00, n)])
    return np.vstack([front, back])


def _bleed(rng, lo, hi, n=60, cross=0.55):
    """Sparse mask-bleed points past a row end (each cross bin gets a
    couple of points -- far under every strength bar)."""
    return np.column_stack([rng.uniform(lo, hi, n),
                            rng.uniform(-cross, cross, n),
                            rng.uniform(0.35, 2.00, n)])


def _end_box(lo, hi, cross=0.55):
    return OrientedBox(center=(0.5 * (lo + hi), 0.0, 1.05),
                       size=(hi - lo, 2.0 * cross, 2.10), yaw=0.0)


def test_end_snap_trims_mask_bleed():
    """A standalone row whose right end is inflated by mask bleed
    snaps back to the plateau edge; the clean left end stays."""
    rng = np.random.default_rng(11)
    pts = np.vstack([_row(rng, -3.0, 3.0), _floor(rng),
                     _bleed(rng, 3.02, 3.28)])
    b = _end_box(-3.0, 3.30)
    nb, info = snap_box_ends(b, pts)
    assert info["moved"], info
    assert abs(info["right"][1] - 3.0) < 0.06, \
        f"the right end must trim to the plateau edge (~3.0), " \
        f"got {info['right'][1]}"
    assert abs(info["left"][1] - info["left"][0]) < 1e-9, \
        "the clean left end must not move"
    print(f"PASS end snap trims bleed (right 3.30 -> "
          f"{info['right'][1]:.2f})")


def test_end_snap_extends_under_measured():
    """An under-measured end (the true edge visible inside the tight
    outward window) extends out to the plateau edge."""
    rng = np.random.default_rng(12)
    pts = np.vstack([_row(rng, -3.0, 3.0), _floor(rng)])
    b = _end_box(-3.0, 2.90)
    nb, info = snap_box_ends(b, pts)
    assert info["moved"], info
    assert abs(info["right"][1] - 3.0) < 0.06, \
        f"the right end must extend to ~3.0, got {info['right'][1]}"
    print(f"PASS end snap extends under-measured (right 2.90 -> "
          f"{info['right'][1]:.2f})")


def test_end_snap_row_pieces_only_outer_ends():
    """User direction: a SPLIT row's internal seam is never touched
    (snap_row_seams owns it); only the row's outermost ends snap --
    even when the internal face carries bleed."""
    rng = np.random.default_rng(13)
    pts = np.vstack([
        _row(rng, -3.0, 3.0),
        _floor(rng),
        _bleed(rng, 0.02, 0.15),        # bleed past A's right (seam) face
        _bleed(rng, 3.02, 3.28)])       # bleed past B's right (outer) face
    a = _end_box(-3.0, 0.15)            # A: left free, right = seam
    b = _end_box(0.0, 3.30)             # B: left = seam, right free
    scene = Scene(points=pts)
    scene.boxes = [a, b]
    n = snap_faces_to_mesh(scene)
    ea = scene.boxes[0]
    eb = scene.boxes[1]
    assert abs(ea.size[0] - 3.15) < 1e-9, \
        f"A's internal seam face must stay (bleed and all), " \
        f"got length {ea.size[0]:.3f}"
    assert abs(ea.center[0] - (-1.425)) < 1e-9, "A must not shift"
    assert abs((eb.center[0] + eb.size[0] / 2.0) - 3.0) < 0.06, \
        f"B's OUTER right end must trim to ~3.0, got " \
        f"{eb.center[0] + eb.size[0] / 2.0:.2f}"
    assert abs((eb.center[0] - eb.size[0] / 2.0) - 0.0) < 1e-9, \
        "B's internal seam face must stay at 0.0"
    assert "end_snap" in eb.meta and "end_snap" not in ea.meta
    print("PASS end snap: split row -- only the outer ends move")


def test_end_snap_no_move_when_edge_not_visible():
    """Under-measured beyond the outward window: the plateau reaches
    the window's outer boundary, the true edge is not visible -- the
    face does not move (no snap to an arbitrary window edge)."""
    rng = np.random.default_rng(14)
    pts = np.vstack([_row(rng, -3.0, 3.0), _floor(rng)])
    b = _end_box(-3.0, 2.70)            # 0.30 short -- past the 0.15 window
    nb, info = snap_box_ends(b, pts)
    assert not info["moved"], \
        f"an edge beyond the window must not move the face: {info}"
    print("PASS end snap no-move when the edge is not visible")


def test_end_snap_union_not_amputated():
    """A union box spanning row + gap + clump (post long-split removal
    these ride whole; splitting is stageC's job): the end face sits on
    the OUTER structure's edge and must not be pulled back to the
    row's edge -- no amputation."""
    rng = np.random.default_rng(15)
    pts = np.vstack([_row(rng, 0.0, 3.0),
                     _row(rng, 4.0, 5.0, n_per_m=1800),   # the clump
                     _floor(rng)])
    b = _end_box(0.0, 5.0)
    nb, info = snap_box_ends(b, pts)
    assert not info["moved"], \
        f"the union's ends are already on the outer edges: {info}"
    assert abs(nb.size[0] - 5.0) < 1e-9, "the union must stay whole"
    print("PASS end snap: union box not amputated")


# ---------- nested trim + own-point profile tests ----------

def test_nested_trim_cuts_over_covering_big_box():
    """The user's case: ONE VLM rect covering a row + the small device
    beside it left a BIG box still covering the device while the
    device's own small box also survived (two detections at the
    device). The big box's face on the device's side is cut back to
    the small box's near boundary, then -- with the device's points
    excluded from its profile -- the cross snap lands on the ROW's own
    sheet. The small box keeps its territory."""
    rng = np.random.default_rng(21)
    # the row: front/back sheets at cross +/-0.55, along [-2, 2]
    row = np.vstack([_sheet(rng, 0.55, along=2.0),
                     _sheet(rng, -0.55, along=2.0)])
    # the small device beside the row's front side: a shell at
    # cross 0.64 / 1.20 (deterministic peaks for its own snap)
    dev = np.vstack([
        np.column_stack([rng.uniform(-0.4, 0.4, 400),
                         rng.uniform(0.63, 0.65, 400),
                         rng.uniform(0.35, 1.00, 400)]),
        np.column_stack([rng.uniform(-0.4, 0.4, 400),
                         rng.uniform(1.19, 1.21, 400),
                         rng.uniform(0.35, 1.00, 400)])])
    pts = np.vstack([row, dev, _floor(rng)])
    big = OrientedBox(center=(0.0, 0.35, 1.05),
                      size=(4.0, 1.90, 2.10), yaw=0.0)
    small = OrientedBox(center=(0.0, 0.92, 0.675),
                        size=(0.88, 0.68, 0.75), yaw=0.0)
    scene = Scene(points=pts)
    scene.boxes = [big, small]
    trim_nested_boxes(scene)
    snap_faces_to_mesh(scene)
    b, s = scene.boxes
    assert "nested_trim" in b.meta, \
        "the over-covering big box must be trimmed"
    assert abs(b.size[1] - 1.11) < 0.06, \
        f"the big box's depth must land on the ROW's sheets (~1.11), " \
        f"got {b.size[1]:.2f} -- the device's points must be excluded " \
        f"from its profile"
    bmax = b.center[1] + b.size[1] / 2.0
    assert bmax < 0.62, \
        f"the big box must be off the device (front {bmax:.2f})"
    # the small box keeps its own territory beside the row
    smin = s.center[1] - s.size[1] / 2.0
    assert smin > 0.55, \
        f"the small box must keep its territory (min cross {smin:.2f})"
    assert abs(s.center[0]) < 0.1, "the small box stays put laterally"
    print(f"PASS nested trim (big cut to {bmax:.2f} then snapped to "
          f"the row, depth {b.size[1]:.2f}; small intact)")


def test_nested_trim_guard_phantom_small_box():
    """A PHANTOM small box drawn inside a CORRECT big box must not
    trigger the trim: the cut region is the big box's own mass (the
    row's sheet + interior, mostly outside the phantom) and the >= 80%
    guard rejects -- the big box is left alone."""
    rng = np.random.default_rng(22)
    pts = np.vstack([_sheet(rng, 0.55, along=2.0),
                     _sheet(rng, -0.55, along=2.0), _floor(rng)])
    big = OrientedBox(center=(0.0, 0.0, 1.05),
                      size=(4.0, 1.10, 2.10), yaw=0.0)
    phantom = OrientedBox(center=(0.0, 0.25, 1.0),
                          size=(1.0, 0.50, 2.0), yaw=0.0)
    scene = Scene(points=pts)
    scene.boxes = [big, phantom]
    trim_nested_boxes(scene)
    snap_faces_to_mesh(scene)
    b = scene.boxes[0]
    assert "nested_trim" not in b.meta, \
        "a phantom small box inside a correct big box must not trim it"
    assert abs(b.size[1] - 1.10) < 0.06 and abs(b.size[0] - 4.0) < 0.06, \
        f"the correct big box must be untouched, got " \
        f"{b.size[0]:.2f} x {b.size[1]:.2f}"
    print("PASS nested trim guard (phantom small box rejected)")


def test_nested_trim_skips_sam_split_pieces():
    """Independent pieces from one SAM split must not be treated as a
    nested duplicate pair by the final face polish."""
    rng = np.random.default_rng(24)
    pts = np.vstack([_row(rng, -3.0, 3.0), _floor(rng)])
    left = _end_box(-3.0, 0.2)
    right = _end_box(0.1, 3.0)
    left.meta.update({"sam_refined": True, "sam_split_seed": "seed-1"})
    right.meta.update({"sam_refined": True, "sam_split_seed": "seed-1"})
    scene = Scene(points=pts)
    scene.boxes = [left, right]
    snap_faces_to_mesh(scene)
    assert "nested_trim" not in left.meta
    assert "nested_trim" not in right.meta
    print("PASS nested trim skips independent SAM split pieces")


def test_end_snap_inward_priority_over_ladder():
    """INWARD PRIORITY (user report: the end face EXTENDED onto the
    cable ladder beside the row's end -- the old farthest-run
    tie-break sought the outermost mass and found the ladder). A face
    sitting past the row's own mass must trim INWARD to the plateau
    edge; the ladder beyond the gap is a neighbour and must never
    attract the face, however dense it is."""
    rng = np.random.default_rng(23)
    # the row: plateau along [-3, 3]
    pts = np.vstack([_row(rng, -3.0, 3.0), _floor(rng)])
    # a vertical cable ladder just beyond the row's end (along
    # [3.12, 3.22]): rails+rungs density -- ABOVE every strength bar
    # (a real mesh structure, enough to attract the old farthest-run
    # rule) but realistic, not denser than the row's own sheets
    ladder = np.column_stack([rng.uniform(3.12, 3.22, 400),
                              rng.uniform(-0.55, 0.55, 400),
                              rng.uniform(0.35, 2.00, 400)])
    pts = np.vstack([pts, ladder])
    # the end face sits in the GAP past the row's true end (3.0),
    # before the ladder: the old rule sought the farthest run (the
    # ladder) and extended onto it
    b = _end_box(-3.0, 3.10)
    nb, info = snap_box_ends(b, pts)
    if info.get("moved"):
        assert info["right"][1] < 3.06, \
            f"the face must trim INWARD to the row's edge (~3.0), " \
            f"not extend onto the ladder: {info}"
    else:
        pass  # a no-op is also acceptable (both beat extending)
    end = nb.center[0] + nb.size[0] / 2.0
    assert end < 3.06, \
        f"the box end must stay off the ladder (got {end:.2f})"
    print(f"PASS end snap inward priority (end {end:.2f}, ladder "
          "ignored)")


def test_end_snap_density_cliff_splits_ladder_from_row():
    """DENSITY CLIFF on the on-mass path (user report: the face sat ON
    the cable ladder beside the row's end -- the gap's density was
    above the 30% bar, row + gap + ladder formed ONE qualifying run,
    and the old 'take the run's outer edge' logic extended onto the
    ladder's far end). The cliff scan finds the row's true end where
    the density drops from plateau level to gap level."""
    rng = np.random.default_rng(24)
    # the row: dense plateau along [-3, 3] (~27 pts/bin from two
    # full-height face sheets in the cross-slice)
    row = np.vstack([
        np.column_stack([rng.uniform(-3.0, 3.0, 2700),
                         rng.uniform(0.54, 0.56, 2700),
                         rng.uniform(0.35, 2.00, 2700)]),
        np.column_stack([rng.uniform(-3.0, 3.0, 2700),
                         rng.uniform(-0.56, -0.54, 2700),
                         rng.uniform(0.35, 2.00, 2700)])])
    # sparse bridge: gap density ABOVE the 30% qualifying bar (~9/bin,
    # so the gap bins qualify and row+gap+ladder merge into ONE run)
    # but WELL BELOW the row's peak (~27/bin, so the 50% cliff fires)
    bridge = np.column_stack([rng.uniform(3.0, 3.1, 35),
                              rng.uniform(-0.55, 0.55, 35),
                              rng.uniform(0.35, 2.00, 35)])
    # the ladder: a bit denser than the bridge but still below the
    # row's cliff threshold (50% x 27 = 13.5/bin)
    ladder = np.column_stack([rng.uniform(3.1, 3.2, 45),
                              rng.uniform(-0.55, 0.55, 45),
                              rng.uniform(0.35, 2.00, 45)])
    pts = np.vstack([row, bridge, ladder])
    # the face sits ON the gap/bridge (on-mass path -- the gap bins
    # qualify above the 30% bar, so the face bin is inside the merged
    # run)
    b = _end_box(-3.0, 3.10)
    nb, info = snap_box_ends(b, pts)
    assert info["moved"], \
        f"the density cliff must trim the face to the row's edge: {info}"
    assert info["right"][1] < 3.06, \
        f"the face must stop at the row's edge (~3.0), not the " \
        f"ladder's far end: {info['right']}"
    end = nb.center[0] + nb.size[0] / 2.0
    assert end < 3.06, \
        f"the box end must stay off the ladder (got {end:.2f})"
    print(f"PASS end snap density cliff (end {end:.2f}, row edge "
          "~3.0, ladder ignored)")
