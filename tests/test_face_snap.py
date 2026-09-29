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
                                        snap_faces_to_mesh)


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


def test_face_snap_door_only_window_does_not_move():
    """When the inward window contains ONLY the door plateau (the box
    is inflated beyond the window), the global strength bar rejects
    the plateau and the face does not move -- no snap onto noise."""
    rng = np.random.default_rng(7)
    pts = np.vstack([
        _sheet(rng, 0.55), _sheet(rng, -0.55),
        _plateau(rng, 0.56, 1.30, n=500),
        _floor(rng)])
    b = _box(1.00)                      # inflated far beyond the window
    nb, info = snap_box_faces(b, pts)
    assert not info["moved"], \
        f"a door-only window must not snap onto its own plateau: {info}"
    print("PASS face snap no-move on a door-only window")


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
    snap_faces_to_mesh(scene)
    b = scene.boxes[0]
    assert "nested_trim" not in b.meta, \
        "a phantom small box inside a correct big box must not trim it"
    assert abs(b.size[1] - 1.10) < 0.06 and abs(b.size[0] - 4.0) < 0.06, \
        f"the correct big box must be untouched, got " \
        f"{b.size[0]:.2f} x {b.size[1]:.2f}"
    print("PASS nested trim guard (phantom small box rejected)")


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
