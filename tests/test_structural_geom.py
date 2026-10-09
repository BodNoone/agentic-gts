"""Tests for the structural geometry filter (post-processing wall and
pillar detection by shape, replacing the reverted VLM structural
class): a box is LOW when it is BOTH taller than any real device
(>2.80m) AND has no cable-tray mesh points above it."""
import os
import sys

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from agentic_gts.core.models import (BoxSource, Confidence, OrientedBox, Scene)
from agentic_gts.tools.geometry import (
    filter_low_furniture_by_geometry, filter_structural_by_geometry,
    has_overhead_structure)


def _box(cx, cy, h, w=1.1, d=0.6):
    return OrientedBox(center=(cx, cy, h / 2.0), size=(w, d, h), yaw=0.0)


def _pts_above(rng, box, z_lo, z_hi, n=100, pad=0.5):
    """Tray/ladder points above a box."""
    cx, cy = box.center[0], box.center[1]
    return np.column_stack([rng.uniform(cx - pad, cx + pad, n),
                            rng.uniform(cy - pad, cy + pad, n),
                            rng.uniform(z_lo, z_hi, n)])


def test_structural_filter_flags_tall_box_without_trays():
    """A wall (3.8m tall, no cable trays above) is flagged LOW; a
    normal rack (2.1m) with trays above it is kept."""
    rng = np.random.default_rng(41)
    wall = _box(0.0, 0.0, 3.8, w=2.0, d=0.3)    # tall + thin = wall
    rack = _box(5.0, 0.0, 2.1)                   # normal height
    # trays above the rack (2.1 + 0.2 to 2.1 + 0.8 = 2.3 to 2.9)
    trays = _pts_above(rng, rack, 2.3, 2.9, n=200)
    # nothing above the wall (just air to the ceiling)
    scene = Scene(points=trays)
    scene.boxes = [wall, rack]
    n = filter_structural_by_geometry(scene)
    assert n == 1, f"exactly the wall flags, got {n}"
    assert scene.boxes[0].confidence == Confidence.LOW
    assert "structural_geom" in scene.boxes[0].meta
    assert scene.boxes[1].confidence != Confidence.LOW
    print("PASS structural filter: tall wall flagged, rack with "
          "trays kept")


def test_structural_filter_keeps_tall_rack_with_trays():
    """A tall rack (2.9m -- just above the 2.8m bar) WITH cable trays
    above it is NOT flagged (the overhead check saves it)."""
    rng = np.random.default_rng(42)
    tall_rack = _box(0.0, 0.0, 2.9)              # above the 2.8m bar
    trays = _pts_above(rng, tall_rack, 3.1, 3.7, n=200)
    scene = Scene(points=trays)
    scene.boxes = [tall_rack]
    n = filter_structural_by_geometry(scene)
    assert n == 0, f"a tall rack WITH trays must not flag, got {n}"
    assert tall_rack.confidence != Confidence.LOW
    print("PASS structural filter: tall rack with trays kept")


def test_structural_filter_keeps_short_box_without_trays():
    """A short box (1.0m -- a low AC unit) without trays is NOT
    flagged (the height check saves it: only tall boxes are
    candidates)."""
    rng = np.random.default_rng(43)
    ac = _box(0.0, 0.0, 1.0, w=0.8, d=0.8)       # short = not tall
    scene = Scene(points=np.zeros((10, 3)))
    scene.boxes = [ac]
    n = filter_structural_by_geometry(scene)
    assert n == 0, f"a short box without trays must not flag, got {n}"
    print("PASS structural filter: short box without trays kept")


def test_has_overhead_structure():
    """The overhead check: mesh points in the band above the box,
    within its padded footprint."""
    rng = np.random.default_rng(44)
    box = _box(0.0, 0.0, 2.0)
    # trays above the box (2.2 to 2.8, within the 0.30 padded footprint)
    trays = _pts_above(rng, box, 2.2, 2.8, n=200, pad=0.3)
    scene = Scene(points=trays)
    assert has_overhead_structure(scene, box), \
        "tray points above the box must be detected"
    # nothing above
    empty = Scene(points=np.zeros((10, 3)))
    assert not has_overhead_structure(empty, box), \
        "no points above -> no overhead structure"
    # points above but far away (outside the padded footprint)
    far = _pts_above(rng, OrientedBox(center=(10, 10, 1), size=(1, 1, 2),
                                      yaw=0.0), 2.2, 2.8, n=200)
    far_scene = Scene(points=far)
    assert not has_overhead_structure(far_scene, box), \
        "points above but outside the footprint -> no overhead"
    print("PASS has_overhead_structure (band + footprint check)")


def test_low_furniture_filter_drops_low_raw_box_but_keeps_refined_piece():
    rng = np.random.default_rng(45)
    raw = _box(0.0, 0.0, 0.9, w=1.2, d=0.7)
    refined = _box(5.0, 0.0, 0.9, w=1.2, d=0.7)
    refined.source = BoxSource.AGENT_FIX
    # Sparse low furniture points: a few tabletop / leg samples only.
    sparse = np.column_stack([rng.uniform(-0.6, 0.6, 50),
                              rng.uniform(-0.35, 0.35, 50),
                              rng.uniform(0.05, 0.9, 50)])
    scene = Scene(points=sparse)
    scene.boxes = [raw, refined]
    n = filter_low_furniture_by_geometry(scene)
    assert n == 1
    assert raw.confidence == Confidence.LOW
    assert "low_furniture_geom" in raw.meta
    assert refined.confidence != Confidence.LOW
    print("PASS low furniture filter protects refined pieces")


def test_low_furniture_filter_keeps_solid_battery_cabinet_without_trays():
    rng = np.random.default_rng(46)
    battery = _box(0.0, 0.0, 0.9, w=0.8, d=0.65)
    # Dense samples on the cabinet's four vertical body faces.
    faces = []
    for x in (-0.4, 0.4):
        faces.append(np.column_stack([
            np.full(700, x), rng.uniform(-0.325, 0.325, 700),
            rng.uniform(0.0, 0.9, 700)]))
    for y in (-0.325, 0.325):
        faces.append(np.column_stack([
            rng.uniform(-0.4, 0.4, 700), np.full(700, y),
            rng.uniform(0.0, 0.9, 700)]))
    scene = Scene(points=np.vstack(faces))
    scene.boxes = [battery]
    n = filter_low_furniture_by_geometry(scene)
    assert n == 0
    assert "low_furniture_geom" not in battery.meta
    print("PASS low furniture filter keeps solid battery cabinet")
