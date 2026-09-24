"""Tests for the row-completion recall fallback (complete_row_gaps).

Covers:
  - interior gap fill: a cabinet the VLM never grounded, standing
    between two fitted boxes, is recovered by the point-support probe
  - row-end walk: point support continuing past the last box extends
    the row
  - no fill without support: an empty gap stays empty
  - wall guard: a thin partition running past the row end is NOT
    filled (fills only one probe axis; a cabinet fills both)
"""
from __future__ import annotations

import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from agentic_gts.core.models import (BoxSource, Confidence, OrientedBox,
                                      Scene)
from agentic_gts.tools.geometry import complete_row_gaps, snap_row_seams


def _cabinet(cx, cy=0.0, rng=None, w=0.6, d=1.1, h=2.1, n=1500):
    """Surface points of one closed cabinet at (cx, cy)."""
    rng = rng or np.random.default_rng(0)
    half_d = d / 2.0
    pts = []
    for face in (+half_d, -half_d):     # front / back bands
        pts.append(np.column_stack([rng.uniform(cx - w / 2, cx + w / 2, n),
                                    np.full(n, cy + face),
                                    rng.uniform(0.0, h, n)]))
    # side walls
    for side in (cx - w / 2, cx + w / 2):
        pts.append(np.column_stack([np.full(n, side),
                                    rng.uniform(cy - half_d, cy + half_d, n),
                                    rng.uniform(0.0, h, n)]))
    return np.vstack(pts)


def _scene(boxes, points, yaw=0.0):
    scene = Scene(points=points)
    scene.meta["yaw"] = yaw
    scene.boxes = list(boxes)
    return scene


def _box(cx, cy=0.0, w=0.6, d=1.1, h=2.1):
    return OrientedBox(center=(cx, cy, h / 2.0), size=(w, d, h), yaw=0.0)


def test_interior_gap_filled():
    """Cabinets at x=0 and x=1.2, a THIRD one (points present) at x=0.6
    that grounding missed -> the probe recovers it."""
    rng = np.random.default_rng(1)
    pts = np.vstack([_cabinet(0.0, rng=rng), _cabinet(0.6, rng=rng),
                     _cabinet(1.2, rng=rng)])
    scene = _scene([_box(0.0), _box(1.2)], pts)
    added = complete_row_gaps(scene)
    assert len(added) == 1, f"expected the missed middle cabinet, got {len(added)}"
    b = added[0]
    assert abs(b.center[0] - 0.6) < 0.15, f"fill at {b.center[0]:.2f}, want ~0.6"
    assert abs(b.center[1]) < 0.15
    assert b.source == BoxSource.ROW_COMPLETION
    assert b.confidence == Confidence.LOW
    assert len(scene.boxes) == 3
    print("PASS interior gap fill (missed middle cabinet recovered)")


def test_row_end_walk():
    """One fitted cabinet at x=0, point support continues to x=0.6 ->
    the end walk adds it (and stops: nothing beyond)."""
    rng = np.random.default_rng(2)
    pts = np.vstack([_cabinet(0.0, rng=rng), _cabinet(0.6, rng=rng)])
    scene = _scene([_box(0.0)], pts)
    added = complete_row_gaps(scene)
    assert len(added) == 1, f"expected +1 at the row end, got {len(added)}"
    assert abs(added[0].center[0] - 0.6) < 0.15
    print("PASS row end walk (support beyond the last box extends the row)")


def test_empty_gap_not_filled():
    """Cabinets at x=0 and x=2.4 with NOTHING in between (a real
    aisle cut through the row) -> no fill."""
    rng = np.random.default_rng(3)
    pts = np.vstack([_cabinet(0.0, rng=rng), _cabinet(2.4, rng=rng)])
    scene = _scene([_box(0.0), _box(2.4)], pts)
    added = complete_row_gaps(scene)
    assert added == [], f"empty gap must stay empty, got {len(added)} fill(s)"
    print("PASS empty gap not filled (no support -> no box)")


def test_wall_past_row_end_not_filled():
    """A thin partition (0.2m) running along the row, past its end: it
    has height and plenty of points, but only fills the probe's CROSS
    axis (thin in along) -- the span guard must reject it."""
    rng = np.random.default_rng(4)
    cabinet_pts = _cabinet(0.0, rng=rng)
    # wall band from x=0.5 to x=2.5 at y=0, 0.2m thick, 2.6m tall
    wall = np.column_stack([rng.uniform(0.5, 2.5, 4000),
                            rng.uniform(-0.1, 0.1, 4000),
                            rng.uniform(0.0, 2.6, 4000)])
    scene = _scene([_box(0.0)], np.vstack([cabinet_pts, wall]))
    added = complete_row_gaps(scene)
    assert added == [], \
        f"a thin wall slice must not become a cabinet, got {len(added)}"
    print("PASS wall past row end not filled (footprint span guard)")


def test_rotated_frame():
    """Same interior-gap scenario, whole layout rotated 30 deg in world
    frame (yaw in meta): the fill lands in the rotated position."""
    import math
    yaw = math.radians(30.0)
    rng = np.random.default_rng(5)
    c, s = math.cos(yaw), math.sin(yaw)

    def rot(p):
        return np.column_stack([c * p[:, 0] - s * p[:, 1],
                                s * p[:, 0] + c * p[:, 1], p[:, 2]])

    pts = rot(np.vstack([_cabinet(0.0, rng=rng), _cabinet(0.6, rng=rng),
                         _cabinet(1.2, rng=rng)]))
    # boxes: long axis along the rotated x = yaw
    b0 = OrientedBox(center=(0.0, 0.0, 1.05), size=(0.6, 1.1, 2.1), yaw=yaw)
    b2 = OrientedBox(center=(c * 1.2, s * 1.2, 1.05), size=(0.6, 1.1, 2.1),
                     yaw=yaw)
    scene = _scene([b0, b2], pts, yaw=yaw)
    added = complete_row_gaps(scene)
    assert len(added) == 1, f"expected the rotated middle cabinet, got {len(added)}"
    ex, ey = c * 0.6, s * 0.6
    assert math.hypot(added[0].center[0] - ex, added[0].center[1] - ey) < 0.15
    print("PASS rotated frame (fill follows the row axis)")


def test_snap_row_seams_closes_small_gap():
    """Two split pieces whose seam landed 6cm apart -> snapped to the
    average, so the neighbours share ONE edge."""
    a = OrientedBox(center=(0.0, 0.0, 1.05), size=(0.6, 1.1, 2.1), yaw=0.0)
    b = OrientedBox(center=(0.66, 0.0, 1.05), size=(0.6, 1.1, 2.1), yaw=0.0)
    n = snap_row_seams([a, b], 0.0)
    assert n == 1, f"expected one snapped seam, got {n}"
    a_hi = a.center[0] + a.size[0] / 2.0
    b_lo = b.center[0] - b.size[0] / 2.0
    assert abs(a_hi - b_lo) < 1e-9, f"seam not shared ({a_hi:.3f} vs {b_lo:.3f})"
    assert abs(a_hi - 0.33) < 1e-9, f"seam must be the average, got {a_hi:.3f}"
    print("PASS snap row seams (small gap averaged to a shared edge)")


def test_snap_row_seams_overlap_averaged():
    """A slight overlap (mask bleed) is normalised to the midpoint too."""
    a = OrientedBox(center=(0.0, 0.0, 1.05), size=(0.6, 1.1, 2.1), yaw=0.0)
    b = OrientedBox(center=(0.55, 0.0, 1.05), size=(0.6, 1.1, 2.1), yaw=0.0)
    n = snap_row_seams([a, b], 0.0)
    assert n == 1
    a_hi = a.center[0] + a.size[0] / 2.0
    b_lo = b.center[0] - b.size[0] / 2.0
    assert abs(a_hi - b_lo) < 1e-9
    assert abs(a_hi - 0.275) < 1e-9, f"seam must be the midpoint, got {a_hi:.3f}"
    print("PASS snap row seams (overlap averaged)")


def test_snap_row_seams_leaves_real_aisle():
    """A 0.5m aisle between two cabinets is NOT a seam: untouched."""
    a = OrientedBox(center=(0.0, 0.0, 1.05), size=(0.6, 1.1, 2.1), yaw=0.0)
    b = OrientedBox(center=(1.1, 0.0, 1.05), size=(0.6, 1.1, 2.1), yaw=0.0)
    n = snap_row_seams([a, b], 0.0)
    assert n == 0, f"a real aisle must not be snapped, got {n}"
    assert abs(a.center[0]) < 1e-9 and abs(b.center[0] - 1.1) < 1e-9
    print("PASS snap row seams (real aisle left alone)")


def test_snap_row_seams_unifies_cross_vertices():
    """Pieces whose facing CORNER PAIRS are each close get their shared
    edge's cross vertices averaged -> one collinear edge."""
    a = OrientedBox(center=(0.0, 0.0, 1.05), size=(0.6, 1.1, 2.1), yaw=0.0)
    b = OrientedBox(center=(0.6, 0.06, 1.05), size=(0.6, 1.06, 2.1), yaw=0.0)
    n = snap_row_seams([a, b], 0.0)
    assert n == 1
    # a corners: -0.55 / +0.55 ; b corners: -0.47 / +0.59 -> merged
    assert abs(a.center[1] - 0.03) < 1e-9 and abs(b.center[1] - 0.03) < 1e-9
    assert abs(a.size[1] - 1.08) < 1e-9 and abs(b.size[1] - 1.08) < 1e-9
    print("PASS snap row seams (cross vertices averaged, collinear edge)")


def test_snap_row_seams_keeps_far_cross():
    """When one facing corner pair is far apart, the cross extent is
    NOT merged -- only the along seam is normalised, and the different
    heights are irrelevant to this 2D edge rule."""
    a = OrientedBox(center=(0.0, 0.0, 1.05), size=(0.6, 1.1, 2.1), yaw=0.0)
    b = OrientedBox(center=(0.6, 0.0, 1.05), size=(0.6, 0.6, 2.1), yaw=0.0)
    n = snap_row_seams([a, b], 0.0)
    assert n == 1
    assert abs(a.size[1] - 1.1) < 1e-9 and abs(b.size[1] - 0.6) < 1e-9, \
        "a far corner pair must not be merged"
    a_hi = a.center[0] + a.size[0] / 2.0
    b_lo = b.center[0] - b.size[0] / 2.0
    assert abs(a_hi - b_lo) < 1e-9, "the along seam is still shared"
    print("PASS snap row seams (far cross kept, along edge still normalised)")


def test_snap_row_seams_height_step_skips():
    """Devices on different height planes (>5cm between their TOPS) are
    NEVER seamed, however small the gap (user directive: the split
    separated different-height cabinets on purpose)."""
    # tops 2.0 vs 2.2 (0.20m apart); gap 0.06m -- would snap at 0.12
    a = OrientedBox(center=(0.0, 0.0, 1.0), size=(0.6, 1.1, 2.0), yaw=0.0)
    b = OrientedBox(center=(0.66, 0.0, 1.1), size=(0.6, 1.1, 2.2), yaw=0.0)
    n = snap_row_seams([a, b], 0.0)
    assert n == 0, "a 6cm gap across a 20cm height step must NOT snap"
    assert abs(a.center[0]) < 1e-9 and abs(b.center[0] - 0.66) < 1e-9, \
        "both boxes must stay untouched"
    # even a truly TOUCHING seam (gap 0.04) does not snap across a step
    c = OrientedBox(center=(0.0, 0.0, 1.0), size=(0.6, 1.1, 2.0), yaw=0.0)
    d = OrientedBox(center=(0.64, 0.0, 1.1), size=(0.6, 1.1, 2.2), yaw=0.0)
    n2 = snap_row_seams([c, d], 0.0)
    assert n2 == 0, "no seam across a height step, even a touching one"
    assert abs(c.center[0]) < 1e-9 and abs(d.center[0] - 0.64) < 1e-9
    print("PASS snap row seams height step (never seamed across a step)")


def test_refit_box_orientations_angled():
    """Fan-shaped rooms: a device at 20 deg to the frame has its AABB
    inflated to 0.94x1.24 (true 0.6x1.1, +74% area). The per-box
    orientation refit rotates it onto its own principal axis and
    re-measures -- the footprint comes back to ~0.6x1.1 at yaw ~= 20,
    height / bottom untouched."""
    import math
    from agentic_gts.tools.geometry import refit_box_orientations
    rng = np.random.default_rng(31)
    yaw_d = math.radians(20.0)
    c, s = math.cos(yaw_d), math.sin(yaw_d)

    def dev(u, v, z):
        return np.column_stack([c * u - s * v, s * u + c * v, z])

    def face(u0, u1, v0, v1, z0, z1):
        return dev(rng.uniform(u0, u1, 6000), rng.uniform(v0, v1, 6000),
                   rng.uniform(z0, z1, 6000))

    pts = np.vstack([
        face(-0.30, 0.30, -0.55, -0.50, 0.0, 2.0),   # back sheet
        face(-0.30, 0.30, 0.50, 0.55, 0.0, 2.0),     # front sheet
        face(-0.30, -0.25, -0.55, 0.55, 0.0, 2.0),   # left wall
        face(0.25, 0.30, -0.55, 0.55, 0.0, 2.0),     # right wall
        face(-0.30, 0.30, -0.55, 0.55, 1.95, 2.0),   # top
    ])
    scene = Scene(points=pts)
    scene.meta["yaw"] = 0.0
    # the seed as the pipeline fits it: the frame AABB of the device,
    # long side on the yaw axis (1.24 > 0.94 -> yaw = pi/2)
    scene.boxes = [OrientedBox(center=(0.0, 0.0, 1.0),
                               size=(1.24, 0.94, 2.0),
                               yaw=math.pi / 2.0)]
    n = refit_box_orientations(scene)
    assert n == 1, "the angled device must be refit"
    b = scene.boxes[0]
    # yaw comes back as ~20 deg (mod 90; the depth rides the yaw axis)
    dy = math.degrees(math.remainder(b.yaw - yaw_d, math.pi / 2.0))
    assert abs(dy) < 2.0, f"refit yaw off by {dy:.1f} deg"
    assert abs(b.size[0] - 1.1) < 0.12, f"depth {b.size[0]:.2f} (want ~1.1)"
    assert abs(b.size[1] - 0.6) < 0.12, f"width {b.size[1]:.2f} (want ~0.6)"
    assert abs(b.size[2] - 2.0) < 1e-9 and abs(b.center[2] - 1.0) < 1e-9, \
        "height / bottom untouched"
    assert b.meta.get("orient_refit"), "the refit must be recorded in meta"
    print("PASS refit box orientations (angled device -> true OBB)")


def test_refit_box_orientations_aligned_untouched():
    """Row-aligned boxes -- long side on the frame axis (yaw=0) OR on
    the +90 cross axis (the single-cabinet convention) -- are untouched:
    the snap guard keeps the shared frame and per-box PCA noise can
    never wobble a straight row."""
    import math
    from agentic_gts.tools.geometry import refit_box_orientations
    rng = np.random.default_rng(32)

    def face(x0, x1, y0, y1, z0, z1):
        return np.column_stack([rng.uniform(x0, x1, 6000),
                                rng.uniform(y0, y1, 6000),
                                rng.uniform(z0, z1, 6000)])

    # a 2 m row piece (long side on frame x, yaw=0) ...
    row = np.vstack([
        face(-1.0, 1.0, -0.55, -0.50, 0.0, 2.0),
        face(-1.0, 1.0, 0.50, 0.55, 0.0, 2.0),
        face(-1.0, -0.95, -0.55, 0.55, 0.0, 2.0),
        face(0.95, 1.0, -0.55, 0.55, 0.0, 2.0),
        face(-1.0, 1.0, -0.55, 0.55, 1.95, 2.0)])
    # ... and a single cabinet (depth on frame y -> yaw=pi/2), at (3,3)
    cab = np.vstack([
        face(2.70, 3.30, 2.45, 2.50, 0.0, 2.0),
        face(2.70, 3.30, 3.50, 3.55, 0.0, 2.0),
        face(2.70, 2.75, 2.45, 3.55, 0.0, 2.0),
        face(3.25, 3.30, 2.45, 3.55, 0.0, 2.0),
        face(2.70, 3.30, 2.45, 3.55, 1.95, 2.0)])
    scene = Scene(points=np.vstack([row, cab]))
    scene.meta["yaw"] = 0.0
    scene.boxes = [
        OrientedBox(center=(0.0, 0.0, 1.0), size=(2.0, 1.1, 2.0), yaw=0.0),
        OrientedBox(center=(3.0, 3.0, 1.0), size=(1.1, 0.6, 2.0),
                    yaw=math.pi / 2.0)]
    n = refit_box_orientations(scene)
    assert n == 0, f"aligned boxes must stay untouched, refit {n}"
    assert scene.boxes[0].yaw == 0.0 \
        and scene.boxes[0].size == (2.0, 1.1, 2.0)
    assert scene.boxes[1].yaw == math.pi / 2.0 \
        and scene.boxes[1].size == (1.1, 0.6, 2.0)
    print("PASS refit box orientations (aligned boxes untouched)")


if __name__ == "__main__":
    test_interior_gap_filled()
    test_row_end_walk()
    test_empty_gap_not_filled()
    test_wall_past_row_end_not_filled()
    test_rotated_frame()
    test_snap_row_seams_closes_small_gap()
    test_snap_row_seams_overlap_averaged()
    test_snap_row_seams_leaves_real_aisle()
    test_snap_row_seams_unifies_cross_vertices()
    test_snap_row_seams_keeps_far_cross()
    test_snap_row_seams_height_step_skips()
    test_refit_box_orientations_angled()
    test_refit_box_orientations_aligned_untouched()
