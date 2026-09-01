"""Adversarial alignment/yaw test: tilted cloud + sparse floor + dense
rack-top field, mirroring failure modes seen on real 3DGS exports.

The cloud is tilted 5.7/1.5 deg, offset 0.8 m, the floor is decimated to
15% (poorly reconstructed), and a dense coplanar rack-top plane at z=2.0
covers the device layout. align_to_ground must level via the rack-top
plane's normal but set z=0 at the real floor, and estimate_yaw must pick
the row direction from vertical faces only (30 deg).
"""
import math
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np

from agentic_gts.pipeline import align_to_ground
from agentic_gts.segment.orientation import estimate_yaw
from agentic_gts.synth.generator import SynthConfig, generate


def _rot_axis(axis, deg):
    axis = np.asarray(axis, dtype=float)
    axis /= np.linalg.norm(axis)
    k = np.array([[0, -axis[2], axis[1]],
                  [axis[2], 0, -axis[0]],
                  [-axis[1], axis[0], 0]])
    t = math.radians(deg)
    return np.eye(3) + math.sin(t) * k + (1 - math.cos(t)) * (k @ k)


def test_align_and_yaw_on_adversarial_cloud():
    rng = np.random.default_rng(3)
    scene, _, _ = generate(SynthConfig(seed=42, room_yaw_deg=30))
    pts = scene.points

    floor_m = pts[:, 2] < 0.05
    keep = ~floor_m | (rng.random(len(pts)) < 0.15)
    pts = pts[keep]

    # dense rack-top plane following the (rotated) device layout
    dev_xy = pts[(pts[:, 2] > 0.1) & (pts[:, 2] < 2.0)][:, :2]
    sel = rng.integers(0, len(dev_xy), 120_000)
    tops = np.column_stack([
        dev_xy[sel, 0] + rng.normal(0, 0.15, 120_000),
        dev_xy[sel, 1] + rng.normal(0, 0.15, 120_000),
        np.full(120_000, 2.0),
    ])
    pts = np.vstack([pts, tops])

    R = _rot_axis([1, 0, 0], 5.7) @ _rot_axis([0, 1, 0], 1.5)
    adversarial = pts @ R.T + np.array([0.0, 0.0, 0.8])

    fixed = align_to_ground(adversarial)

    # floor near z=0: the bottom decile hugs the floor (racks reach it)
    zb = fixed[fixed[:, 2] < 0.25]
    assert len(zb) > 100
    assert abs(float(np.median(zb[:, 2]))) < 0.15, "floor not normalized to z~0"

    # rack tops ~2 m above the floor (rigidity sanity)
    h, e = np.histogram(fixed[:, 2], bins=np.arange(fixed[:, 2].min(),
                                                    fixed[:, 2].max() + 0.05, 0.05))
    top_bin = float(e[int(np.argmax(h))])
    assert 1.5 < top_bin < 2.5, f"rack-top plane at {top_bin:.2f} m, expected ~2.0"

    # row direction: 30 deg layout, from vertical faces only
    yaw = estimate_yaw(fixed)
    err = abs(30.0 - math.degrees(yaw))
    assert err < 3.0, f"yaw error {err:.1f} deg"


def test_align_with_subfloor_noise():
    """Regression: marginal noise spike BELOW the floor must not win.

    Real 3DGS exports carry floaters under the floor; a bottom-up
    first-above-threshold scan latches onto them and shifts the whole cloud
    up (floor lands inside the device height band, drowning Stage A rows).
    """
    rng = np.random.default_rng(7)
    scene, _, _ = generate(SynthConfig(seed=42))
    pts = scene.points
    lo, hi = pts.min(axis=0), pts.max(axis=0)

    # diffuse floater layer + a concentrated spike below the floor
    n = 40_000
    noise = np.column_stack([
        rng.uniform(lo[0], hi[0], n), rng.uniform(lo[1], hi[1], n),
        rng.uniform(-2.0, -0.3, n)])
    spike = np.column_stack([
        rng.uniform(lo[0], hi[0], 5_000), rng.uniform(lo[1], hi[1], 5_000),
        rng.uniform(-0.65, -0.60, 5_000)])
    R = _rot_axis([1, 0, 0], 3.0) @ _rot_axis([0, 1, 0], 1.0)
    adversarial = np.vstack([pts, noise, spike]) @ R.T + np.array([0, 0, 0.5])

    fixed = align_to_ground(adversarial)

    # strongest spike below 0.4 m must be the floor at z~0, not the noise layer
    h, e = np.histogram(fixed[:, 2], bins=np.arange(-3.0, 3.0, 0.05))
    m = e[:-1] < 0.4
    zc = float(e[:-1][m][int(np.argmax(h[m]))])
    assert abs(zc) < 0.15, f"floor mode at z={zc:.2f}, expected ~0"


def test_stageA_rejects_thick_noisy_wall():
    """Regression: thick/noisy GS walls are paired into a fake device row.

    A wall reconstructed with thickness yields TWO parallel cross-axis bands
    whose gap (0.5-1.5m) matches the rack-depth pairing window, so a wall
    becomes a "row" of boxes. Height is the robust discriminator: wall
    columns reach the ceiling, rack columns stop at ~2.4m.
    """
    rng = np.random.default_rng(11)
    pts = []
    # device row: 8 racks (0.6 x 1.1 x 2.0) along x, centered at y=0
    for i in range(8):
        cx = 1.0 + i * 0.7
        for face_y in (-0.55, 0.55):
            n = 1500
            pts.append(np.column_stack([
                rng.uniform(cx - 0.3, cx + 0.3, n),
                np.full(n, face_y) + rng.normal(0, 0.01, n),
                rng.uniform(0.05, 2.0, n)]))
    # floor
    pts.append(np.column_stack([
        rng.uniform(-1, 8, 20000), rng.uniform(-3, 6, 20000),
        np.abs(rng.normal(0, 0.01, 20000))]))
    # THICK noisy wall at y~3.8..4.7: two sheets 0.9m apart, full height 4m
    for wy in (3.8, 4.7):
        n = 12000
        pts.append(np.column_stack([
            rng.uniform(-1, 8, n),
            np.full(n, wy) + rng.normal(0, 0.05, n),
            rng.uniform(0.05, 4.0, n)]))
    cloud = np.vstack(pts)

    from agentic_gts.core.models import Scene
    from agentic_gts.segment.coarse import coarse_segment
    s = Scene(points=cloud, boxes=[])
    boxes = coarse_segment(s, {"yaw": 0.0})

    wall_boxes = [b for b in boxes if 3.5 < b.center[1] < 5.0]
    assert not wall_boxes, f"{len(wall_boxes)} boxes on the thick wall"
    assert any(abs(b.center[1]) < 1.0 for b in boxes), "rack row missing"


if __name__ == "__main__":
    test_align_and_yaw_on_adversarial_cloud()
    print("PASS  test_align_and_yaw_on_adversarial_cloud")
    test_align_with_subfloor_noise()
    print("PASS  test_align_with_subfloor_noise")
    test_stageA_rejects_thick_noisy_wall()
    print("PASS  test_stageA_rejects_thick_noisy_wall")
