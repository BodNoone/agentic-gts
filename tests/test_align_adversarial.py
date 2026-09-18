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


def test_estimate_residual_yaw():
    """Closed-loop residual: re-estimating on the -yaw-rotated cloud.

    1. residual of the estimator's OWN answer must be ~0 (rotated
       rows are axis-aligned -- the self-consistency the hijacked
       first pass cannot fake);
    2. a WRONG yaw claim (0 on a 20-deg layout) must be exposed: the
       rotated rows sit at the error angle and the re-estimate
       returns it as the residual."""
    from agentic_gts.segment.orientation import estimate_residual_yaw
    scene, _, _ = generate(SynthConfig(seed=7, room_yaw_deg=20))
    pts = scene.points

    y = estimate_yaw(pts)
    r_self = estimate_residual_yaw(pts, y)
    assert abs(math.degrees(r_self)) < 2.0, \
        f"self-consistent yaw showed residual {math.degrees(r_self):.1f} deg"

    r_wrong = estimate_residual_yaw(pts, 0.0)
    assert 12.0 < math.degrees(r_wrong) < 28.0, \
        f"20-deg layout claimed as 0 must expose ~20 deg residual, " \
        f"got {math.degrees(r_wrong):.1f}"
    print(f"PASS residual yaw (self={math.degrees(r_self):.1f} deg, "
          f"wrong-claim={math.degrees(r_wrong):.1f} deg)")


def test_seed_axis_delta():
    """StageG feedback signal: the seeds' OWN PCA yaws vote directly
    (weighted median, folded mod-90). The old feedback re-ran the
    global histogram estimator on points inside the fitted boxes --
    a pool carved along the ASSUMED yaw, so slanted rows re-confirmed
    the assumed yaw (user report: imperfect yaw surviving feedback).
    """
    from agentic_gts.core.models import OrientedBox
    from agentic_gts.segment.orientation import seed_axis_delta

    def _box(yaw_deg, n_pts, length=6.0):
        return OrientedBox(center=(0.0, 0.0, 1.0), size=(length, 1.0, 2.0),
                           yaw=math.radians(yaw_deg), meta={"n_pts": n_pts})

    # three rows slanted 8 deg off the render yaw: median ~ +8 deg
    boxes = [_box(8.0, 900), _box(8.2, 800), _box(7.8, 700)]
    d = seed_axis_delta(boxes, cur_yaw=0.0)
    assert d is not None and abs(math.degrees(d) - 8.0) < 0.5, \
        f"three agreeing seeds must vote their own axis, got {d}"

    # perpendicular rows agree (mod-90 fold): +98 deg == +8 deg
    boxes = [_box(8.0, 900), _box(98.0, 800)]
    d = seed_axis_delta(boxes, cur_yaw=0.0)
    assert d is not None and abs(math.degrees(d) - 8.0) < 0.5, \
        f"perpendicular rows must fold to the same direction, got {d}"

    # a heavy stray fit cannot drag the median: two big rows at +8,
    # one heavier wall-ish box at -20 -> still ~+8
    boxes = [_box(8.0, 900), _box(8.0, 800), _box(-20.0, 1200)]
    d = seed_axis_delta(boxes, cur_yaw=0.0)
    assert d is not None and abs(math.degrees(d) - 8.0) < 1.0, \
        f"weighted median must resist one stray fit, got {math.degrees(d):.1f}"

    # stubby boxes (AC units) do not vote: their PCA axis is noise
    boxes = [_box(37.0, 400, length=0.8), _box(-11.0, 300, length=1.2)]
    assert seed_axis_delta(boxes, cur_yaw=0.0) is None, \
        "no box >= 2m long -> no trustworthy vote -> None"

    # already-straight rows: delta ~ 0 (feedback never fires)
    boxes = [_box(0.4, 900), _box(-0.3, 800)]
    d = seed_axis_delta(boxes, cur_yaw=0.0)
    assert d is not None and abs(math.degrees(d)) < 1.0
    print(f"PASS seed axis delta ({math.degrees(d):+.1f} deg on straight, "
          f"+8 recovered on slanted)")


def test_align_with_subfloor_noise():
    """Regression: marginal noise spike BELOW the floor must not win.

    Real 3DGS exports carry floaters under the floor; a bottom-up
    first-above-threshold scan latches onto them and shifts the whole cloud
    up (floor lands inside the device height band, drowning the device rows).
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


if __name__ == "__main__":
    test_align_and_yaw_on_adversarial_cloud()
    print("PASS  test_align_and_yaw_on_adversarial_cloud")
    test_align_with_subfloor_noise()
    print("PASS  test_align_with_subfloor_noise")
