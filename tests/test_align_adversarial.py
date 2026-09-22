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


def test_top_yaw_candidates_user_scene():
    """Fold + rank on the USER'S REAL candidate scores (two mesh runs
    of one scene, from the logs): the truth (-17.5 deg) scored 468/430
    -- strong but NEVER the argmax (winners 504/440 were both wrong,
    8-10 deg off). Both orientations of each histogram peak appear in
    the list with different band scores (80.5 and -9.5 are the SAME
    direction mod 90); folding must merge them, keep the best score,
    and the top-3 must CONTAIN the truth in both runs -- that list is
    what the grounding-yield arbitration tries."""
    from agentic_gts.segment.orientation import top_yaw_candidates

    def _near(got, want_deg):
        return any(abs(math.degrees(w) - want_deg) < 2.0 for w, _ in got)

    # run 1 pass-1 candidate scores (verbatim from the log)
    run1 = [(72.5, 201), (-17.5, 468), (80.5, 504), (-9.5, 143),
            (63.5, 472), (-26.5, 305), (45.5, 336)]
    top1 = top_yaw_candidates({"candidates": run1}, k=3)
    assert len(top1) == 3
    assert _near(top1, -17.5), \
        f"run-1 truth -17.5 deg must be in the top-3, got {top1}"
    # 80.5 and -9.5 fold together -> ONE -9.5 entry at the max score
    d95 = [s for w, s in top1 if abs(math.degrees(w) + 9.5) < 2.0]
    assert d95 and d95[0] == 504, "both orientations must merge at max"

    # run 2 pass-1 candidate scores (verbatim from the log)
    run2 = [(72.5, 242), (-17.5, 430), (80.5, 412), (-9.5, 168),
            (63.5, 399), (-26.5, 440), (47.5, 261)]
    top2 = top_yaw_candidates({"candidates": run2}, k=3)
    assert _near(top2, -17.5), \
        f"run-2 truth -17.5 deg must be in the top-3, got {top2}"
    # 63.5 folds to -26.5: the two entries merge at 440, not 399
    d265 = [s for w, s in top2 if abs(math.degrees(w) + 26.5) < 2.0]
    assert d265 and d265[0] == 440, "63.5 must merge into -26.5 at max"
    print("PASS top yaw candidates (user scene: truth in top-3 both runs)")


def test_pick_yaw_trial():
    """Arbitration preference: agreement (the fitted boxes' own
    directions match the render yaw) is unique to the TRUE direction
    -- at every wrong yaw the box PCA votes carry the ERROR angle --
    so an agreeing trial beats a higher-yield disagreeing one; among
    agreeing trials the most boxes wins (a straight view detects
    more: 5 vs 2 in the user's logs); with no agreement anywhere the
    yield alone decides."""
    from agentic_gts.segment.orientation import pick_yaw_trial

    R = math.radians
    # the wrong yaw found 2 boxes that voted 8.5 deg off; the true
    # yaw found 5 boxes that agree -- truth must win
    win = pick_yaw_trial([
        {"yaw": R(-26.5), "n": 3, "delta": R(9.0)},
        {"yaw": R(-9.5), "n": 2, "delta": R(-8.5)},
        {"yaw": R(-17.5), "n": 5, "delta": R(0.4)},
    ])
    assert abs(math.degrees(win["yaw"]) + 17.5) < 0.5 and win["n"] == 5
    # yield breaks ties among agreeing trials
    win = pick_yaw_trial([
        {"yaw": R(-17.5), "n": 5, "delta": R(0.4)},
        {"yaw": R(-16.9), "n": 5, "delta": R(0.1)},
    ])
    assert abs(math.degrees(win["yaw"]) + 16.9) < 0.5, \
        "equal yield, tighter agreement wins"
    # no agreement at all -> most boxes, delta ignored
    win = pick_yaw_trial([
        {"yaw": R(-9.5), "n": 4, "delta": None},
        {"yaw": R(-26.5), "n": 2, "delta": R(12.0)},
    ])
    assert abs(math.degrees(win["yaw"]) + 9.5) < 0.5
    # empty -> None
    assert pick_yaw_trial([]) is None
    print("PASS pick yaw trial (agreement > yield, yield > delta)")


def test_yaw_arbitration_needed():
    """Trigger condition: near-tied top-2 folded candidates must fire
    the arbitration EVEN WHEN the residual self-check passed -- the
    check measures CONSISTENCY, not correctness (user run 3: a wrong
    yaw of -26.7 deg over a GENUINE -26.5 structure re-aligned that
    structure, residual 0.7 'passed', and the run produced 2 boxes
    instead of the true yaw's 5). Run 3's folded top-2: -26.5 @468 vs
    -17.5 @410 -> ratio 0.876 -> arbitrate. All three user runs'
    candidate sets must trigger; a clear-cut scene must not."""
    from agentic_gts.segment.orientation import yaw_arbitration_needed

    # run 3 (verbatim from the log, residual check PASSED)
    run3 = [(72.5, 197), (-17.5, 410), (80.5, 405), (-9.5, 142),
            (63.5, 468), (-26.5, 304), (45.5, 336)]
    assert yaw_arbitration_needed({"candidates": run3}, yaw_suspect=False)
    # runs 1 and 2 (also near-tied, from the earlier logs)
    run1 = [(72.5, 201), (-17.5, 468), (80.5, 504), (-9.5, 143),
            (63.5, 472), (-26.5, 305), (45.5, 336)]
    run2 = [(72.5, 242), (-17.5, 430), (80.5, 412), (-9.5, 168),
            (63.5, 399), (-26.5, 440), (47.5, 261)]
    assert yaw_arbitration_needed({"candidates": run1}, yaw_suspect=False)
    assert yaw_arbitration_needed({"candidates": run2}, yaw_suspect=True)
    # clear-cut scene: one dominant direction -> no arbitration
    clear = [(0.5, 514), (-89.5, 380), (18.5, 223), (-71.5, 210)]
    assert not yaw_arbitration_needed({"candidates": clear},
                                      yaw_suspect=False)
    # suspect alone always fires; no candidates never does
    assert yaw_arbitration_needed({"candidates": []}, yaw_suspect=True)
    assert not yaw_arbitration_needed({"candidates": []},
                                      yaw_suspect=False)
    print("PASS yaw arbitration trigger (near-tied fires without suspect)")


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
    """StageG feedback signal: each grounded box's direction is
    MEASURED by PCA on the device-band points inside it (the boxes
    are axis-aligned in the row frame -- their own yaw carries no
    information), votes folded mod-90, weighted median. The old pool
    feedback re-ran the GLOBAL estimator on the union of the boxes'
    points -- a pool carved along the ASSUMED yaw, so slanted rows
    re-confirmed the assumed yaw (user report: wrong yaw surviving
    the feedback)."""
    from agentic_gts.core.models import OrientedBox
    from agentic_gts.segment.orientation import seed_axis_delta

    def _row(yaw_deg, cy, n=3000, seed=5):
        rng = np.random.default_rng(seed)
        a = math.radians(yaw_deg)
        u = np.array([math.cos(a), math.sin(a)])
        v = np.array([-math.sin(a), math.cos(a)])
        t = rng.uniform(0.0, 8.0, (n, 1))
        w = rng.uniform(-0.5, 0.5, (n, 1))
        xy = np.array([0.0, cy]) + t * u + w * v
        return np.column_stack([xy, rng.uniform(0.1, 2.0, (n, 1))])

    def _aabb_box(pts):
        lo, hi = pts[:, :2].min(axis=0) - 0.1, pts[:, :2].max(axis=0) + 0.1
        return OrientedBox(center=(float((lo[0] + hi[0]) / 2),
                                   float((lo[1] + hi[1]) / 2), 1.1),
                           size=(float(hi[0] - lo[0]), float(hi[1] - lo[1]),
                                 2.2), yaw=0.0)

    # two rows slanted 8 deg: per-box PCA recovers the slant
    r1, r2 = _row(8.0, 0.0, seed=1), _row(8.2, 4.0, seed=2)
    P = np.vstack([r1, r2])
    boxes = [_aabb_box(r1), _aabb_box(r2)]
    d = seed_axis_delta(boxes, P, cur_yaw=0.0)
    assert d is not None and abs(math.degrees(d) - 8.0) < 1.0, \
        f"slanted rows must vote their own axis, got {d}"

    # perpendicular rows agree (mod-90 fold): 98 deg == 8 deg
    r1, r2 = _row(8.0, 0.0, seed=1), _row(98.0, 4.0, seed=2)
    P = np.vstack([r1, r2])
    boxes = [_aabb_box(r1), _aabb_box(r2)]
    d = seed_axis_delta(boxes, P, cur_yaw=0.0)
    assert d is not None and abs(math.degrees(d) - 8.0) < 1.0, \
        f"perpendicular rows must fold to one direction, got {d}"

    # a heavier stray wall-ish blob cannot drag the weighted median
    r1, r2, wall = _row(8.0, 0.0, seed=1), _row(8.0, 4.0, seed=2), \
        _row(-20.0, 8.0, n=4000, seed=3)
    P = np.vstack([r1, r2, wall])
    boxes = [_aabb_box(r1), _aabb_box(r2), _aabb_box(wall)]
    d = seed_axis_delta(boxes, P, cur_yaw=0.0)
    assert d is not None and abs(math.degrees(d) - 8.0) < 1.0, \
        f"weighted median must resist one stray fit, got {math.degrees(d):.1f}"

    # straight rows: delta ~ 0 (the feedback never fires)
    r1, r2 = _row(0.3, 0.0, seed=1), _row(-0.2, 4.0, seed=2)
    P = np.vstack([r1, r2])
    boxes = [_aabb_box(r1), _aabb_box(r2)]
    d = seed_axis_delta(boxes, P, cur_yaw=0.0)
    assert d is not None and abs(math.degrees(d)) < 1.0

    # dd21246's measurement context: the MIDDLE z-slice with a
    # z_top+0.10 pool cut -- tray remnants floating ABOVE the rows
    # (diagonal sprinkles, off-axis) must not drag the votes; with the
    # whole device band they would
    rng = np.random.default_rng(9)
    r1, r2 = _row(8.0, 0.0, seed=1), _row(8.1, 4.0, seed=2)
    trays = np.column_stack([rng.uniform(-1, 9, 2500),
                              rng.uniform(-1, 6, 2500),
                              np.full(2500, 2.3)])   # above the rows
    P = np.vstack([r1, r2, trays])
    boxes = [_aabb_box(r1), _aabb_box(r2)]
    d = seed_axis_delta(boxes, P, cur_yaw=0.0, top_cut=2.0 + 0.10)
    assert d is not None and abs(math.degrees(d) - 8.0) < 1.0, \
        f"top-cut pool + middle slice must ignore tray remnants, got {d}"

    # THIN wall boxes cannot vote (user report: with mesh the yaw is
    # right on some runs, wrong on others). A mesh wall slips the
    # sliver guard at ~0.25m, is LONG (passes min_len) and DENSE
    # (heaviest weight); when the VLM randomly calls it a rack row the
    # vote drags the median and CORRUPTS a correct yaw. No device
    # category is thinner than 0.35m; walls are 0.1-0.3m.
    def _thin_wall_box(yaw_deg, cy, length=10.0, thick=0.25):
        a = math.radians(yaw_deg)
        return OrientedBox(
            center=(0.0, cy, 1.1), size=(length, thick, 2.2), yaw=0.0)

    r1, r2 = _row(8.0, 0.0, seed=1), _row(8.0, 4.0, seed=2)
    P = np.vstack([r1, r2])
    wall_box = _thin_wall_box(0.0, 8.0)
    # the wall's own points, 0-deg, dense (mesh-exact)
    wt = rng2 = np.random.default_rng(5)
    t = wt.uniform(-5.0, 5.0, (4000, 1))
    w = wt.uniform(-0.12, 0.12, (4000, 1))
    Pw = np.vstack([P, np.column_stack([t, np.full((4000, 1), 8.0) + w,
                                        wt.uniform(0.1, 2.0, (4000, 1))])])
    d = seed_axis_delta([_aabb_box(r1), _aabb_box(r2), wall_box],
                        Pw, cur_yaw=0.0)
    assert d is not None and abs(math.degrees(d) - 8.0) < 1.0, \
        f"thin wall box must not vote, got {math.degrees(d):.1f} deg"
    print(f"PASS seed axis delta ({math.degrees(d):+.2f} deg on straight, "
          f"~8 deg recovered on slanted, trays ignored)")


def _wall_room(theta_deg: float = 17.5, seed: int = 0):
    """A mesh-like rectangular room: dense straight wall lines, device
    rows PARALLEL to the long wall, plus floor/ceiling layers the
    z-band slice must ignore. Returned in the rotated frame."""
    rng = np.random.default_rng(seed)
    a = math.radians(theta_deg)
    R = np.array([[math.cos(a), -math.sin(a)],
                  [math.sin(a), math.cos(a)]])
    W, H = 16.0, 10.0

    def _wall(p0, p1, n):
        u = np.array([p1[0] - p0[0], p1[1] - p0[1]])
        L = float(np.hypot(*u))
        u = u / L
        v = np.array([-u[1], u[0]])
        t = rng.uniform(0.0, 1.0, (n, 1))
        w = rng.normal(0.0, 0.05, (n, 1))
        return np.asarray(p0) + t * u + w * v

    xys = [_wall((0, 0), (W, 0), 4000), _wall((W, 0), (W, H), 2500),
           _wall((W, H), (0, H), 4000), _wall((0, H), (0, 0), 2500)]
    # device rows parallel to the long (x) walls
    for y in np.arange(2.0, 9.0, 2.0):
        t = rng.uniform(2.0, 14.0, (1500, 1))
        w = rng.uniform(-0.3, 0.3, (1500, 1))
        xys.append(np.column_stack([t, np.full_like(t, y) + w]))
    xy = np.vstack(xys) @ R.T
    z = rng.uniform(0.2, 3.2, len(xy))
    pts = np.column_stack([xy, z])
    # floor and ceiling layers: outside the device z-band, must be cut
    nf = 8000
    fxy = rng.uniform(0.0, W, (nf, 2)) * [1.0, H / W]
    fxy = fxy @ R.T
    floor = np.column_stack([fxy, rng.uniform(0.0, 0.1, nf)])
    ceil = np.column_stack([fxy, rng.uniform(2.8, 3.5, nf)])
    return np.vstack([pts, floor, ceil])


def test_fit_wall_yaw():
    """Mesh fast path: the outer wall lines give the layout direction
    directly -- devices sit parallel to the walls, so the dominant
    hull-edge family IS the yaw (an orthogonal room's walls all fold
    into one mod-90 family). Floor/ceiling layers must be ignored
    (z-band slice)."""
    from agentic_gts.segment.orientation import fit_wall_yaw

    pts = _wall_room(theta_deg=17.5)
    r = fit_wall_yaw(pts)
    assert r is not None, "rectangular room must yield a wall yaw"
    yaw, conf = r
    err = abs(17.5 - math.degrees(yaw))
    assert err < 1.0, f"wall yaw off by {err:.1f} deg"
    assert conf >= 0.9, f"orthogonal room: one family, got share {conf:.2f}"

    # negative rotation folds to [-45, 45)
    r = fit_wall_yaw(_wall_room(theta_deg=-26.0, seed=1))
    assert r is not None and abs(math.degrees(r[0]) + 26.0) < 1.0

    # slanted device rows inside a straight room: the WALL direction
    # wins by design (mesh premise: devices parallel to walls)
    pts = _wall_room(theta_deg=0.0, seed=2)
    rng = np.random.default_rng(3)
    sa = math.radians(35.0)
    u = np.array([math.cos(sa), math.sin(sa)])
    t = rng.uniform(2.0, 12.0, (3000, 1))
    w = rng.uniform(-0.3, 0.3, (3000, 1))
    slant = np.column_stack([4.0 + t * u[0] - w * u[1],
                             5.0 + t * u[1] + w * u[0],
                             rng.uniform(0.3, 2.4, (3000, 1))])
    r = fit_wall_yaw(np.vstack([pts, slant]))
    assert r is not None and abs(math.degrees(r[0])) < 1.0, \
        "wall lines must beat slanted interior rows"

    # curved outer wall: no dominant straight family -> None (caller
    # falls back to the device-vote estimator)
    rng = np.random.default_rng(4)
    th = rng.uniform(0, 2 * math.pi, 12000)
    rr = 8.0 + rng.normal(0, 0.05, 12000)
    disk = np.column_stack([rr * np.cos(th), rr * np.sin(th),
                            rng.uniform(0.2, 3.0, 12000)])
    assert fit_wall_yaw(disk) is None, "curved wall must not yield a yaw"
    print("PASS fit wall yaw (mesh fast path)")


def test_fit_wall_yaw_bin_boundary_split():
    """Regression (user scene dianchifang): the true walls at -17.5 deg
    sit EXACTLY between rigid 5-deg histogram bins, and per-edge
    wobble (devices against the wall pushing hull vertices out)
    split the family into -20 and -15 bins -- 5766 + 5269 pts, best
    rigid share 33%, fast path skipped, wrong vote-chain yaw. Tolerance
    clustering must merge the wobbling family and return the wall
    yaw."""
    from agentic_gts.segment.orientation import fit_wall_yaw

    rng = np.random.default_rng(11)
    theta = math.radians(-17.5)
    jitters = [math.radians(d) for d in (-1.8, 1.6, -1.2, 2.0)]
    W, H = 16.0, 10.0
    segs = [((0, 0), (W, 0)), ((W, 0), (W, H)),
            ((W, H), (0, H)), ((0, H), (0, 0))]
    pts = []
    for (p0, p1), jit in zip(segs, jitters):
        p0 = np.array(p0, dtype=float) - np.array([W / 2, H / 2])
        p1 = np.array(p1, dtype=float) - np.array([W / 2, H / 2])
        mid = (p0 + p1) / 2                       # tilt around midpoint
        c, s = math.cos(jit), math.sin(jit)
        Rj = np.array([[c, -s], [s, c]])
        p0, p1 = Rj @ (p0 - mid) + mid, Rj @ (p1 - mid) + mid
        u = (p1 - p0) / np.linalg.norm(p1 - p0)
        v = np.array([-u[1], u[0]])
        n = 4000
        t = rng.uniform(0.0, 1.0, (n, 1))
        w = rng.normal(0, 0.05, (n, 1))
        seg = p0 + t * u + w * v
        pts.append(np.column_stack([seg, rng.uniform(0.5, 2.4, (n, 1))]))
    # device rows parallel to the long walls (nominal direction)
    for y in (-3.0, 0.0, 3.0):
        t = rng.uniform(-6.0, 6.0, (1500, 1))
        w = rng.uniform(-0.3, 0.3, (1500, 1))
        pts.append(np.column_stack([t, np.full_like(t, y) + w,
                                     rng.uniform(0.5, 2.4, (1500, 1))]))
    P = np.vstack(pts)
    c, s = math.cos(theta), math.sin(theta)
    R = np.array([[c, -s], [s, c]])
    P[:, :2] = P[:, :2] @ R.T
    r = fit_wall_yaw(P)
    assert r is not None, "bin-boundary family must cluster by tolerance"
    yaw, conf = r
    assert abs(math.degrees(yaw) + 17.5) < 1.5, \
        f"expected ~-17.5 deg, got {math.degrees(yaw):.1f}"
    assert conf >= 0.9, f"one wobbly family, got share {conf:.2f}"
    print(f"PASS wall yaw bin-boundary split (yaw="
          f"{math.degrees(yaw):+.1f} deg, share {conf:.0%})")


def test_estimate_yaw_detailed_mesh_flag():
    """mesh=True runs the wall fast path FIRST and marks the source;
    mesh=False (3DGS inputs) keeps the pure device-vote path."""
    from agentic_gts.segment.orientation import estimate_yaw_detailed

    pts = _wall_room(theta_deg=17.5)
    info = estimate_yaw_detailed(pts, mesh=True)
    assert info.get("yaw_source") == "wall", \
        f"mesh input must be wall-sourced, got {info.get('yaw_source')}"
    assert abs(17.5 - math.degrees(info["yaw"])) < 1.5
    # byproducts (layout bootstrap) survive the fast path
    assert info.get("device_cells") is not None
    assert info.get("z_top") is not None

    info = estimate_yaw_detailed(pts, mesh=False)
    assert "yaw_source" not in info or info["yaw_source"] != "wall"
    print("PASS estimate yaw detailed mesh flag (wall source marked)")


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


def _mesh_room(rows_yaw_deg: float, part_delta_deg: float | None,
               rng, n_face=9000, n_top=9000, n_part=20000,
               beyond: bool = False):
    """A discretized-MESH-like machine room: perfect planes, no noise.

    Rows at `rows_yaw_deg` as HOLLOW SHELLS (two vertical face sheets
    + a top plane -- what a mesh sampling of closed racks actually
    gives); floor / ceiling slabs; four outer walls (rectangular room,
    on the hull boundary); optionally one INTERIOR partition wall
    (fire / glass partition) rotated `part_delta_deg` from the rows and
    kept fully inside the room -- interior walls are NOT on the convex
    hull, so the boundary strip cannot remove them, and a mesh renders
    them as perfect dense planes (with 3DGS they were haze and lost
    every vote)."""
    a = math.radians(rows_yaw_deg)
    u = np.array([math.cos(a), math.sin(a)])          # along the row
    v = np.array([-math.sin(a), math.cos(a)])         # across
    pts = []
    room_x = 16.0                                      # along-row extent
    for cy in (-4.0, -1.2, 2.4):                       # three rows 1.1m deep
        base = np.array([2.0, 0.0])
        for face in (-0.55, 0.55):                     # the two face sheets
            t = rng.uniform(0.0, room_x, (n_face, 1))
            xy = base + t * u + (cy + face) * v
            pts.append(np.column_stack([xy, rng.uniform(
                0.05, 2.05, (n_face, 1))]))
        # rack tops: dense PERFECT horizontal planes (mesh-exact)
        t = rng.uniform(0.0, room_x, (n_top, 1))
        w = rng.uniform(-0.55, 0.55, (n_top, 1))
        xy = base + t * u + (cy + w) * v
        pts.append(np.column_stack([xy, np.full((n_top, 1), 2.05)]))
    # floor / ceiling slabs
    fx = rng.uniform(-2.0, 20.0, (60000, 1))
    fy = rng.uniform(-7.0, 5.0, (60000, 1))
    pts.append(np.column_stack([fx, fy, np.zeros((60000, 1))]))
    pts.append(np.column_stack([fx, fy, np.full((60000, 1), 3.0)]))
    # four outer walls (axis-aligned room frame, on the hull boundary)
    for wx in (-2.0, 20.0):
        pts.append(np.column_stack([
            np.full((n_face, 1), wx),
            rng.uniform(-7.0, 5.0, (n_face, 1)),
            rng.uniform(0.05, 2.95, (n_face, 1))]))
    for wy in (-7.0, 5.0):
        pts.append(np.column_stack([
            rng.uniform(-2.0, 20.0, (n_face, 1)),
            np.full((n_face, 1), wy),
            rng.uniform(0.05, 2.95, (n_face, 1))]))
    if part_delta_deg is not None:
        # interior partition wall: perfect 0.2m-thick vertical plane,
        # kept fully INSIDE the room (off the hull), exactly the
        # structure the boundary strip cannot touch
        pa = math.radians(rows_yaw_deg + part_delta_deg)
        pu = np.array([math.cos(pa), math.sin(pa)])
        pv = np.array([-math.sin(pa), math.cos(pa)])
        s = rng.uniform(-4.0, 5.0, (n_part, 1))
        w = rng.uniform(-0.10, 0.10, (n_part, 1))
        c = np.array([8.0, -1.0]) + s * pu + w * pv
        pts.append(np.column_stack([c, rng.uniform(0.05, 2.95, (n_part, 1))]))
    if beyond:
        # the reconstruction extends PAST the machine room (captured
        # corridor / neighboring space): the hull moves outward and
        # the room's OUTER WALLS -- perfect dense planes in a mesh --
        # are interior now, exactly what the boundary strip CANNOT
        # remove. The walls vote at the ROOM frame angle (0 deg) while
        # the rows sit at rows_yaw_deg.
        ex = rng.uniform(-4.0, 22.0, (40000, 1))
        ey = rng.uniform(-9.0, 7.0, (40000, 1))
        pts.append(np.column_stack([ex, ey, np.zeros((40000, 1))]))
        pts.append(np.column_stack([ex, ey, np.full((40000, 1), 3.4)]))
        for wx in (-4.0, 22.0):
            pts.append(np.column_stack([
                np.full((n_face, 1), wx),
                rng.uniform(-9.0, 7.0, (n_face, 1)),
                rng.uniform(0.05, 3.35, (n_face, 1))]))
        for wy in (-9.0, 7.0):
            pts.append(np.column_stack([
                rng.uniform(-4.0, 22.0, (n_face, 1)),
                np.full((n_face, 1), wy),
                rng.uniform(0.05, 3.35, (n_face, 1))]))
    return np.vstack(pts)


def test_yaw_clean_mesh_interior_partition():
    """MESH geometry (user report: yaw far off with --mesh-cloud, which
    'should not happen -- mesh noise is small'): a mesh renders an
    interior partition wall as a PERFECT dense plane. The boundary
    strip only removes hull-adjacent cells, and _row_band_score
    accepted bands as thin as 0.1m -- one long wall concentrates its
    whole mass into a single sub-0.3m band and OUT-SCORES the real
    rows (whose mass splits across several 0.6-1.2m bands). The
    residual self-check re-runs the same estimator, the same wall
    wins again, residual ~0: the hijack passes as consistency. The
    fix: no device category is thinner than ~0.5m -- bands under
    0.45m score nothing."""
    rng = np.random.default_rng(9)
    pts = _mesh_room(rows_yaw_deg=8.0, part_delta_deg=47.0, rng=rng)
    yaw = estimate_yaw(pts)
    err = abs(8.0 - math.degrees(yaw)) % 90.0
    err = min(err, 90.0 - err)
    assert err < 3.0, \
        f"mesh room yaw {math.degrees(yaw):.1f} deg (rows at 8 deg) -- " \
        f"interior partition hijacked the estimate"
    # no-partition control: the same room without the wall must also pass
    pts2 = _mesh_room(rows_yaw_deg=8.0, part_delta_deg=None, rng=rng)
    yaw2 = estimate_yaw(pts2)
    err2 = abs(8.0 - math.degrees(yaw2)) % 90.0
    err2 = min(err2, 90.0 - err2)
    assert err2 < 3.0, f"control (no wall) yaw {math.degrees(yaw2):.1f} deg"
    print(f"PASS clean-mesh yaw (with 47-deg partition: "
          f"{math.degrees(yaw):.1f} deg, control {math.degrees(yaw2):.1f} deg)")


def test_yaw_mesh_reconstruction_beyond_room():
    """The other mesh-specific hijack (user report: yaw far off ONLY
    with --mesh-cloud): a reconstruction that extends PAST the machine
    room (captured corridor / neighboring space). The hull moves
    outward, the room's own walls -- PERFECT dense planes in a mesh,
    axis-aligned to the building frame -- become interior cells the
    boundary strip cannot remove, and they out-vote the rows whenever
    the room frame differs from the row direction. With 3DGS the same
    walls were haze and lost every vote."""
    rng = np.random.default_rng(11)
    pts = _mesh_room(rows_yaw_deg=8.0, part_delta_deg=None, rng=rng,
                     beyond=True)
    yaw = estimate_yaw(pts)
    err = abs(8.0 - math.degrees(yaw)) % 90.0
    err = min(err, 90.0 - err)
    assert err < 3.0, \
        f"mesh-beyond-room yaw {math.degrees(yaw):.1f} deg (rows at 8) -- " \
        f"the room-frame walls hijacked the estimate"
    print(f"PASS mesh beyond room (yaw {math.degrees(yaw):.1f} deg, rows at 8)")


if __name__ == "__main__":
    test_align_and_yaw_on_adversarial_cloud()
    print("PASS  test_align_and_yaw_on_adversarial_cloud")
    test_align_with_subfloor_noise()
    print("PASS  test_align_with_subfloor_noise")
