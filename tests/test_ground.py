"""Tests for the VLM 2D grounding stage (no server / GPU needed).

Covers:
  - unproject_ground: pixel -> world roundtrip through the god-view cam
  - _fit_region_box: a region rect becomes a FULL-DEPTH box (the
    thin-fragment killer: both face bands + hollow interior inside)
  - ground_stage end-to-end with a patched VLM answer (scatter path)
"""
from __future__ import annotations

import math
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from agentic_gts.core.models import Scene


def _row_points(x0, x1, y=0.0, depth=1.1, height=2.1, rng=None, n=6000):
    """A closed-cabinet row: two face bands + TOP face + hollow
    interior (surface points only, like a real 3DGS / mesh sampling of
    a closed rack row -- the top band is what real clouds carry and
    what lets interior footprint cells count as occupied)."""
    rng = rng or np.random.default_rng(0)
    half = depth / 2.0
    pts = []
    for face in (+half, -half):     # front / back bands
        pts.append(np.column_stack([rng.uniform(x0, x1, n),
                                    np.full(n, y + face),
                                    rng.uniform(0.0, height, n)]))
    pts.append(np.column_stack([rng.uniform(x0, x1, n // 2),
                                rng.uniform(y - half, y + half, n // 2),
                                np.full(n // 2, height - 0.05)]))
    return np.vstack(pts)


def _bootstrap_meta(scene, footprint, z_top=2.1):
    """Fill scene.meta the way stage0's estimate_yaw_detailed does."""
    scene.meta["z_top"] = z_top
    scene.meta["device_footprint"] = footprint


def test_render_topdown_tilt_camera_geometry():
    """Recall tilt views (user direction 1): the L/R cameras deviate
    from vertical by ~tilt_deg along the ROW-frame cross axis (eye on
    -y / +y, sight line pointing back across), up stays orthogonal to
    the sight line, and pixel->world roundtrip through each tilted
    camera still closes (unproject_ground handles non-nadir rays)."""
    from agentic_gts.agent import ground
    from agentic_gts.output.gs_render import unproject_ground

    rng = np.random.default_rng(11)
    pts = np.vstack([_row_points(0.0, 6.0, y=0.0, rng=rng),
                     _row_points(0.0, 6.0, y=3.0, rng=rng)])
    scene = Scene(points=pts)
    scene.meta["yaw"] = 0.0
    _bootstrap_meta(scene, (-0.5, -0.8, 6.5, 3.8))
    _, cam0, W, H = ground._render_topdown(scene, 0.0)
    _, camL, _, _ = ground._render_topdown(scene, 0.0, tilt_deg=20.0,
                                           tilt_dir=-1)
    _, camR, _, _ = ground._render_topdown(scene, 0.0, tilt_deg=20.0,
                                           tilt_dir=+1)
    # eyes sit on opposite sides of the layout, across the rows
    assert camL.eye[1] < cam0.eye[1] - 0.5, "L eye must sit toward -y"
    assert camR.eye[1] > cam0.eye[1] + 0.5, "R eye must sit toward +y"
    for cam, sgn in ((camL, -1), (camR, +1)):
        fwd = np.asarray(cam.target, float) - np.asarray(cam.eye, float)
        # the sight line tilts back across the rows by ~tilt_deg
        ang = math.degrees(math.atan2(abs(float(fwd[1])),
                                       abs(float(fwd[2]))))
        assert abs(ang - 20.0) < 2.0, f"tilt angle {ang:.1f} deg"
        assert fwd[1] * sgn < 0, "sight line must point back at the layout"
        # up stays a proper image up: orthogonal to the sight line
        u = np.asarray(cam.up, float)
        assert abs(float(u @ fwd)) < 1e-6 * float(np.linalg.norm(fwd))
    # roundtrip through the tilted cameras: project a world patch,
    # unproject onto the SAME z-plane -> the same XY
    world = np.column_stack([rng.uniform(-0.5, 6.5, 40),
                            rng.uniform(-0.8, 3.8, 40), np.full(40, 1.0)])
    for cam in (camL, camR):
        uv = cam.project_cv(world)
        assert ((uv[:, 0] >= 0) & (uv[:, 0] < W)
                & (uv[:, 1] >= 0) & (uv[:, 1] < H)).all(), \
            "the whole layout must stay in the tilted frame"
        back = unproject_ground(cam, uv, z_plane=1.0)
        assert np.allclose(back[:, :2], world[:, :2], atol=0.01), \
            f"tilt roundtrip error " \
            f"{np.abs(back[:, :2] - world[:, :2]).max():.4f} m"
    print("PASS tilt camera geometry (L/R eyes, 20 deg sight lines, "
          "roundtrip closed)")


def test_ground_stage_tilt_views_add_recall():
    """End-to-end recall (user direction 1): the nadir view outlines
    only row 1; the tilted R view ALSO sees row 2 and a lone AC block
    -- the pipeline must ground all three. Each view's rects are
    fabricated through ITS OWN camera (the fake judge dispatches on
    the saved png name), so the tilted rects travel the real
    pixel->world->snap->fit path."""
    from agentic_gts.agent import ground
    from agentic_gts.agent.judge import VLMJudge

    rng = np.random.default_rng(17)
    row1 = _row_points(0.0, 6.0, y=0.0, rng=rng)
    row2 = _row_points(0.0, 6.0, y=3.0, rng=rng)
    ac = _row_points(8.0, 9.0, y=0.0, depth=1.0, height=1.0, rng=rng)
    scene = Scene(points=np.vstack([row1, row2, ac]))
    scene.meta["yaw"] = 0.0
    _bootstrap_meta(scene, (-0.5, -0.8, 9.5, 3.8))
    scene.boxes = []

    _, cam0, W, H = ground._render_topdown(scene, 0.0)
    _, camL, _, _ = ground._render_topdown(scene, 0.0, tilt_deg=20.0,
                                           tilt_dir=-1)
    _, camR, _, _ = ground._render_topdown(scene, 0.0, tilt_deg=20.0,
                                           tilt_dir=+1)
    view_rects = {
        "groundview.png": (cam0, [((-0.2, 6.2), (-0.6, 0.6))]),
        # the L view repeats the nadir rect (a duplicate): the recall
        # gate must SKIP it -- a tilted rect may never re-fit an area
        # the nadir views already own (its perspective-inflated fit
        # would out-support and EAT the correct nadir box in dedup)
        "groundview_L.png": (camL, [((-0.2, 6.2), (-0.6, 0.6))]),
        "groundview_R.png": (camR, [((-0.2, 6.2), (-0.6, 0.6)),
                                     ((-0.2, 6.2), (2.4, 3.6)),
                                     ((7.8, 9.2), (-0.6, 0.6))]),
    }

    def _fake_ground(png, W_, H_, png_path=None):
        name = os.path.basename(png_path or "groundview.png")
        cam, specs = view_rects[name]
        rects = []
        for (xa, xb), (ya, yb) in specs:
            uv = cam.project_cv(np.column_stack(
                [[xa, xb, xb, xa], [ya, ya, yb, yb], np.full(4, 1.0)]))
            rects.append((float(np.clip(uv[:, 0].min(), 0, W_)),
                          float(np.clip(uv[:, 1].min(), 0, H_)),
                          float(np.clip(uv[:, 0].max(), 0, W_)),
                          float(np.clip(uv[:, 1].max(), 0, H_))))
        return rects

    judge = VLMJudge(backend="qwen")
    judge.ground_regions = _fake_ground
    import tempfile
    with tempfile.TemporaryDirectory() as td:
        ok = ground.ground_stage(scene, judge, out_dir=td)
        assert ok, "tilt-view grounding must succeed"
        for name in view_rects:
            assert os.path.exists(os.path.join(td, name)), \
                f"{name} (recall view input) was not saved"
    assert len(scene.boxes) == 3, \
        f"row1 + tilt-seen row2 + tilt-seen AC, got {len(scene.boxes)}: " \
        + str([(round(b.center[0], 1), round(b.center[1], 1),
                (round(b.size[0], 1), round(b.size[1], 1)))
               for b in scene.boxes])
    acb = [b for b in scene.boxes if b.center[0] > 6.5]
    assert len(acb) == 1, "the AC must come from the tilted R view"
    assert abs(acb[0].center[1]) < 0.3 and 0.8 < acb[0].size[1] < 1.4
    rows = sorted((b for b in scene.boxes if b.center[0] < 6.5),
                  key=lambda b: b.center[1])
    assert len(rows) == 2 and abs(rows[1].center[1] - 3.0) < 0.3, \
        "row2 (tilt-seen) must also be grounded"
    print("PASS tilt views add recall (nadir missed 2, tilted R "
          "recovered both)")


def test_render_topdown_mesh_framing_room_centre():
    """MESH input (user directive): the sampling is clean, the cloud
    IS the room -- the nadir camera must sit over the WHOLE-cloud
    centre (the room's centre), not the bootstrap layout's. The
    3DGS branch keeps the layout framing (outer haze must stay out
    of frame). Devices shoved into one corner of a 12x8 room make
    the two centres 3m+ apart, so the branch taken is unambiguous."""
    from agentic_gts.agent import ground

    rng = np.random.default_rng(23)
    # room walls: four thin bands around a 12x8 footprint
    wall = []
    for _x0, _x1, _y0, _y1 in ((-6, 6, -4, -3.8), (-6, 6, 3.8, 4),
                               (-6, -5.8, -3.8, 3.8), (5.8, 6, -3.8, 3.8)):
        wall.append(np.column_stack([
            rng.uniform(_x0, _x1, 3000), rng.uniform(_y0, _y1, 3000),
            rng.uniform(0.0, 3.0, 3000)]))
    # devices crammed into the room's +x/+y corner
    dev = np.vstack([_row_points(3.0, 5.6, y=1.0, rng=rng),
                     _row_points(3.0, 5.6, y=2.8, rng=rng)])
    pts = np.vstack(wall + [dev])
    room_c = (float(pts[:, 0].min() + pts[:, 0].max()) / 2.0,
              float(pts[:, 1].min() + pts[:, 1].max()) / 2.0)

    scene = Scene(points=pts)
    scene.meta["yaw"] = 0.0
    _bootstrap_meta(scene, (2.8, 0.6, 5.8, 3.2))
    scene.meta["geometry_is_mesh"] = True
    _, cam_m, _, _ = ground._render_topdown(scene, 0.0)
    assert (abs(float(cam_m.eye[0]) - room_c[0]) < 0.3
            and abs(float(cam_m.eye[1]) - room_c[1]) < 0.3), \
        (f"mesh camera must centre on the ROOM ({room_c}), got eye "
         f"({cam_m.eye[0]:.2f}, {cam_m.eye[1]:.2f})")

    # the 3DGS branch (no mesh flag) keeps the layout framing: eye
    # near the layout centre, NOT the room centre
    scene.meta["geometry_is_mesh"] = False
    _, cam_g, _, _ = ground._render_topdown(scene, 0.0)
    assert abs(float(cam_g.eye[0]) - 4.3) < 0.4 \
        and abs(float(cam_g.eye[1]) - 1.9) < 0.4, \
        ("3DGS camera must keep the layout framing, got eye "
         f"({cam_g.eye[0]:.2f}, {cam_g.eye[1]:.2f})")
    print(f"PASS mesh framing: room centre ({room_c[0]:.1f},"
          f"{room_c[1]:.1f}) vs 3DGS layout "
          f"({cam_g.eye[0]:.1f},{cam_g.eye[1]:.1f})")


def test_unproject_ground_roundtrip():
    from agentic_gts.output.gs_render import (make_godview_cam,
                                              unproject_ground)
    rng = np.random.default_rng(5)
    pts = rng.uniform(-8, 8, (200, 3))
    pts[:, 2] = rng.uniform(0, 2.2, 200)
    cam = make_godview_cam(pts, nadir=True)
    world = np.column_stack([rng.uniform(-5, 5, 50),
                             rng.uniform(-4, 4, 50),
                             np.full(50, 1.0)])
    uv = cam.project_cv(world)
    back = unproject_ground(cam, uv, z_plane=1.0)
    assert np.allclose(back[:, :2], world[:, :2], atol=0.01), \
        f"roundtrip error {np.abs(back[:, :2] - world[:, :2]).max():.4f} m"
    print("PASS unproject_ground roundtrip (nadir cam, 1cm)")


def test_fit_region_box_full_depth():
    from agentic_gts.agent.ground import _fit_region_box
    rng = np.random.default_rng(7)
    row = _row_points(0.0, 6.0, rng=rng)
    # a loose region rect (what a coarse VLM outline looks like)
    bb = _fit_region_box(row, (-0.2, -0.8, 6.2, 0.8))
    assert bb is not None, "region with two face bands must produce a box"
    assert 5.5 < bb.size[0] < 6.4, f"length {bb.size[0]:.2f} (want ~6.0)"
    # THE assertion: the box spans BOTH faces -- full depth, not a thin
    # fragment hugging one face
    assert 0.85 < bb.size[1] < 1.35, f"depth {bb.size[1]:.2f} (want ~1.1)"
    assert 1.9 < bb.size[2] < 2.3, f"height {bb.size[2]:.2f} (want ~2.1)"
    assert abs(bb.center[2] - bb.size[2] / 2.0) < 0.05, "bottom must sit on the floor"
    # hallucination guard: empty region -> None
    empty = np.zeros((10, 3))
    assert _fit_region_box(empty, (10, 10, 11, 11)) is None
    # floor patch guard: only sub-0.3m points -> None
    floor = np.column_stack([rng.uniform(-1, 1, 500),
                              rng.uniform(-1, 1, 500),
                              rng.uniform(0.0, 0.2, 500)])
    assert _fit_region_box(floor, (-1.2, -1.2, 1.2, 1.2)) is None
    print(f"PASS region fit full depth "
          f"(L={bb.size[0]:.2f} D={bb.size[1]:.2f} H={bb.size[2]:.2f})")


def test_fit_region_box_haze_immune():
    """The middle-slice strong-bin fit does not flinch at 3DGS aisle
    haze inside a loose rect: the sheets are tall narrow bins, haze is
    a low plateau, and the peak-relative threshold keeps the body.
    The old whole-band percentile fit let a few-percent haze tail
    inflate every edge (user report: boxes not snug to devices)."""
    from agentic_gts.agent.ground import _fit_region_box
    rng = np.random.default_rng(13)
    row = _row_points(0.0, 6.0, rng=rng)
    # haze: uniform scatter through the whole loose rect, ~3% of the
    # device mass -- ABOVE the 0.5% a P0.5-P99.5 cut trims
    haze = np.column_stack([rng.uniform(-0.3, 6.3, 400),
                            rng.uniform(-1.0, 1.0, 400),
                            rng.uniform(0.4, 2.0, 400)])
    bb = _fit_region_box(np.vstack([row, haze]), (-0.3, -1.0, 6.3, 1.0))
    assert bb is not None, "haze must not kill the fit"
    assert 5.5 < bb.size[0] < 6.4, \
        f"length {bb.size[0]:.2f} (haze inflated the row ends?)"
    assert 0.85 < bb.size[1] < 1.35, \
        f"depth {bb.size[1]:.2f} (haze inflated the faces?)"
    print(f"PASS region fit haze-immune "
          f"(L={bb.size[0]:.2f} D={bb.size[1]:.2f})")


def test_fit_region_box_starved_back_face():
    """A rack row whose BACK face is much sparser than the front (the
    wall-facing side of a 3DGS reconstruction) must keep its FULL
    depth. The plain strong-bin estimator thresholded at 40% of the
    GLOBAL peak: a 10-20%-density back face fell below it, the depth
    span collapsed to the front face (~5cm) and the sliver guard
    rejected the whole region -- real devices lost their boxes
    (user report: colored rects with obvious depth, no red box).
    _region_axis_span peels peaks per-cluster (back face survives at
    >= 15% of the front) and falls back to the percentile extent when
    the peel still collapses (< 50% of P0.5-P99.5)."""
    from agentic_gts.agent.ground import _fit_region_box, _region_axis_span

    # direct estimator checks on the depth axis
    for n_back in (1200, 600):        # 20% and 10% of the front face
        v = np.concatenate([np.full(6000, 2.45), np.full(n_back, 3.55)])
        span = _region_axis_span(v)
        assert span is not None
        assert span[1] - span[0] > 0.9, \
            f"span {span} collapsed with back face at {n_back} pts"
    # two healthy faces + haze: the peel succeeds and haze is trimmed
    # (a single face + haze cannot be told apart from a starved second
    # face by shape alone -- there the percentile floor deliberately
    # errs wide: a fat box beats a missing box)
    rng = np.random.default_rng(21)
    v = np.concatenate([np.full(6000, 2.45), np.full(3000, 3.55),
                        rng.uniform(3.2, 4.2, 400)])
    span = _region_axis_span(v)
    assert span is not None and 3.4 < span[1] < 3.7, \
        f"haze extended the span to {span}"

    # end-to-end: a row with a starved back face keeps a full-depth box
    for n_back in (1200, 300):
        front = np.column_stack([rng.uniform(0.0, 6.0, 6000),
                                 np.full(6000, 0.55),
                                 rng.uniform(0.0, 2.1, 6000)])
        back = np.column_stack([rng.uniform(0.0, 6.0, n_back),
                                np.full(n_back, -0.55),
                                rng.uniform(0.0, 2.1, n_back)])
        bb = _fit_region_box(np.vstack([front, back]),
                             (-0.2, -0.8, 6.2, 0.8))
        assert bb is not None, \
            f"starved back face ({n_back} pts) must not kill the box"
        assert 0.85 < bb.size[1] < 1.35, \
            f"depth {bb.size[1]:.2f} (want ~1.1, back face dropped?)"
    print("PASS region fit starved-back-face (full depth kept)")


def test_robust_span_bin_boundary():
    """_robust_span must not drop the topmost values: a mass sitting
    EXACTLY on a bin boundary (3.55) once fell beyond arange's last
    fp-drifted edge and np.histogram silently discarded it -- the span
    collapsed to the far face and the fitted box went thin. Fixed by
    the explicit bin-count edge construction."""
    from agentic_gts.agent.mask_refine import _robust_span
    v = np.concatenate([np.full(100, 2.45), np.full(100, 3.55)])
    span = _robust_span(v)
    assert span is not None
    assert span[1] - span[0] > 1.0, \
        f"span {span} collapsed (top-of-range mass dropped)"
    print(f"PASS robust span bin boundary ({span[0]:.2f}..{span[1]:.2f})")


def test_ground_stage_with_patched_vlm():
    """End-to-end grounding on the scatter path: the VLM answer is
    fabricated by projecting the TRUE row rects through the same cam the
    renderer builds (deterministic), so the full pixel->world->fit path
    is exercised without a server."""
    from agentic_gts.agent import ground
    from agentic_gts.agent.judge import VLMJudge

    rng = np.random.default_rng(3)
    # two parallel rows, the classic thin-fragment setup
    pts = np.vstack([_row_points(0.0, 6.0, y=0.0, rng=rng),
                     _row_points(-1.0, 5.0, y=3.0, rng=rng)])
    # CEILING slab spanning the whole room at z~2.95: the raw cloud
    # carries it (only the RENDER band cuts it); the FIT must cut it
    # too, or every fitted box spans the loose rect at tray height
    # (user report: red boxes all too large and wrong)
    ceil = np.column_stack([rng.uniform(-2.0, 7.0, 3000),
                            rng.uniform(-2.0, 5.0, 3000),
                            rng.uniform(2.9, 3.0, 3000)])
    pts = np.vstack([pts, ceil])
    scene = Scene(points=pts)
    scene.meta["yaw"] = 0.0
    _bootstrap_meta(scene, (-1.5, -0.8, 6.5, 3.8))
    scene.boxes = []
    # build the same deterministic render to fabricate the VLM answer
    _, cam, W, H = ground._render_topdown(scene, 0.0)
    true_rects = [((-0.5, 6.5), (-0.8, 0.8)),      # row 1 XY
                  ((-0.5, 6.5), (-0.8, 0.8)),      # row 1 AGAIN: the VLM
                  # often outlines one device twice (user report:
                  # duplicated mask_prompt_<id>_front.png renders)
                  ((-1.5, 5.5), (2.2, 3.8))]       # row 2 XY
    import json as _json
    regions = []
    for (xa, xb), (ya, yb) in true_rects:
        uv = cam.project_cv(np.column_stack([
            [xa, xb, xb, xa], [ya, ya, yb, yb], np.full(4, 1.0)]))
        # clamp to the image like a real reply: the official 0-1000
        # relative grid cannot express coordinates beyond the frame
        px = (np.clip(uv[:, 0].min(), 0, W), np.clip(uv[:, 1].min(), 0, H),
              np.clip(uv[:, 0].max(), 0, W), np.clip(uv[:, 1].max(), 0, H))
        regions.append({"bbox_2d": [
            int(round(px[0] / W * 1000)),
            int(round(px[1] / H * 1000)),
            int(round(px[2] / W * 1000)),
            int(round(px[3] / H * 1000))],
            "label": "row"})
    reply = ("Row 1: one long joined row at the bottom.\n"
             "Row 2: one long joined row above it.\n"
             + _json.dumps(regions))

    judge = VLMJudge(backend="qwen")

    def _fake_call(png, prompt, *a, **k):
        return reply
    judge._qwen_image_call = _fake_call    # canned VLM answer
    import tempfile
    with tempfile.TemporaryDirectory() as td:
        ok = ground.ground_stage(scene, judge, out_dir=td)
        assert ok, "grounding must succeed with a valid VLM reply"
        # result audit image: the view's own raw rects + the final
        # boxes projected through the same camera
        assert os.path.exists(os.path.join(td, "grounded.png")), \
            "grounded.png (result audit view) was not saved"
        assert os.path.exists(os.path.join(td, "groundview.png")), \
            "groundview.png (input view) was not saved"
        # same-base contract: the two images must be pixel-identical
        # apart from the red result overlays -- different ceiling cuts /
        # camera framing would confound the before/after comparison
        from PIL import Image
        a = np.asarray(Image.open(os.path.join(td, "groundview.png")))
        g = np.asarray(Image.open(os.path.join(td, "grounded.png")))
        assert a.shape == g.shape, "audit views must share one base render"
        diff = (np.abs(a.astype(int) - g.astype(int)).sum(axis=2) > 24)
        frac = float(diff.mean())
        assert frac < 0.05, \
            f"views differ over {frac:.1%} of pixels (base not shared?)"
    # 3 rects (row 1 outlined TWICE) -> still exactly 2 boxes: the
    # duplicate fit must be dropped by the IoU >= 0.5 dedup, or the
    # local refine stage renders one mask_prompt_<id>_front.png per
    # box -- duplicated images of one device
    assert len(scene.boxes) == 2, f"want 2 row boxes, got {len(scene.boxes)}"
    rows = sorted(scene.boxes, key=lambda b: b.center[1])
    # row 1: full length ~6m, FULL depth ~1.1m, height ~2.1m
    b = rows[0]
    assert 5.5 < b.size[0] < 6.5, f"row1 length {b.size[0]:.2f}"
    assert 0.85 < b.size[1] < 1.35, f"row1 depth {b.size[1]:.2f} (FULL, not thin)"
    assert 1.9 < b.size[2] < 2.35, \
        f"row1 height {b.size[2]:.2f} (ceiling must not be fitted!)"
    assert abs(b.center[1]) < 0.2, f"row1 y {b.center[1]:.2f}"
    # row 2
    b2 = rows[1]
    assert 5.5 < b2.size[0] < 6.5 and 0.85 < b2.size[1] < 1.35
    assert abs(b2.center[1] - 3.0) < 0.2
    print(f"PASS ground stage end-to-end "
          f"(row1 {rows[0].size[0]:.2f}x{rows[0].size[1]:.2f}, "
          f"row2 {b2.size[0]:.2f}x{b2.size[1]:.2f})")


def test_fit_region_box_row_along_y():
    """A row running along the ROTATED-Y axis: the fit must ride the
    long side on the yaw axis (yaw = pi/2, size = (length, depth)). A
    yaw=0 (dx, dy) fit puts the THICKNESS on the yaw axis, and the
    downstream local refine -- which projects along-row spans on the
    seed's yaw axis -- then splits the row ACROSS ITS DEPTH (user
    report: 'split into 3 instances' along the thickness)."""
    import math
    from agentic_gts.agent.ground import _fit_region_box
    rng = np.random.default_rng(11)
    row = _row_points(0.0, 6.0, rng=rng)
    row = row[:, [1, 0, 2]]       # transpose: the row now runs along y
    bb = _fit_region_box(row, (-0.8, -0.2, 0.8, 6.2))
    assert bb is not None, "a y-running row must produce a box"
    assert abs(bb.yaw - math.pi / 2.0) < 1e-9, \
        f"yaw must be pi/2 (long side on the yaw axis), got {bb.yaw}"
    assert 5.5 < bb.size[0] < 6.4, \
        f"size[0] must be the ROW LENGTH, got {bb.size[0]:.2f}"
    assert 0.85 < bb.size[1] < 1.35, \
        f"size[1] must be the depth, got {bb.size[1]:.2f}"
    # corners must cover the same extent as the equivalent yaw=0 fit
    cs = bb.corners_2d()
    assert -0.7 < cs[:, 0].min() < -0.4 and 0.4 < cs[:, 0].max() < 0.7, \
        "x extent must be the depth"
    assert -0.1 < cs[:, 1].min() < 0.1 and 5.9 < cs[:, 1].max() < 6.1, \
        "y extent must be the row length"
    print(f"PASS region fit rides the long side on yaw "
          f"(y-row: yaw=pi/2, L={bb.size[0]:.2f}, D={bb.size[1]:.2f})")


def test_floor_map_stepped():
    """Stepped room (small level change): _floor_map must recover each
    section's own floor so height-relative band cuts work again over
    the raised section (user report: ceiling remnants in part of the
    groundview; the raised slab entering the device band)."""
    from agentic_gts.agent.ground import _floor_map
    rng = np.random.default_rng(21)
    # lower section x in [-6, 0): floor slab 0-0.05 + racks 0..2.1
    # raised section x in [0, 6): slab 0.40-0.45 + racks 0.40..2.50
    slab_lo = rng.uniform([-6, -4, 0.0], [0, 4, 0.05], (8000, 3))
    slab_hi = rng.uniform([0, -4, 0.40], [6, 4, 0.45], (8000, 3))
    racks = np.vstack([
        rng.uniform([-4.5, -3.0, 0.0], [-3.5, 3.0, 2.1], (3000, 3)),
        rng.uniform([-1.5, -3.0, 0.0], [-0.5, 3.0, 2.1], (3000, 3)),
        rng.uniform([1.0, -3.0, 0.40], [2.0, 3.0, 2.50], (3000, 3)),
        rng.uniform([4.0, -3.0, 0.40], [5.0, 3.0, 2.50], (3000, 3)),
    ])
    pts = np.vstack([slab_lo, slab_hi, racks])
    fl = _floor_map(pts)
    f_lo, f_hi = float(fl(-3.0, 0.0)), float(fl(3.0, 0.0))
    assert -0.05 < f_lo < 0.10, f"lower floor {f_lo:.3f} (want ~0)"
    assert 0.35 < f_hi < 0.50, f"raised floor {f_hi:.3f} (want ~0.4)"
    # height semantics: slab points are the RAISED section's floor
    hh = slab_hi[:, 2] - fl(slab_hi[:, 0], slab_hi[:, 1])
    assert hh.max() < 0.30, "slab must fall below the device band"
    # raised racks keep their true 2.1m height above THEIR floor
    top = racks[racks[:, 0] > 0.5]
    hh_top = top[:, 2] - fl(top[:, 0], top[:, 1])
    assert 1.9 < hh_top.max() < 2.3, \
        f"raised rack height {hh_top.max():.2f} (want ~2.1)"
    print(f"PASS floor map stepped (lower {f_lo:.3f}, raised {f_hi:.3f})")


def test_fit_region_box_stepped_floor():
    """A rack standing on a raised slab (floor_z = 0.4): the fitted box
    must BOTTOM on the slab and carry the TRUE rack height -- without
    floor_z the box runs a step too deep (bottom 0) and a step too
    tall (slab-to-top)."""
    from agentic_gts.agent.ground import _fit_region_box
    rng = np.random.default_rng(22)
    body = rng.uniform([0.0, 0.0, 0.40], [3.0, 0.6, 2.50], (4000, 3))
    slab = rng.uniform([0.0, 0.0, 0.40], [3.0, 0.6, 0.45], (800, 3))
    pts = np.vstack([body, slab])
    rect = (-0.2, -0.2, 3.2, 0.8)
    bb = _fit_region_box(pts, rect, floor_z=0.40)
    assert bb is not None, "stepped-floor region must produce a box"
    bottom = bb.center[2] - bb.size[2] / 2.0
    assert 0.30 < bottom < 0.50, \
        f"bottom {bottom:.2f} (must sit ON the raised slab, ~0.40)"
    assert 1.9 < bb.size[2] < 2.3, \
        f"height {bb.size[2]:.2f} (must be the true rack height, not " \
        f"slab-to-top)"
    print(f"PASS region fit over raised floor "
          f"(bottom {bottom:.2f}, height {bb.size[2]:.2f})")


def test_fit_region_boxes_two_rows_in_one_rect():
    """One VLM rect drawn around TWO opposing rows (front + back, an
    aisle between): the union-depth fit must SPLIT across the thickness
    at the aisle -- stageC can never do it (its spans project on the
    ROW axis). Hollow-rack interiors must NOT attract the split (the
    min-side-depth rule rejects them: one face sheet on a side is a
    row interior, not an aisle)."""
    from agentic_gts.agent.ground import _fit_region_boxes
    rng = np.random.default_rng(23)
    row_a = _row_points(0.0, 6.0, y=-0.9, rng=rng)   # faces at -1.45/-0.35
    row_b = _row_points(0.0, 6.0, y=+0.9, rng=rng)   # faces at +0.35/+1.45
    pts = np.vstack([row_a, row_b])
    rect = (-0.2, -1.7, 6.2, 1.7)                    # ONE rect, both rows
    bbs = _fit_region_boxes(pts, rect)
    assert len(bbs) == 2, \
        f"two opposing rows in one rect must split, got {len(bbs)}"
    for bb in bbs:
        assert 0.85 < bb.size[1] < 1.35, \
            f"split piece depth {bb.size[1]:.2f} (must be ONE row, ~1.1)"
    ys = sorted(float(b.center[1]) for b in bbs)
    assert abs(ys[0] + 0.9) < 0.15 and abs(ys[1] - 0.9) < 0.15, \
        f"piece centres {ys} (must sit on the two rows, ~ +/-0.9)"
    print(f"PASS deep rect split (2 rows, centres y={ys[0]:.2f}/{ys[1]:.2f}, "
          f"depth {bbs[0].size[1]:.2f}/{bbs[1].size[1]:.2f})")


def test_fit_region_boxes_solid_deep_structure_kept():
    """A genuinely deep SOLID block with no aisle gap: no split is
    possible (no weak run), and the fit must stay whole rather than
    be shredded at a spurious location."""
    from agentic_gts.agent.ground import _fit_region_boxes
    rng = np.random.default_rng(24)
    solid = rng.uniform([0.0, 0.0, 0.0], [4.0, 2.2, 2.1], (12000, 3))
    bbs = _fit_region_boxes(solid, (-0.2, -0.2, 4.2, 2.4))
    assert len(bbs) == 1, \
        f"a solid deep block has no gap to split at, got {len(bbs)}"
    assert 2.0 < bbs[0].size[1] < 2.45, "depth must stay the whole extent"
    print(f"PASS solid deep block kept whole (depth {bbs[0].size[1]:.2f})")


def test_render_cut_mesh_mode():
    """mesh_mode trims the nadir render at HALF height: mesh cable
    trays are real gapless geometry that drags z_top up with them, so
    the 0.70 trim no longer clears them (user report)."""
    from agentic_gts.agent.ground import _render_cut
    gs = _render_cut(2.5)
    mesh = _render_cut(2.5, mesh_mode=True)
    assert gs == min(2.5 - 0.10, max(0.70 * 2.5, 1.0)) == 1.75
    assert mesh == min(2.5 - 0.10, max(0.50 * 2.5, 1.0)) == 1.25
    # a tray-inflated top (3.2m) still cuts BELOW the tray band (~2.2m)
    assert _render_cut(3.2, mesh_mode=True) <= 1.6, \
        "half-height cut must clear tray geometry dragged into z_top"
    # low structures keep the 1.0m floor in both modes
    assert _render_cut(0.9, mesh_mode=True) == 0.8
    print(f"PASS render cut mesh mode (gs {gs}, mesh {mesh})")


def test_cluster_pool_tray_stitch():
    """MESH recall-net blind spot (user report: with --mesh-cloud the
    grounding still misses many devices DESPITE the cluster recall
    net). A mesh renders cable trays as dense gapless planes
    connected to the rack tops; the anchored z_top walk runs right up
    them, so z_top IS the tray top and the FIT pool (0.30 .. z_top +
    0.10) carries tray planes spanning every aisle. Occupancy over
    that pool + the 0.3m dilation stitches the whole room into ONE
    cluster -- every VLM rect lands inside it (reverse containment),
    the net calls everything covered and goes SILENT. Cluster
    occupancy must use the RENDER-cut pool (trays live above it, every
    device category below it)."""
    from agentic_gts.agent.ground import _cluster_candidates, _render_cut
    rng = np.random.default_rng(17)
    rows = np.vstack([_row_points(0.0, 6.0, y=0.0, rng=rng),
                      _row_points(0.0, 6.0, y=3.0, rng=rng)])
    z_top = 2.6                            # dragged to the tray top
    trays = np.column_stack([rng.uniform(-0.5, 6.5, 8000),
                             rng.uniform(-1.5, 4.5, 8000),
                             np.full(8000, 2.55)])
    pts = np.vstack([rows, trays])
    # the bug, documented: fit-pool occupancy merges the whole room
    fit_pool = pts[(pts[:, 2] > 0.30) & (pts[:, 2] <= z_top + 0.10)]
    merged = _cluster_candidates(fit_pool)
    assert len(merged) == 1, \
        "trays spanning the aisles must (bug) stitch the rows into one"
    # the fix: render-cut occupancy separates the two rows, trays gone
    cut = _render_cut(z_top, mesh_mode=True)
    clu = pts[(pts[:, 2] > 0.30) & (pts[:, 2] <= cut)]
    cands = _cluster_candidates(clu)
    assert len(cands) == 2, \
        f"render-cut pool must give 2 row clusters (got {len(cands)})"
    print(f"PASS cluster tray stitch (fit-pool {len(merged)} cluster, "
          f"render-cut pool {len(cands)} clusters)")


def test_render_keep_mask_opacity_dual_band():
    """Opacity-aware dual-band floor cut: a SOLID gaussian (opacity
    >= 0.5) is real geometry and keeps the band from 0.30m -- a 0.7m
    AC bank shows its full body -- while LOW-opacity ones (3DGS haze)
    stay under the 1.00m trim. A blanket 1.00m trim cut sub-1m
    devices entirely; a blanket 0.30 washed the view in floor haze
    (user reports: both, in sequence)."""
    from agentic_gts.agent.ground import _render_keep_mask
    cut = 1.75
    # (h, opacity, expected_keep)
    cases = [
        (0.70, 0.90, True),    # SOLID low device (AC bank): KEPT
        (0.50, 0.55, True),    # solid, just over the 0.30 band
        (0.20, 0.95, False),   # solid floor slab: below the band
        (0.80, 0.10, False),   # haze floater under 1m: cut (the leak)
        (1.30, 0.15, True),    # haze above 1m still kept (rare, dim)
        (1.90, 0.95, False),   # solid tray ABOVE the ceiling cut
        (2.00, 0.20, False),   # low-opacity above the ceiling cut too
    ]
    hg = np.array([c[0] for c in cases])
    op = np.array([c[1] for c in cases])
    keep = _render_keep_mask(hg, op, cut)
    for (h, o, want), got in zip(cases, keep):
        assert bool(got) is want, \
            f"h={h:.2f} op={o:.2f}: keep={bool(got)}, want {want}"
    print("PASS render keep mask (opacity dual band)")


def test_cluster_candidates_basic():
    """The cluster recall net: after the ground / top cuts the fit
    pool holds nothing but walls, devices and junk (user insight) --
    every structure is a density clump. Two rows + a wall = three
    connected components; the coverage filter separates VLM-covered
    rows from an UNCOVERED wall clump."""
    from agentic_gts.agent.ground import (_cluster_candidates,
                                          _rect_covered)
    rng = np.random.default_rng(31)
    row1 = _row_points(0.0, 6.0, y=0.0, rng=rng)
    row2 = _row_points(0.0, 5.0, y=4.0, rng=rng)
    wall = np.column_stack([rng.uniform(-8.0, 8.0, 3000),
                            np.full(3000, 8.5),
                            rng.uniform(0.1, 2.0, 3000)])
    cands = _cluster_candidates(np.vstack([row1, row2, wall]))
    assert len(cands) == 3, \
        f"want 3 clusters (2 rows + wall), got {len(cands)}"
    rects = [r for r, _n in cands]
    # a VLM rect over row 1 covers row 1's cluster (containment), and
    # the wall cluster is NOT covered by either row rect
    near_row1 = min(rects, key=lambda r: abs((r[1] + r[3]) / 2.0))
    assert _rect_covered(near_row1, (-0.2, -0.8, 6.2, 0.8)), \
        "row 1 cluster must read as covered by its VLM rect"
    wall_rect = max(rects, key=lambda r: r[3])
    assert not _rect_covered(wall_rect, (-0.2, -0.8, 6.2, 0.8)), \
        "wall cluster must NOT be covered by the row-1 rect"
    assert not _rect_covered(wall_rect, (-0.2, 3.2, 5.2, 4.8)), \
        "wall cluster must NOT be covered by the row-2 rect"
    # sparse haze: a uniform scatter of a few points per cell stays
    # below the occupancy threshold -> no phantom clusters
    haze = np.column_stack([rng.uniform(-5.0, 5.0, 3000),
                            rng.uniform(-2.0, 7.0, 3000),
                            rng.uniform(0.4, 2.0, 3000)])
    cands_h = _cluster_candidates(np.vstack([row1, row2, wall, haze]),
                                  min_cell_pts=25)
    assert len(cands_h) == 3, "haze cells must not become clusters"
    print("PASS cluster candidates (3 blobs, coverage filter, "
          "haze-immune)")


def test_wall_adjacent_devices():
    """Devices standing AGAINST a wall (user question): the wall
    merges into the device's density cluster, so the net must not
    (a) re-propose a VLM-covered wall-adjacent device -- the reverse
    containment reads the VLM rect ~wholly inside the blob as covered
    (a wall-inflated refit would OUT-SUPPORT the correct box on wall
    points and eat it in the dedup), and (b) let the wall inflate a
    MISSED device's fitted box -- the cluster fit splits at the
    wall/device gap (relaxed min_side) and the wall strip dies in the
    sliver guard."""
    from agentic_gts.agent.ground import (_cross_gap_split,
                                          _fit_region_boxes,
                                          _rect_covered)

    # (a) reverse containment: VLM rect = the device only, cluster =
    # wall + gap + device (much bigger area -> forward containment
    # fails, reverse must hold)
    cluster = (0.0, 0.0, 6.0, 3.0)
    vlm_rect = (0.0, 0.0, 6.0, 1.1)
    assert _rect_covered(cluster, vlm_rect), \
        "wall+device blob must read as covered by the device-only rect"

    # (b) missed device against a wall: row y in [-0.55, 0.55], wall
    # sheet at y = 1.05 (0.2m thick) -> gap 0.5m, blob depth 1.8m.
    # The strict min_side (0.40) rejects the wall/device gap (the wall
    # side is only 0.2m strong); the relaxed cluster fit must split
    # there and the wall strip must die in the sliver guard.
    rng = np.random.default_rng(51)
    row = _row_points(0.0, 6.0, y=0.0, rng=rng)
    # wall sheet at face-comparable density (real reconstructions
    # render strong vertical surfaces dense): under the 25%-of-peak
    # strong threshold it reads as a strong side of the gap
    wall = np.column_stack([rng.uniform(-0.5, 6.5, 16000),
                            rng.uniform(1.05, 1.25, 16000),
                            rng.uniform(0.1, 2.4, 16000)])
    blob = np.vstack([row, wall])
    boxes = _fit_region_boxes(blob, (-0.4, -0.8, 6.4, 1.45),
                              max_depth=1.35, min_side=0.15)
    assert len(boxes) == 1, \
        f"wall strip must die in the sliver guard, got {len(boxes)}"
    b = boxes[0]
    assert 0.85 < b.size[1] < 1.35, \
        f"depth {b.size[1]:.2f} inflated by the wall (want ~1.1)"
    assert abs(b.center[1]) < 0.15, \
        f"center y {b.center[1]:.2f} dragged toward the wall"
    # the strict default still refuses: a hollow row's interior gap is
    # NOT a split candidate (regression guard for min_side plumbing)
    interior = _cross_gap_split(
        np.concatenate([np.full(3000, 0.0), np.full(3000, 1.1)]))
    assert interior is None, \
        "hollow-row faces must stay glued under the default min_side"
    print(f"PASS wall-adjacent devices (covered reverse-containment, "
          f"fit depth {b.size[1]:.2f}m wall-free)")


def test_ground_stage_cluster_recall():
    """The VLM MISSED a whole row on the nadir view (user report: a
    clean view cannot always be split into the wanted categories and
    devices get missed): the cluster recall net must recover it --
    connected density components PROPOSE the uncovered clump, and
    (user directive: recall-first) EVERY proposal becomes a box with
    NO VLM classification gate -- wrong proposals are cheap, missed
    devices are lost; stageC's local views type-confirm and cull.
    A wall-thin cluster is the one geometric refusal: a wall blob
    out-supports real boxes and would eat them in the dedup."""
    from agentic_gts.agent import ground
    from agentic_gts.agent.judge import VLMJudge

    rng = np.random.default_rng(3)
    pts = np.vstack([_row_points(0.0, 6.0, y=0.0, rng=rng),
                     _row_points(-1.0, 5.0, y=3.0, rng=rng)])
    # a free-standing wall sheet (0.2m thick) far from both rows: the
    # net finds its cluster but the wall-thin guard must refuse it
    wall = np.column_stack([rng.uniform(0.0, 8.0, 16000),
                            rng.uniform(5.8, 6.0, 16000),
                            rng.uniform(0.1, 2.4, 16000)])
    ceil = np.column_stack([rng.uniform(-2.0, 7.0, 3000),
                            rng.uniform(-2.0, 7.0, 3000),
                            rng.uniform(2.9, 3.0, 3000)])
    scene = Scene(points=np.vstack([pts, wall, ceil]))
    scene.meta["yaw"] = 0.0
    _bootstrap_meta(scene, (-1.5, -0.8, 8.5, 6.5))
    scene.boxes = []
    _, cam, W, H = ground._render_topdown(scene, 0.0)
    # the VLM grounds ONLY row 1 -- row 2 is the missed device
    uv = cam.project_cv(np.column_stack(
        [[-0.5, 6.5, 6.5, -0.5], [-0.8, -0.8, 0.8, 0.8],
         np.full(4, 1.0)]))
    px = (max(float(uv[:, 0].min()), 0.0), max(float(uv[:, 1].min()), 0.0),
          min(float(uv[:, 0].max()), W), min(float(uv[:, 1].max()), H))
    import json as _json
    reply = ("One row visible.\n" + _json.dumps(
        [{"bbox_2d": [int(round(px[0] / W * 1000)),
                      int(round(px[1] / H * 1000)),
                      int(round(px[2] / W * 1000)),
                      int(round(px[3] / H * 1000))],
          "label": "row"}]))

    judge = VLMJudge(backend="qwen")

    def _fake_call(png, prompt, *a, **k):
        return reply                         # grounding: row 1 only
    judge._qwen_image_call = _fake_call
    import tempfile
    with tempfile.TemporaryDirectory() as td:
        ok = ground.ground_stage(scene, judge, out_dir=td)
        assert ok, "grounding must succeed (row 1 grounded)"
        assert os.path.exists(os.path.join(td, "cluster_check.png")), \
            "cluster candidates image was not saved"
    assert len(scene.boxes) == 2, \
        f"missed row must be recovered by the cluster net, " \
        f"got {len(scene.boxes)} boxes"
    ys = sorted(b.center[1] for b in scene.boxes)
    assert abs(ys[0]) < 0.25, f"row1 y {ys[0]:.2f}"
    assert abs(ys[1] - 3.0) < 0.25, \
        f"recovered row must sit at y~3.0, got {ys[1]:.2f}"
    assert 5.0 < scene.boxes[-1].size[0] < 6.5, \
        "recovered row keeps its length"
    print("PASS ground stage cluster recall (missed row recovered)")


def test_ground_stage_adjacent_clump_recall():
    """The recall net's coverage must judge what was ACTUALLY
    DETECTED (user report: the net never fired, obvious rectangular
    clumps left unboxed). A generous VLM rect drawn over a row AND an
    adjacent AC-sized clump fits ONLY the row (the peak-peeling span
    keeps the dominant run) -- the old rect-overlap rules called the
    clump 'covered' by the same rect (forward containment on area),
    but its points sit in NO fitted box: point coverage must
    re-propose it."""
    from agentic_gts.agent import ground
    from agentic_gts.agent.judge import VLMJudge

    rng = np.random.default_rng(17)
    row = _row_points(0.0, 6.0, y=0.0, rng=rng)
    # LOW AC-sized clump 1.5m off the row's end (z up to 0.65m): the
    # rect's own fit NEVER sees it -- the XY fit's middle z-slice
    # starts at 0.35 x height ~0.74m, entirely above the clump -- so
    # the fitted box covers the row only, while the clump's area sits
    # inside the generous rect (the old area-overlap rules called
    # that 'covered' and the clump stayed missed)
    ac = rng.uniform([7.5, -0.5, 0.05], [8.5, 0.5, 0.65], (600, 3))
    ceil = np.column_stack([rng.uniform(-2.0, 9.5, 3000),
                            rng.uniform(-2.0, 2.0, 3000),
                            rng.uniform(2.9, 3.0, 3000)])
    scene = Scene(points=np.vstack([row, ac, ceil]))
    scene.meta["yaw"] = 0.0
    _bootstrap_meta(scene, (-1.5, -0.9, 9.5, 0.9))
    scene.boxes = []
    _, cam, W, H = ground._render_topdown(scene, 0.0)
    # ONE generous rect: the row plus the whole clump (the fit snaps
    # to the row's denser run; the clump's points stay unfitted)
    uv = cam.project_cv(np.column_stack(
        [[-0.6, 8.6, 8.6, -0.6], [-0.8, -0.8, 0.8, 0.8],
         np.full(4, 1.0)]))
    px = (max(float(uv[:, 0].min()), 0.0), max(float(uv[:, 1].min()), 0.0),
          min(float(uv[:, 0].max()), W), min(float(uv[:, 1].max()), H))
    import json as _json
    reply = ("Row and unit.\n" + _json.dumps(
        [{"bbox_2d": [int(round(px[0] / W * 1000)),
                      int(round(px[1] / H * 1000)),
                      int(round(px[2] / W * 1000)),
                      int(round(px[3] / H * 1000))],
          "label": "row"}]))

    judge = VLMJudge(backend="qwen")

    def _fake_call(png, prompt, *a, **k):
        return reply
    judge._qwen_image_call = _fake_call
    import tempfile
    with tempfile.TemporaryDirectory() as td:
        ok = ground.ground_stage(scene, judge, out_dir=td)
        assert ok, "grounding must succeed"
    assert len(scene.boxes) == 2, \
        f"row + re-proposed clump; got {len(scene.boxes)} boxes: " \
        + str([(round(b.center[0], 2), round(b.center[1], 2))
               for b in scene.boxes])
    acb = min(scene.boxes, key=lambda b: abs(b.center[0] - 8.0))
    assert abs(acb.center[0] - 8.0) < 0.3 and acb.size[1] > 0.5, \
        f"the clump must get its own box, got centre {acb.center[:2]} " \
        f"size {acb.size[:2]}"
    print("PASS adjacent clump recall (point coverage, not rect area)")


def test_floor_map_mesh_mode():
    """mesh_mode: a mesh sampling has no under-floor haze, so a tile's
    floor is its plain MINIMUM z -- even when the slab is sparsely
    sampled (min_pts relaxed) and there are no points below the floor
    to trick a percentile."""
    from agentic_gts.agent.ground import _floor_map
    rng = np.random.default_rng(25)
    slab_lo = rng.uniform([-6, -4, 0.0], [0, 4, 0.02], (400, 3))
    slab_hi = rng.uniform([0, -4, 0.40], [6, 4, 0.42], (400, 3))
    racks = np.vstack([
        rng.uniform([-4.5, -3.0, 0.0], [-3.5, 3.0, 2.1], (3000, 3)),
        rng.uniform([1.0, -3.0, 0.40], [2.0, 3.0, 2.50], (3000, 3))])
    pts = np.vstack([slab_lo, slab_hi, racks])
    fl = _floor_map(pts, mesh_mode=True)
    f_lo, f_hi = float(fl(-3.0, 0.0)), float(fl(3.0, 0.0))
    assert -0.05 < f_lo <= 0.02, f"lower floor {f_lo:.3f} (want the min)"
    assert 0.38 < f_hi <= 0.42, f"raised floor {f_hi:.3f} (want the min)"
    print(f"PASS floor map mesh mode (lower {f_lo:.3f}, raised {f_hi:.3f})")


def test_tile_frames():
    """Tiling decision: a layout that fits ONE nadir view stays
    single-view (no extra VLM calls); a big layout tiles with exact
    cover, bounded span and >= overlap so boundary structures appear
    whole in at least one tile."""
    from agentic_gts.agent.ground import (_MAX_SINGLE_SPAN, _TILE_OVERLAP,
                                           _tile_frames)
    assert _tile_frames(None) is None, "no layout -> single view"
    assert _tile_frames((np.array([0.0, 0.0]), np.array([20.0, 12.0]))) \
        is None, "20x12m fits one view -> single view"
    tiles = _tile_frames((np.array([0.0, 0.0]), np.array([60.0, 12.0])))
    assert tiles is not None, "60m span must tile"
    assert all(len(t) == 4 for t in tiles)
    xs = sorted(set((t[0], t[2]) for t in tiles))
    # exact cover of the split axis, span bounded, seams >= overlap
    assert xs[0][0] <= 0.0 and xs[-1][1] >= 60.0
    for (a0, a1), (b0, b1) in zip(xs, xs[1:]):
        assert b1 - b0 <= _MAX_SINGLE_SPAN + 1e-9
        assert a1 - b0 >= _TILE_OVERLAP - 1e-9, \
            f"adjacent tiles must overlap >= {_TILE_OVERLAP}m"
    for t in tiles:
        assert t[2] - t[0] <= _MAX_SINGLE_SPAN + 1e-9
    print(f"PASS tile frames (60m -> {len(tiles)} tiles, seams ok)")


def test_ground_stage_tiled_views():
    """End-to-end TILED grounding: a 40m layout exceeds one nadir view,
    so the stage renders per-tile cameras and calls the VLM once per
    tile; per-tile rects back-project through their OWN camera and the
    cross-tile row pieces must heal (adjacency merge) into the one
    40m row. Small layouts stay single-view (previous tests)."""
    import json as _json
    import os as _os
    import tempfile
    from agentic_gts.agent import ground
    from agentic_gts.agent.judge import VLMJudge

    rng = np.random.default_rng(31)
    pts = _row_points(0.0, 40.0, y=0.0, rng=rng, n=20000)
    scene = Scene(points=pts)
    scene.meta["yaw"] = 0.0
    _bootstrap_meta(scene, (-0.5, -0.8, 40.5, 0.8))
    scene.boxes = []
    # the VLM outlines the WHOLE tile (full-image rect) on every call
    reply = ("One long joined row.\n" + _json.dumps(
        [{"bbox_2d": [0, 0, 1000, 1000], "label": "row"}]))
    judge = VLMJudge(backend="qwen")
    judge._qwen_image_call = lambda png, prompt, *a, **k: reply
    with tempfile.TemporaryDirectory() as td:
        calls = []
        orig = judge._qwen_image_call

        def _count(png, prompt, *a, **k):
            calls.append(1)
            return orig(png, prompt, *a, **k)
        judge._qwen_image_call = _count
        ok = ground.ground_stage(scene, judge, out_dir=td)
        assert ok, "tiled grounding must succeed"
        assert len(calls) >= 2, \
            f"a 40m layout must be tiled (>=2 VLM calls), got {len(calls)}"
        assert _os.path.exists(_os.path.join(td, "groundview_t0.png")) \
            and _os.path.exists(_os.path.join(td, "grounded_t0.png")), \
            "tiled audit renders must be saved per tile"
        assert len(scene.boxes) == 1, \
            f"cross-tile row pieces must merge into one, " \
            f"got {len(scene.boxes)}"
        b = scene.boxes[0]
        assert 38.0 < b.size[0] < 41.5, \
            f"merged row length {b.size[0]:.2f} (want ~40m)"
        print(f"PASS tiled grounding ({len(calls)} VLM calls -> "
              f"{len(scene.boxes)} merged row of {b.size[0]:.1f}m)")


def test_ground_stage_row_along_y():
    """End-to-end grounding of a joined row running along the y-axis:
    the emitted box must carry yaw = pi/2, or the downstream local
    refine splits the row across its thickness (user report)."""
    import math
    import tempfile
    from agentic_gts.agent import ground
    from agentic_gts.agent.judge import VLMJudge

    rng = np.random.default_rng(3)
    row = _row_points(0.0, 6.0, rng=rng)
    row = row[:, [1, 0, 2]]       # the row now runs along y
    scene = Scene(points=row)
    scene.meta["yaw"] = 0.0
    _bootstrap_meta(scene, (-0.8, -0.5, 0.8, 6.5))
    scene.boxes = []
    _, cam, W, H = ground._render_topdown(scene, 0.0)
    # true rect: depth on x, length on y
    uv = cam.project_cv(np.column_stack(
        [[-0.8, 0.8, 0.8, -0.8], [-0.5, -0.5, 6.5, 6.5], np.full(4, 1.0)]))
    px = (np.clip(uv[:, 0].min(), 0, W), np.clip(uv[:, 1].min(), 0, H),
          np.clip(uv[:, 0].max(), 0, W), np.clip(uv[:, 1].max(), 0, H))
    reply = ('[{"bbox_2d": [%d, %d, %d, %d], "label": "row"}]'
             % (round(px[0] / W * 1000), round(px[1] / H * 1000),
                round(px[2] / W * 1000), round(px[3] / H * 1000)))
    judge = VLMJudge(backend="qwen")
    judge._qwen_image_call = lambda png, prompt, *a, **k: reply
    with tempfile.TemporaryDirectory() as td:
        ok = ground.ground_stage(scene, judge, out_dir=td)
        assert ok, "grounding must succeed on a y-running row"
    assert len(scene.boxes) == 1
    b = scene.boxes[0]
    assert abs(b.yaw - math.pi / 2.0) < 1e-6, \
        f"world yaw must be pi/2, got {b.yaw:.3f}"
    assert 5.5 < b.size[0] < 6.4, \
        f"size[0] must be the row length, got {b.size[0]:.2f}"
    assert 0.85 < b.size[1] < 1.35, \
        f"size[1] must be the depth, got {b.size[1]:.2f}"
    assert abs(b.center[1] - 3.0) < 0.2, "centre must sit on the row"
    print(f"PASS ground stage y-row "
          f"(yaw=pi/2, L={b.size[0]:.2f}, D={b.size[1]:.2f})")


def test_yaw_bootstrap_byproducts():
    """The hint-free bootstrap: estimate_yaw_detailed's surviving cells
    (device layout; walls dropped as boundary cells, ceiling outside
    the height band) must export z_top ~= the rack top (NOT the
    ceiling) and a footprint that excludes the walls -- these feed
    ground_stage's framing and ceiling cut when no hint boxes exist
    (user request: no-hint input)."""
    from agentic_gts.segment.orientation import estimate_yaw_detailed
    import math
    rng = np.random.default_rng(5)
    pts = np.vstack([
        _row_points(0.0, 6.0, y=0.0, rng=rng),      # rack row 1
        _row_points(0.0, 6.0, y=3.0, rng=rng),      # rack row 2
    ])
    # walls well OUTSIDE the layout (boundary cells get dropped)
    wall1 = np.column_stack([np.full(3000, -3.0),
                             rng.uniform(-2.0, 5.0, 3000),
                             rng.uniform(0.0, 3.0, 3000)])
    wall2 = np.column_stack([np.full(3000, 9.0),
                             rng.uniform(-2.0, 5.0, 3000),
                             rng.uniform(0.0, 3.0, 3000)])
    # ceiling slab at 2.9m spanning the whole room
    ceil = np.column_stack([rng.uniform(-3.5, 9.5, 3000),
                            rng.uniform(-2.5, 5.5, 3000),
                            np.full(3000, 2.9)])
    info = estimate_yaw_detailed(np.vstack([pts, wall1, wall2, ceil]))
    assert abs(math.degrees(info["yaw"])) < 3.0
    z_top = info["z_top"]
    assert z_top is not None, "the bootstrap must measure a device top"
    assert 1.9 < z_top < 2.2, \
        f"z_top must track the rack top (~2.1), got {z_top:.2f}"
    fp = info["device_footprint"]
    assert fp is not None
    assert -1.0 < fp[0] < 0.5 and 5.5 < fp[2] < 7.0, \
        f"footprint x must hug the rows, not the walls: {fp}"
    assert -1.0 < fp[1] < 0.5 and 2.5 < fp[3] < 4.0, \
        f"footprint y must hug the rows: {fp}"
    print(f"PASS yaw bootstrap byproducts "
          f"(z_top={z_top:.2f}, footprint={tuple(round(v, 2) for v in fp)})")


def test_ground_stage_bootstrap_driven():
    """Grounding end-to-end driven purely by the stage0 bootstrap
    byproducts (meta z_top + device_footprint): no box input of any
    kind. The VLM answer is fabricated by projecting the TRUE row
    rects through the same cam the renderer builds -- full
    pixel->world->fit path."""
    import tempfile
    from agentic_gts.agent import ground
    from agentic_gts.agent.judge import VLMJudge

    rng = np.random.default_rng(3)
    pts = np.vstack([_row_points(0.0, 6.0, y=0.0, rng=rng),
                     _row_points(-1.0, 5.0, y=3.0, rng=rng)])
    ceil = np.column_stack([rng.uniform(-2.0, 7.0, 3000),
                            rng.uniform(-2.0, 5.0, 3000),
                            np.full(3000, 2.9)])   # ceiling at 2.9m
    scene = Scene(points=np.vstack([pts, ceil]))
    scene.meta["yaw"] = 0.0
    _bootstrap_meta(scene, (-0.5, -0.8, 6.5, 3.8))
    scene.boxes = []
    _, cam, W, H = ground._render_topdown(scene, 0.0)
    true_rects = [(( -0.5, 6.5), (-0.8, 0.8)),
                  ((-1.5, 5.5), (2.2, 3.8))]
    import json as _json
    regions = []
    for (xa, xb), (ya, yb) in true_rects:
        uv = cam.project_cv(np.column_stack([
            [xa, xb, xb, xa], [ya, ya, yb, yb], np.full(4, 1.0)]))
        px = (np.clip(uv[:, 0].min(), 0, W), np.clip(uv[:, 1].min(), 0, H),
              np.clip(uv[:, 0].max(), 0, W), np.clip(uv[:, 1].max(), 0, H))
        regions.append({"bbox_2d": [
            int(round(px[0] / W * 1000)), int(round(px[1] / H * 1000)),
            int(round(px[2] / W * 1000)), int(round(px[3] / H * 1000))],
            "label": "row"})
    reply = _json.dumps(regions)
    judge = VLMJudge(backend="qwen")
    judge._qwen_image_call = lambda png, prompt, *a, **k: reply
    with tempfile.TemporaryDirectory() as td:
        ok = ground.ground_stage(scene, judge, out_dir=td)
        assert ok, "grounding must succeed on the bootstrap footprint"
    assert len(scene.boxes) == 2, \
        f"two rows grounded from the bootstrap footprint, got {len(scene.boxes)}"
    rows = sorted(scene.boxes, key=lambda b: b.center[1])
    for b in rows:
        assert 5.0 < b.size[0] < 6.5, f"length {b.size[0]:.2f}"
        assert 0.85 < b.size[1] < 1.35, f"depth {b.size[1]:.2f} (FULL)"
        assert 1.9 < b.size[2] < 2.35, \
            f"height {b.size[2]:.2f} (ceiling must be excluded!)"
    print(f"PASS bootstrap-driven ground stage "
          f"(row1 {rows[0].size[0]:.2f}x{rows[0].size[1]:.2f}, "
          f"row2 {rows[1].size[0]:.2f}x{rows[1].size[1]:.2f})")


def test_nadir_framing_tight_for_rotated_layout():
    """Regression: rotated layouts must not DOUBLE-inflate the nadir
    framing. The old path AABB'd the kept cells in WORLD frame (a 45-deg
    row layout bounds to a big square) and then AABB'd again after
    rotating that square by -yaw -- the camera rose and groundview came
    back mostly empty. The fix rotates the CELLS by the actual yaw and
    AABBs once, so the devices fill the frame again."""
    import math
    from agentic_gts.agent import ground
    yaw = math.radians(45.0)
    rng = np.random.default_rng(3)
    # 4 rows of 6m, 2m apart, all running at 45 deg in world frame
    rows = []
    for k in range(4):
        t = rng.uniform(0.0, 6.0, 300)
        off = k * 2.0
        x = t * math.cos(yaw) - off * math.sin(yaw)
        y = t * math.sin(yaw) + off * math.cos(yaw)
        rows.append(np.column_stack([x, y]))
    cells = np.vstack(rows)
    # cloud: thin vertical walls along the same rows (scatter path)
    pts = []
    for k in range(4):
        t = rng.uniform(0.0, 6.0, 1500)
        off = k * 2.0
        pts.append(np.column_stack([
            t * math.cos(yaw) - off * math.sin(yaw),
            t * math.sin(yaw) + off * math.cos(yaw),
            rng.uniform(0.4, 1.9, 1500)]))
    scene = Scene(points=np.vstack(pts))
    scene.meta["z_top"] = 2.1
    scene.meta["device_cells"] = cells
    scene.meta["device_footprint"] = (
        float(cells[:, 0].min()), float(cells[:, 1].min()),
        float(cells[:, 0].max()), float(cells[:, 1].max()))
    _, cam, W, H = ground._render_topdown(scene, yaw)
    # measure at the TOP of the band: the perspective nadir camera is
    # fitted so the rack-top ring fills the frame (it is closest to the
    # eye and spreads most); the floor ring is necessarily smaller
    uv = cam.project_cv(np.column_stack([cells, np.full(len(cells), 1.9)]))
    fill_w = (uv[:, 0].max() - uv[:, 0].min()) / W
    fill_h = (uv[:, 1].max() - uv[:, 1].min()) / H
    # tight framing: with the fix the top ring fills most of the frame;
    # the double-inflated framing showed it at ~39% width / ~48% height
    assert fill_w > 0.6, f"cells fill only {fill_w:.0%} of the width"
    assert fill_h > 0.85, f"cells fill only {fill_h:.0%} of the height"
    print(f"PASS rotated-layout framing tight (fill {fill_w:.0%} x {fill_h:.0%})")


def test_render_cut_relative_not_conservative():
    """The nadir render cut is RELATIVE (user decision: the view only
    needs each device's basic features, not a complete structure, and
    the relative cut also tolerates a top reference dragged upward by
    trays/ceiling). Tall structures trim to 70%; low structures keep
    nearly everything; the fit pool is independent so box heights are
    unaffected."""
    from agentic_gts.agent.ground import _render_cut
    import math
    # tall racks: 2.1m top -> 1.47m cut (70%), not 2.1-0.45=1.65
    assert abs(_render_cut(2.1) - 1.47) < 1e-9
    # over-estimated top (trays dragged it to 2.6): still a deep cut,
    # the real 2.1m racks render fully below it
    assert _render_cut(2.6) < 2.0
    # LOW structures keep nearly everything (0.9m bank -> 0.8m cut)
    assert abs(_render_cut(0.9) - 0.8) < 1e-9
    # no reference at all -> no cut
    assert math.isinf(_render_cut(None)) and math.isinf(_render_cut(0.0))
    print("PASS relative render cut (70% of top, low structures kept)")


def test_parse_ground_regions_official_format():
    """The official Qwen3-VL grounding reply format (per the 2d_grounding
    cookbook) parses correctly: bare JSON array of {"bbox_2d": [x1,y1,
    x2,y2]} in RELATIVE 0-1000 coords, possibly behind markdown fences
    / a thinking preamble. Legacy {"regions": [...]} pixel dicts still
    parse (backward robustness), and pixel replies with values >1000
    are detected as absolute."""
    from agentic_gts.agent.judge import _parse_ground_regions
    W, H = 1280, 1024
    # official: relative 0-1000, fenced, after a reasoning preamble
    text = ("Thinking... two rows visible.\n```json\n"
            '[{"bbox_2d": [100, 200, 900, 400], "label": "row 1"},\n'
            ' {"bbox_2d": [100, 500, 900, 700], "label": "row 2"}]\n'
            "```")
    rects = _parse_ground_regions(text, W, H)
    assert len(rects) == 2, f"want 2 rects, got {len(rects)}"
    r = rects[0]
    assert abs(r[0] - 128.0) < 1e-6 and abs(r[1] - 204.8) < 1e-6, \
        "0-1000 relative coords must rescale by W/1000, H/1000"
    assert abs(r[2] - 1152.0) < 1e-6 and abs(r[3] - 409.6) < 1e-6
    # absolute pixels (values > 1000 -> treated as pixels, no scaling)
    px = _parse_ground_regions(
        '[{"bbox_2d": [110, 120, 1150, 900]}]', W, H)
    assert abs(px[0][2] - 1150.0) < 1e-6, "pixel replies must not rescale"
    # legacy dict format still honoured (label defaults to "device")
    lg = _parse_ground_regions(
        '{"regions": [{"x0": 10, "y0": 20, "x1": 30, "y1": 40}]}', W, H)
    assert lg[0][:4] == (10.0, 20.0, 30.0, 40.0)
    assert lg[0][4] == "device", "missing label defaults to 'device'"
    assert _parse_ground_regions(
        '[{"bbox_2d": [100, 200, 900, 400], "label": "rack row"}]',
        W, H)[0][4] == "rack row"
    # noise / no JSON -> nothing
    assert _parse_ground_regions("just prose, no json", W, H) == []
    print("PASS parse official bbox_2d (0-1000 relative, fences, legacy)")


def test_parse_ground_regions_salvage():
    """Truncated + malformed reply still yields every COMPLETE bbox.

    Real Qwen reply shape on row-heavy rooms (user report): objects
    wrapped in parentheses instead of a JSON array, 'bbox 2d' /
    'bbox _2d' key typos, and the tail cut off mid-item by the token
    budget. The salvage scanner must recover all complete boxes and
    silently drop the truncated one."""
    from agentic_gts.agent.judge import _parse_ground_regions
    txt = (
        '("bbox 2d": [328, 44, 400, 118], "label": "server rack row"),\n'
        '("bbox_2d": [424, 50, 493, 118], "label": "server rack row"),\n'
        "('bbox _2d': [514, 73, 643, 127], \"label\": \"row\"),\n"
        '("bbox 2d": [645, 82, 782, 133], "label": "server rack row"),\n'
        '("bbox 2d": [627, 870, 667, 92'      # truncated tail, no match
    )
    px = _parse_ground_regions(txt, 1280, 1024)
    assert len(px) == 4, f"salvage must find 4 complete boxes, got {len(px)}"
    # relative 0-1000 grid -> pixel scaling
    assert abs(px[0][0] - 328 / 1000 * 1280) < 1e-6
    assert abs(px[0][3] - 118 / 1000 * 1024) < 1e-6
    assert px[0][4] == "server rack row"
    assert px[2][4] == "row"
    # >1000 value = absolute pixels, not rescaled
    big = _parse_ground_regions(
        '("bbox 2d": [1100, 20, 1200, 900])', 1280, 1024)
    assert big and abs(big[0][2] - 1200.0) < 1e-6
    print("PASS parse salvage (truncated + malformed reply recovered)")


def test_ground_mock_returns_false():
    """Mock backend / no VLM -> grounding fails soft: the scene stays
    EMPTY (there are no fallback boxes without hint input).

    The failure must also be VISIBLE: grounded.png is written with a
    red GROUNDING FAILED banner (previously it only appeared on
    success, so a failed run left nothing but raw renders)."""
    import tempfile
    from agentic_gts.agent import ground
    from agentic_gts.agent.judge import VLMJudge
    rng = np.random.default_rng(3)
    pts = _row_points(0.0, 6.0, rng=rng)
    scene = Scene(points=pts)
    scene.meta["yaw"] = 0.0
    _bootstrap_meta(scene, (-0.5, -0.8, 6.5, 0.8))
    scene.boxes = []
    judge = VLMJudge(backend="mock")
    with tempfile.TemporaryDirectory() as td:
        assert ground.ground_stage(scene, judge, out_dir=td) is False
        gpng = os.path.join(td, "grounded.png")
        assert os.path.isfile(gpng) and os.path.getsize(gpng) > 500, \
            "failure audit grounded.png (banner) must be written"
    assert scene.boxes == [], \
        "grounding failure must leave the scene empty (no fallback)"
    print("PASS grounding fails soft (mock, scene stays empty)")


def test_merge_adjacent_boxes():
    """Tightly-adjacent over-split pieces of ONE row merge into their
    point-support-refitted union; separate rows with an EMPTY lateral
    gap and corner-kiss boxes stay separate."""
    from agentic_gts.agent.ground import _merge_adjacent_boxes
    from agentic_gts.core.models import OrientedBox
    rng = np.random.default_rng(11)
    row = _row_points(0.0, 6.0, rng=rng)
    pts_fit = row[row[:, 2] > 0.30]
    # two pieces of one CONTINUOUS row, fitted with a 0.2m seam where
    # the VLM drew the rect boundary (a regular layout reads as
    # several bands) -- the device band fills the seam -> merge
    a = OrientedBox(center=(1.45, 0.0, 1.05), size=(2.9, 1.1, 2.1),
                    yaw=0.0)
    b = OrientedBox(center=(4.55, 0.0, 1.05), size=(2.9, 1.1, 2.1),
                    yaw=0.0)
    out = _merge_adjacent_boxes([a, b], pts_fit, 0.0)
    assert len(out) == 1, \
        f"continuous-row pieces must merge, got {len(out)}"
    m = out[0]
    assert 5.5 < m.size[0] < 6.4, f"merged length {m.size[0]:.2f} (want ~6.0)"
    assert 0.85 < m.size[1] < 1.35, f"merged depth {m.size[1]:.2f}"
    assert m.meta.get("merged_from") == 2
    # separate rows, 0.4m lateral gap, NOTHING between: the facing
    # SURFACES sit exactly at the gap edges and must NOT count as
    # bridge points -> no merge (old B0 convention)
    r1 = OrientedBox(center=(3.0, 0.0, 1.05), size=(6.0, 1.1, 2.1),
                    yaw=0.0)
    r2 = OrientedBox(center=(3.0, 1.5, 1.05), size=(6.0, 1.1, 2.1),
                    yaw=0.0)
    row2 = _row_points(0.0, 6.0, y=1.5, rng=rng)
    pf2 = np.vstack([pts_fit, row2[row2[:, 2] > 0.30]])
    out2 = _merge_adjacent_boxes([r1, r2], pf2, 0.0)
    assert len(out2) == 2, "empty-gap rows must stay separate"
    # corner kiss: gap ok on both axes but NO shared band -> no merge
    c1 = OrientedBox(center=(0.5, 0.5, 1.05), size=(1.0, 1.0, 2.1),
                     yaw=0.0)
    c2 = OrientedBox(center=(1.7, 1.7, 1.05), size=(1.0, 1.0, 2.1),
                     yaw=0.0)
    out3 = _merge_adjacent_boxes([c1, c2], pf2, 0.0)
    assert len(out3) == 2, "corner-kiss boxes must not merge"
    # FALSE-BRIDGE regression (user report: two parallel rows a clear
    # aisle apart got merged): the fitted AABBs are inflated by 3DGS
    # aisle haze until only a narrow haze-filled gap remains. The haze
    # points OUTNUMBER the count threshold (old code: touch shortcut or
    # bare count -> merged); the density test must reject them -- haze
    # is orders of magnitude sparser than a cabinet surface
    r1f = OrientedBox(center=(3.0, 0.0, 1.05), size=(6.0, 1.3, 2.1),
                     yaw=0.0)                       # inflated to y<=0.65
    r2f = OrientedBox(center=(3.0, 1.55, 1.05), size=(6.0, 1.3, 2.1),
                     yaw=0.0)                      # inflated to y>=0.90
    haze = np.column_stack([rng.uniform(0.0, 6.0, 60),
                            rng.uniform(0.70, 0.85, 60),
                            rng.uniform(0.5, 2.0, 60)])
    pf3 = np.vstack([pts_fit, row2[row2[:, 2] > 0.30], haze])
    out4 = _merge_adjacent_boxes([r1f, r2f], pf3, 0.0)
    assert len(out4) == 2, \
        "haze-filled gap between separate rows must NOT merge (density)"
    # and the same geometry with a DENSE bridge (the row really is
    # continuous through the gap) must still merge
    bridge = np.column_stack([rng.uniform(0.0, 6.0, 4000),
                              rng.uniform(0.60, 0.95, 4000),
                              rng.uniform(0.0, 2.1, 4000)])
    out5 = _merge_adjacent_boxes([r1f, r2f],
                                 np.vstack([pf3, bridge]), 0.0)
    assert len(out5) == 1, "dense device bridge must merge"
    print(f"PASS adjacency merge ({m.size[0]:.2f}m union; empty-gap, "
          f"corner-kiss and haze-gap kept separate; dense bridge merged)")


def test_merge_adjacent_tray_bridge():
    """MESH tray bridge (user report: close devices with NO overlap on
    the groundview still merged): the fit pool (0.30 .. z_top+0.10)
    carries a dense cable-tray strip spanning the gap ABOVE the
    devices -- dense enough to pass the density ratio like a real
    seam. With probe_pool (the render-cut pool, trays removed) the
    junction holds nothing and the boxes stay separate; the same tray
    WITHOUT the probe pool still merges (the old behaviour, guard)."""
    from agentic_gts.agent.ground import _merge_adjacent_boxes
    from agentic_gts.core.models import OrientedBox
    rng = np.random.default_rng(41)
    rowA = _row_points(0.0, 6.0, y=0.0, rng=rng)
    rowB = _row_points(0.0, 6.0, y=1.6, rng=rng)     # 0.5m lateral gap
    fit_pool = np.vstack([rowA[rowA[:, 2] > 0.30],
                          rowB[rowB[:, 2] > 0.30]])
    # dense tray strip bridging the gap ABOVE the rows (mesh trays are
    # as dense as device surfaces -- the density ratio cannot tell)
    tray = np.column_stack([rng.uniform(0.0, 6.0, 5000),
                            rng.uniform(0.5, 1.1, 5000),
                            rng.uniform(2.2, 2.4, 5000)])
    pf = np.vstack([fit_pool, tray])
    a = OrientedBox(center=(3.0, 0.0, 1.05), size=(6.0, 1.1, 2.1),
                    yaw=0.0)
    b = OrientedBox(center=(3.0, 1.6, 1.05), size=(6.0, 1.1, 2.1),
                    yaw=0.0)
    # old behaviour: the fit pool's tray passes the probe -> merged
    out_old = _merge_adjacent_boxes([a, b], pf, 0.0)
    assert len(out_old) == 1, \
        "guard: the tray DOES bridge the fit-pool probe (old bug)"
    # fixed: the render-cut pool (tray above the cut removed) leaves
    # the junction empty -> separate
    out = _merge_adjacent_boxes([a, b], pf, 0.0, probe_pool=fit_pool)
    assert len(out) == 2, \
        f"tray-bridged close devices must stay separate on the " \
        f"render-cut pool; got {len(out)}"
    print("PASS tray bridge (probe on render-cut pool stays separate)")


def test_containment_2d_nested():
    """containment_2d sees what iou_2d cannot: a small box nested in a
    big one has IoU = area ratio (< 0.5) but containment ~1.0."""
    from agentic_gts.core.models import OrientedBox
    import math
    big = OrientedBox(center=(3.0, 0.0, 1.05), size=(6.0, 1.1, 2.1),
                     yaw=0.0)
    # cross-ways small box INSIDE the row footprint (size[0] rides the
    # world y axis at yaw=pi/2 -> world extent 1.6 x 1.0): iou < 0.5
    small = OrientedBox(center=(3.0, 0.0, 1.05), size=(1.0, 1.6, 2.1),
                        yaw=math.pi / 2.0)
    assert small.iou_2d(big) < 0.5, "precondition: IoU blind to nesting"
    assert small.containment_2d(big) >= 0.99, \
        f"nested box containment {small.containment_2d(big):.2f} (want ~1.0)"
    # reverse direction: the big box is NOT contained in the small one
    # (only its area ratio, 24% here -- far under any drop threshold)
    assert big.containment_2d(small) < 0.5
    # disjoint boxes: no containment either way
    far = OrientedBox(center=(3.0, 5.0, 1.05), size=(1.0, 1.0, 2.1),
                     yaw=0.0)
    assert far.containment_2d(big) == 0.0 and big.containment_2d(far) == 0.0
    print("PASS containment_2d (nested cross-yaw seen, disjoint zero)")


def test_ground_stage_nested_region_dropped():
    """End-to-end: the VLM outlines the WHOLE row and ALSO a sub-section
    of it cross-ways (rect taller than wide in the row frame -> fitted
    yaw=pi/2). The nested small box must be DROPPED by the containment
    guard, not survive as a box-inside-box on the audit render."""
    import json as _json
    import tempfile
    from agentic_gts.agent import ground
    from agentic_gts.agent.judge import VLMJudge

    rng = np.random.default_rng(3)
    pts = np.vstack([_row_points(0.0, 6.0, y=0.0, rng=rng),
                     _row_points(-1.0, 5.0, y=3.0, rng=rng)])
    ceil = np.column_stack([rng.uniform(-2.0, 7.0, 3000),
                            rng.uniform(-2.0, 5.0, 3000),
                            np.full(3000, 2.9)])
    scene = Scene(points=np.vstack([pts, ceil]))
    scene.meta["yaw"] = 0.0
    _bootstrap_meta(scene, (-1.5, -0.8, 6.5, 3.8))
    scene.boxes = []
    _, cam, W, H = ground._render_topdown(scene, 0.0)
    # whole row 1, a cross-ways SUB-rect of row 1 (nested), whole row 2
    true_rects = [((-0.5, 6.5), (-0.8, 0.8)),
                  ((2.4, 3.6), (-0.8, 0.8)),
                  ((-1.5, 5.5), (2.2, 3.8))]
    regions = []
    for (xa, xb), (ya, yb) in true_rects:
        uv = cam.project_cv(np.column_stack([
            [xa, xb, xb, xa], [ya, ya, yb, yb], np.full(4, 1.0)]))
        px = (np.clip(uv[:, 0].min(), 0, W), np.clip(uv[:, 1].min(), 0, H),
              np.clip(uv[:, 0].max(), 0, W), np.clip(uv[:, 1].max(), 0, H))
        regions.append({"bbox_2d": [
            int(round(px[0] / W * 1000)), int(round(px[1] / H * 1000)),
            int(round(px[2] / W * 1000)), int(round(px[3] / H * 1000))],
            "label": "server rack"})
    judge = VLMJudge(backend="qwen")
    judge._qwen_image_call = lambda png, prompt, *a, **k: \
        _json.dumps(regions)
    with tempfile.TemporaryDirectory() as td:
        ok = ground.ground_stage(scene, judge, out_dir=td)
        assert ok
    assert len(scene.boxes) == 2, \
        f"nested sub-box must be dropped (row box + row 2), " \
        f"got {len(scene.boxes)}"
    rows = sorted(scene.boxes, key=lambda b: b.center[1])
    assert 5.5 < rows[0].size[0] < 6.5, \
        f"row 1 must stay the full-row box, got {rows[0].size[0]:.2f}m"
    assert abs(rows[0].center[1]) < 0.2
    assert abs(rows[1].center[1] - 3.0) < 0.2, "row 2 untouched"
    print("PASS nested region dropped by the containment guard")


def test_ground_stage_merges_over_split_regions():
    """End-to-end: the VLM over-split ONE row into two TIGHT rects
    (regular layout mis-read); the two fitted pieces must merge back
    into a single full-row box while the separate row keeps its own."""
    import json as _json
    import tempfile
    from agentic_gts.agent import ground
    from agentic_gts.agent.judge import VLMJudge

    rng = np.random.default_rng(3)
    pts = np.vstack([_row_points(0.0, 6.0, y=0.0, rng=rng),
                     _row_points(-1.0, 5.0, y=3.0, rng=rng)])
    ceil = np.column_stack([rng.uniform(-2.0, 7.0, 3000),
                            rng.uniform(-2.0, 5.0, 3000),
                            np.full(3000, 2.9)])
    scene = Scene(points=np.vstack([pts, ceil]))
    scene.meta["yaw"] = 0.0
    _bootstrap_meta(scene, (-1.5, -0.8, 6.5, 3.8))
    scene.boxes = []
    _, cam, W, H = ground._render_topdown(scene, 0.0)
    # row 1 over-split into two rects TOUCHING at x=3.0; row 2 whole
    true_rects = [((-0.5, 3.0), (-0.8, 0.8)),
                  ((3.0, 6.5), (-0.8, 0.8)),
                  ((-1.5, 5.5), (2.2, 3.8))]
    regions = []
    for (xa, xb), (ya, yb) in true_rects:
        uv = cam.project_cv(np.column_stack([
            [xa, xb, xb, xa], [ya, ya, yb, yb], np.full(4, 1.0)]))
        px = (np.clip(uv[:, 0].min(), 0, W), np.clip(uv[:, 1].min(), 0, H),
              np.clip(uv[:, 0].max(), 0, W), np.clip(uv[:, 1].max(), 0, H))
        regions.append({"bbox_2d": [
            int(round(px[0] / W * 1000)), int(round(px[1] / H * 1000)),
            int(round(px[2] / W * 1000)), int(round(px[3] / H * 1000))],
            "label": "server rack"})
    reply = _json.dumps(regions)
    judge = VLMJudge(backend="qwen")
    judge._qwen_image_call = lambda png, prompt, *a, **k: reply
    with tempfile.TemporaryDirectory() as td:
        ok = ground.ground_stage(scene, judge, out_dir=td)
        assert ok
    assert len(scene.boxes) == 2, \
        f"over-split row must merge to ONE box (+1 for row 2), " \
        f"got {len(scene.boxes)}"
    rows = sorted(scene.boxes, key=lambda b: b.center[1])
    assert 5.0 < rows[0].size[0] < 6.5, \
        f"merged row length {rows[0].size[0]:.2f} (want ~6.0)"
    assert 0.85 < rows[0].size[1] < 1.35
    assert abs(rows[0].center[1]) < 0.2
    assert 5.0 < rows[1].size[0] < 6.5, "row 2 untouched by the merge"
    assert abs(rows[1].center[1] - 3.0) < 0.2
    print(f"PASS ground-stage adjacency merge "
          f"(over-split row healed to {rows[0].size[0]:.2f}m)")


if __name__ == "__main__":
    test_unproject_ground_roundtrip()
    test_fit_region_box_full_depth()
    test_fit_region_box_haze_immune()
    test_fit_region_box_starved_back_face()
    test_floor_map_stepped()
    test_fit_region_box_stepped_floor()
    test_render_cut_mesh_mode()
    test_floor_map_mesh_mode()
    test_fit_region_boxes_two_rows_in_one_rect()
    test_fit_region_boxes_solid_deep_structure_kept()
    test_tile_frames()
    test_ground_stage_tiled_views()
    test_robust_span_bin_boundary()
    test_fit_region_box_row_along_y()
    test_ground_stage_with_patched_vlm()
    test_ground_stage_row_along_y()
    test_yaw_bootstrap_byproducts()
    test_ground_stage_bootstrap_driven()
    test_render_cut_relative_not_conservative()
    test_parse_ground_regions_official_format()
    test_parse_ground_regions_salvage()
    test_ground_mock_returns_false()
    test_merge_adjacent_boxes()
    test_containment_2d_nested()
    test_ground_stage_nested_region_dropped()
    test_ground_stage_merges_over_split_regions()
    print("ALL GROUND TESTS PASSED")
