"""Tests for the VLM 2D grounding stage (no server / GPU needed).

Covers:
  - unproject_ground: pixel -> world roundtrip through the god-view cam
  - _fit_region_box: a region rect becomes a FULL-DEPTH box (the
    thin-fragment killer: both face bands + hollow interior inside)
  - ground_stage end-to-end with a patched VLM answer (scatter path)
"""
from __future__ import annotations

import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from agentic_gts.core.models import Scene


def _row_points(x0, x1, y=0.0, depth=1.1, height=2.1, rng=None, n=6000):
    """A closed-cabinet row: two face bands + hollow interior (surface
    points only, like a real 3DGS of a closed rack row)."""
    rng = rng or np.random.default_rng(0)
    half = depth / 2.0
    pts = []
    for face in (+half, -half):     # front / back bands
        pts.append(np.column_stack([rng.uniform(x0, x1, n),
                                    np.full(n, y + face),
                                    rng.uniform(0.0, height, n)]))
    return np.vstack(pts)


def _hint(cx, cy, size=(0.6, 0.5, 2.1)):
    from agentic_gts.core.models import OrientedBox
    return OrientedBox(center=(cx, cy, size[2] / 2), size=size, yaw=0.0)


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
    # hints: fragmented thin boxes (what the detector would give)
    scene.boxes = [_hint(1.0, 0.0, size=(1.6, 0.45, 2.1)),
                   _hint(4.0, 3.0, size=(1.2, 0.4, 2.1))]
    # build the same deterministic render to fabricate the VLM answer
    _, cam, W, H = ground._render_topdown(scene, scene.boxes, 0.0)
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
    # realistic detector fragments: the row covered by SEVERAL hint
    # pieces (the framing footprint is the boxes' union, so a single
    # tiny centre hint would clip the row out of the nadir frame)
    scene.boxes = [_hint(0.0, 1.0, size=(0.45, 1.2, 2.1)),
                   _hint(0.0, 3.0, size=(0.45, 1.2, 2.1)),
                   _hint(0.0, 5.0, size=(0.45, 1.2, 2.1))]
    _, cam, W, H = ground._render_topdown(scene, scene.boxes, 0.0)
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


def test_ground_stage_without_hints():
    """Hint-FREE grounding end-to-end (user request): no initial boxes;
    the stage0 bootstrap byproducts (meta z_top + device_footprint)
    drive the nadir framing and the ceiling cut. The VLM answer is
    fabricated by projecting the TRUE row rects through the same cam
    the renderer builds -- full pixel->world->fit path, no hints."""
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
    scene.meta["z_top"] = 2.1            # what stage0 exports for racks
    scene.meta["device_footprint"] = (-0.5, -0.8, 6.5, 3.8)
    scene.boxes = []                     # NO hint boxes at all
    _, cam, W, H = ground._render_topdown(scene, [], 0.0)
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
        f"two rows grounded without any hint boxes, got {len(scene.boxes)}"
    rows = sorted(scene.boxes, key=lambda b: b.center[1])
    for b in rows:
        assert 5.0 < b.size[0] < 6.5, f"length {b.size[0]:.2f}"
        assert 0.85 < b.size[1] < 1.35, f"depth {b.size[1]:.2f} (FULL)"
        assert 1.9 < b.size[2] < 2.35, \
            f"height {b.size[2]:.2f} (ceiling must be excluded!)"
    print(f"PASS hint-free ground stage "
          f"(row1 {rows[0].size[0]:.2f}x{rows[0].size[1]:.2f}, "
          f"row2 {rows[1].size[0]:.2f}x{rows[1].size[1]:.2f})")


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
    """Mock backend / no VLM -> grounding must fail soft, keeping hints.

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
    hints = [_hint(3.0, 0.0)]
    scene.boxes = list(hints)
    judge = VLMJudge(backend="mock")
    with tempfile.TemporaryDirectory() as td:
        assert ground.ground_stage(scene, judge, out_dir=td) is False
        gpng = os.path.join(td, "grounded.png")
        assert os.path.isfile(gpng) and os.path.getsize(gpng) > 500, \
            "failure audit grounded.png (banner) must be written"
    assert len(scene.boxes) == 1 and scene.boxes[0] is hints[0], \
        "hints must be kept untouched on grounding failure"
    print("PASS grounding fails soft (mock keeps hints)")


if __name__ == "__main__":
    test_unproject_ground_roundtrip()
    test_fit_region_box_full_depth()
    test_fit_region_box_row_along_y()
    test_ground_stage_with_patched_vlm()
    test_ground_stage_row_along_y()
    test_yaw_bootstrap_byproducts()
    test_ground_stage_without_hints()
    test_parse_ground_regions_official_format()
    test_parse_ground_regions_salvage()
    test_ground_mock_returns_false()
    print("ALL GROUND TESTS PASSED")
