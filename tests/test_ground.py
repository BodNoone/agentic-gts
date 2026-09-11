"""Tests for the VLM 2D grounding + split stage (no server / GPU needed).

Covers:
  - unproject_ground: pixel -> world roundtrip through the god-view cam
  - _fit_region_box: a region rect becomes a FULL-DEPTH box (the
    thin-fragment killer: both face bands + hollow interior inside)
  - _split_row: VLM gap fractions snap to the measured density gaps
  - ground_stage end-to-end with a patched VLM answer (scatter path)
  - _parse_split_reply robustness
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


def test_split_row_snaps_to_profile_gaps():
    from agentic_gts.agent.ground import _split_row
    rng = np.random.default_rng(11)
    # 3 cabinets of 1.8m separated by 0.2m real gaps
    pts = np.vstack([_row_points(0.0, 1.8, rng=rng, n=3000),
                     _row_points(2.0, 3.8, rng=rng, n=3000),
                     _row_points(4.0, 5.8, rng=rng, n=3000)])
    scene = Scene(points=pts)
    row = _hint(2.9, 0.0, size=(5.8, 1.1, 2.1))
    # VLM nominates COARSE fractions (0.30 / 0.72); the measured gaps are
    # at x ~1.9 / 3.9 (local ~ -1.0 / +1.0) -- geometry must snap there
    subs = _split_row(scene, row, 3, [0.30, 0.72])
    assert len(subs) == 3, f"expected 3 cabinets, got {len(subs)}"
    centers = sorted(s.center[0] for s in subs)
    assert abs(centers[0] - 0.9) < 0.15, f"cabinet 1 centre {centers[0]:.2f}"
    assert abs(centers[1] - 2.9) < 0.15, f"cabinet 2 centre {centers[1]:.2f}"
    assert abs(centers[2] - 4.9) < 0.15, f"cabinet 3 centre {centers[2]:.2f}"
    for s in subs:
        assert 1.4 < s.size[0] < 2.1, f"piece length {s.size[0]:.2f}"
        assert 0.7 < s.size[1] < 1.4, f"piece depth {s.size[1]:.2f} (kept full)"
    print(f"PASS split snaps to gaps ({len(subs)} cabinets, centres "
          f"{[round(c, 2) for c in centers]})")


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
    scene = Scene(points=pts)
    scene.meta["yaw"] = 0.0
    # hints: fragmented thin boxes (what the detector would give)
    scene.boxes = [_hint(1.0, 0.0, size=(1.6, 0.45, 2.1)),
                   _hint(4.0, 3.0, size=(1.2, 0.4, 2.1))]
    # build the same deterministic render to fabricate the VLM answer
    _, cam, W, H = ground._render_topdown(scene, scene.boxes, 0.0)
    true_rects = [((-0.5, 6.5), (-0.8, 0.8)),      # row 1 XY
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
        # tilted views answer with nothing extra here: the nadir view
        # alone grounds both rows (the multi-view UNION path is covered
        # by test_merge_rects)
        if "tilted" in prompt.lower():
            return "I see rows but nothing new.\n[]"
        return reply
    judge._qwen_image_call = _fake_call    # canned VLM answer
    import tempfile
    with tempfile.TemporaryDirectory() as td:
        ok = ground.ground_stage(scene, judge, out_dir=td)
        assert ok, "grounding must succeed with a valid VLM reply"
        # the result audit image must exist: red grounded boxes only
        # (no initial-hint overlay)
        assert os.path.exists(os.path.join(td, "grounded.png")), \
            "grounded.png (result audit view) was not saved"
        assert os.path.exists(os.path.join(td, "groundview.png")), \
            "groundview.png (input view) was not saved"
        # oblique complement views are saved too (the nadir blind-spot
        # fix -- centre rows with untrained tops)
        assert os.path.exists(os.path.join(td, "groundview_az90.png")), \
            "groundview_az90.png (oblique view) was not saved"
        assert os.path.exists(os.path.join(td, "groundview_az270.png")), \
            "groundview_az270.png (oblique view) was not saved"
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
    assert len(scene.boxes) == 2, f"want 2 row boxes, got {len(scene.boxes)}"
    rows = sorted(scene.boxes, key=lambda b: b.center[1])
    # row 1: full length ~6m, FULL depth ~1.1m, height ~2.1m
    b = rows[0]
    assert 5.5 < b.size[0] < 6.5, f"row1 length {b.size[0]:.2f}"
    assert 0.85 < b.size[1] < 1.35, f"row1 depth {b.size[1]:.2f} (FULL, not thin)"
    assert abs(b.center[1]) < 0.2, f"row1 y {b.center[1]:.2f}"
    # row 2
    b2 = rows[1]
    assert 5.5 < b2.size[0] < 6.5 and 0.85 < b2.size[1] < 1.35
    assert abs(b2.center[1] - 3.0) < 0.2
    print(f"PASS ground stage end-to-end "
          f"(row1 {rows[0].size[0]:.2f}x{rows[0].size[1]:.2f}, "
          f"row2 {b2.size[0]:.2f}x{b2.size[1]:.2f})")


def test_split_reply_parse():
    from agentic_gts.agent.judge import VLMJudge
    p = VLMJudge._parse_split_reply(
        'Three cabinets, boundaries near a third and two thirds.\n'
        '{"count": 3, "gaps": [0.33, 0.67]}')
    assert p == {"count": 3, "gaps": [0.33, 0.67]}
    # garbage / nonsense -> None (keep whole: the safe default)
    assert VLMJudge._parse_split_reply("cannot tell") is None
    # out-of-range gaps dropped, count clamped
    p = VLMJudge._parse_split_reply('{"count": 2, "gaps": [0.0, 0.5, 1.0]}')
    assert p == {"count": 2, "gaps": [0.5]}
    # thinking-style chain prefix before the JSON
    p = VLMJudge._parse_split_reply(
        'Looking at the two views, this row holds two units. '
        '{"count": 2, "gaps": [0.5]}')
    assert p == {"count": 2, "gaps": [0.5]}
    print("PASS split reply parse (incl. think-block + range clamps)")


def test_merge_rects_multiview_union():
    """The same row outlined in three views (nadir tight, obliques
    shifted/stretched by perspective) must union into ONE rect; a
    disjoint row must never fuse into it."""
    from agentic_gts.agent.ground import _merge_rects
    # row 1 as seen by nadir / az90 / az270 (loose, shifted)
    r1 = [(-0.2, -0.6, 6.1, 0.7), (0.3, -0.8, 6.4, 0.5), (-0.4, -0.5, 5.9, 0.9)]
    # row 2: parallel, 3m away -- must stay separate
    r2 = [(-1.0, 2.4, 5.2, 3.6)]
    out = _merge_rects(r1 + r2)
    assert len(out) == 2, f"expected 2 merged rects, got {len(out)}"
    big = max(out, key=lambda r: (r[2] - r[0]) * (r[3] - r[1]))
    assert abs(big[0] - (-0.4)) < 1e-9 and abs(big[2] - 6.4) < 1e-9, \
        "union rect must span all three captures"
    assert abs(big[1] - (-0.8)) < 1e-9 and abs(big[3] - 0.9) < 1e-9
    # neighbouring rows 1.2m apart with slight VLM slop still stay split
    a = (0.0, 0.0, 6.0, 1.1)
    b = (0.0, 1.35, 6.0, 2.45)      # overlap of 0 -> never merges
    assert len(_merge_rects([a, b])) == 2
    print("PASS merge rects (3-view union, disjoint rows kept)")


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
    # legacy dict format still honoured
    lg = _parse_ground_regions(
        '{"regions": [{"x0": 10, "y0": 20, "x1": 30, "y1": 40}]}', W, H)
    assert lg == [(10.0, 20.0, 30.0, 40.0)]
    # noise / no JSON -> nothing
    assert _parse_ground_regions("just prose, no json", W, H) == []
    print("PASS parse official bbox_2d (0-1000 relative, fences, legacy)")


def test_ground_mock_returns_false():
    """Mock backend / no VLM -> grounding must fail soft, keeping hints."""
    from agentic_gts.agent import ground
    from agentic_gts.agent.judge import VLMJudge
    rng = np.random.default_rng(3)
    pts = _row_points(0.0, 6.0, rng=rng)
    scene = Scene(points=pts)
    scene.meta["yaw"] = 0.0
    hints = [_hint(3.0, 0.0)]
    scene.boxes = list(hints)
    judge = VLMJudge(backend="mock")
    assert ground.ground_stage(scene, judge) is False
    assert len(scene.boxes) == 1 and scene.boxes[0] is hints[0], \
        "hints must be kept untouched on grounding failure"
    print("PASS grounding fails soft (mock keeps hints)")


if __name__ == "__main__":
    test_unproject_ground_roundtrip()
    test_fit_region_box_full_depth()
    test_split_row_snaps_to_profile_gaps()
    test_ground_stage_with_patched_vlm()
    test_split_reply_parse()
    test_merge_rects_multiview_union()
    test_parse_ground_regions_official_format()
    test_ground_mock_returns_false()
    print("ALL GROUND TESTS PASSED")
