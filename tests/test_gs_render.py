"""Tests for the 3DGS I/O + true-render plumbing (no CUDA required).

The rasterizer itself only runs on the GPU server; here we verify:
  - GS PLY detection / parse roundtrip (binary + ascii)
  - graceful degradation: render calls fall back to scatter when no
    CUDA rasterizer is installed
  - camera math sanity (box center projects near the image center)
"""
import math
import os
import sys

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from agentic_gts.tools.gs_io import (GaussianData, is_gaussian_ply,
                                     read_gaussian_ply, write_gaussian_ply)
from agentic_gts.core.models import OrientedBox


def _tiny_gs(n=8):
    rng = np.random.default_rng(0)
    return GaussianData(
        means=rng.uniform(-2, 2, (n, 3)).astype(np.float32),
        log_scales=np.log(rng.uniform(0.01, 0.05, (n, 3))).astype(np.float32),
        quats=np.tile(np.array([[1.0, 0.0, 0.0, 0.0]], dtype=np.float32), (n, 1)),
        raw_opacity=rng.uniform(2, 5, n).astype(np.float32),
        f_dc=rng.uniform(-1, 1, (n, 3)).astype(np.float32),
    )


def test_gs_roundtrip_binary(tmp_path=None):
    gs = _tiny_gs()
    path = os.path.join(str(tmp_path or os.path.dirname(__file__)),
                        "_gs_test_bin.ply")
    try:
        write_gaussian_ply(path, gs)
        assert is_gaussian_ply(path)
        out = read_gaussian_ply(path, use_cache=False)
        assert len(out) == len(gs)
        assert np.allclose(out.means, gs.means, atol=1e-6)
        assert np.allclose(out.log_scales, gs.log_scales, atol=1e-6)
        assert np.allclose(out.quats, gs.quats, atol=1e-6)
        assert np.allclose(out.raw_opacity, gs.raw_opacity, atol=1e-6)
        assert np.allclose(out.f_dc, gs.f_dc, atol=1e-6)
        print("PASS gs ply binary roundtrip")
    finally:
        if os.path.exists(path):
            os.remove(path)


def test_gs_parse_ascii():
    # hand-write a minimal ascii 3DGS ply
    lines = ["ply", "format ascii 1.0", "element vertex 2",
             "property float x", "property float y", "property float z",
             "property float f_dc_0", "property float f_dc_1",
             "property float f_dc_2", "property float opacity",
             "property float scale_0", "property float scale_1",
             "property float scale_2", "property float rot_0",
             "property float rot_1", "property float rot_2",
             "property float rot_3", "end_header",
             "0 0 0 0.1 0.2 0.3 3.0 0.01 0.02 0.03 1 0 0 0",
             "1 2 3 -0.1 -0.2 -0.3 4.0 0.05 0.05 0.05 1 0 0 0"]
    import tempfile
    with tempfile.NamedTemporaryFile("w", suffix=".ply", delete=False) as f:
        f.write("\n".join(lines) + "\n")
        path = f.name
    try:
        assert is_gaussian_ply(path)
        gs = read_gaussian_ply(path, use_cache=False)
        assert len(gs) == 2
        assert abs(gs.means[1, 0] - 1.0) < 1e-6
        assert abs(gs.raw_opacity[0] - 3.0) < 1e-6
        print("PASS gs ply ascii parse")
    finally:
        os.remove(path)


def test_render_falls_back_without_cuda():
    """On a box without gsplat/torch the render must degrade to scatter,
    not raise. (True rasterization is covered on the GPU server.)"""
    from agentic_gts.agent.judge import render_godview_png, render_topdown_image
    from agentic_gts.tools.gs_io import write_gaussian_ply
    import tempfile

    gs = _tiny_gs()
    pts = gs.means.astype(np.float64)
    box = OrientedBox(center=(0.0, 0.0, 1.0), size=(1.0, 0.6, 2.0), yaw=0.0)
    with tempfile.TemporaryDirectory() as td:
        ply = os.path.join(td, "gs.ply")
        write_gaussian_ply(ply, gs)
        png = render_godview_png(pts, [box], gs_ply=ply)
        assert png[:8] == b"\x89PNG\r\n\x1a\n"
        img = render_topdown_image(pts, [box], gs_ply=ply)
        assert img.ndim == 3 and img.shape[2] in (3, 4)
        assert img.shape[0] > 16
    print("PASS render degrades to scatter without CUDA rasterizer")


def test_camera_projection_sanity():
    from agentic_gts.output.gs_render import make_local_cam
    box = OrientedBox(center=(5.0, -3.0, 1.0), size=(1.2, 0.7, 2.0),
                      yaw=math.radians(30.0))
    cam = make_local_cam(box, extent=1.0)
    uv = cam.project_cv(np.asarray(box.center, dtype=float)[None])[0]
    # nadir camera centered on the box: center must land mid-image
    assert abs(uv[0] - cam.W / 2) < 2.0 and abs(uv[1] - cam.H / 2) < 2.0
    # all corners in front of the camera
    from agentic_gts.output.gs_render import _box_corners_3d
    cs = _box_corners_3d(box)
    pc = np.hstack([cs, np.ones((len(cs), 1))]) @ cam.view_cv().T
    assert np.all(pc[:, 2] > 0)
    print("PASS camera projection sanity")


def test_godview_nadir_camera():
    """True top-down godview: frames the whole footprint, no flip, and the
    on-screen axes are axis-aligned with world x/y (no mirroring)."""
    from agentic_gts.output.gs_render import make_godview_cam
    W, H = 640, 480
    pts = np.array([[0, 0, 0], [8, 0, 0], [8, 6, 0], [0, 6, 0],
                    [0, 0, 1.2], [8, 0, 1.2], [8, 6, 1.2], [0, 6, 1.2]])
    cam = make_godview_cam(pts, nadir=True, cam_z=3.0, W=W, H=H)
    assert cam.eye[2] > 3.0          # camera raised to frame the room
    assert cam.up[2] == 0            # screen up is horizontal (not +z)
    corners = np.array([[0, 0, 1.2], [8, 0, 1.2], [8, 6, 1.2], [0, 6, 1.2]])
    uv = cam.project_cv(corners)
    in_frame = (uv[:, 0].min() > 0.03 * W and uv[:, 0].max() < 0.97 * W and
                uv[:, 1].min() > 0.03 * H and uv[:, 1].max() < 0.97 * H)
    assert in_frame, f"footprint not framed: uv={uv}"
    # world +x stays on a horizontal screen line, +y on a vertical line
    uvx = cam.project_cv(np.array([[2, 3, 1.2], [6, 3, 1.2]]))
    uvy = cam.project_cv(np.array([[4, 1, 1.2], [4, 5, 1.2]]))
    assert abs(uvx[0, 1] - uvx[1, 1]) < 1.0      # horizontal
    assert abs(uvy[0, 0] - uvy[1, 0]) < 1.0      # vertical
    print("PASS godview nadir camera (frame + orientation)")


def test_prep_cuts_ceiling():
    """Gaussians above cut_z must be excluded from the render input so the
    ceiling cannot occlude the racks in a top-down view."""
    from agentic_gts.output.gs_render import _prep
    gs = _tiny_gs(50)
    gs.means[:, 2][:25] = 4.0        # 25 gaussians at ceiling height
    gs.means[:, 2][25:] = 1.0        # 25 at rack height
    cut_z = 2.6                       # racks ~1.0m, ceiling ~4.0m
    means, _, _, _, _ = _prep(gs, cut_z)
    assert len(means) == 25, f"expected 25 kept (racks), got {len(means)}"
    assert np.all(means[:, 2] < cut_z)
    print("PASS prep cuts ceiling gaussians at cut_z")


if __name__ == "__main__":
    test_gs_roundtrip_binary()
    test_gs_parse_ascii()
    test_render_falls_back_without_cuda()
    test_camera_projection_sanity()
    test_godview_nadir_camera()
    test_prep_cuts_ceiling()
    print("ALL GS TESTS PASSED")
