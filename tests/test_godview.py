"""Misc regression tests: VLM reply parsing, box IO roundtrip, PLY
artifacts, split profile cuts, refit trust flags.

(The old god-view nomination / repair-loop tests were removed with the
pre-grounding pipeline: the flow is now global nadir 2D grounding ->
per-box local refine -> row split, which has its own tests in
test_ground.py / test_mask_refine.py.)
"""
from __future__ import annotations

import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np

from agentic_gts.core.models import Scene
from agentic_gts.agent.judge import _extract_json


def _scene_with_racks(n: int = 12, seed: int = 0) -> Scene:
    """3 rows x 4 racks of synthetic surface points + an aisle FP box."""
    rng = np.random.default_rng(seed)
    pts = []
    for r in range(3):
        for k in range(4):
            cx, cy = k * 0.62, r * 2.4
            for face, off in [("f", 0.55), ("b", -0.55)]:
                u = rng.uniform(cx - 0.3, cx + 0.3, 300)
                z = rng.uniform(0, 2.0, 300)
                c = np.full(300, cy + off)
                pts.append(np.stack([u, c, z], axis=1))
            u = rng.uniform(cx - 0.3, cx + 0.3, 200)
            v = rng.uniform(cy - 0.55, cy + 0.55, 200)
            z = np.full(200, 2.0)
            pts.append(np.stack([u, v, z], axis=1))
    pts.append(np.stack([rng.uniform(-1, 4, 500), rng.uniform(-1, 7, 500),
                         np.zeros(500)], axis=1))  # floor
    return Scene(points=np.vstack(pts))


def test_extract_json_variants():
    assert _extract_json('{"suspicious": []}') == {"suspicious": []}
    assert _extract_json('Here it is:\n{"suspicious": [{"index": 2, '
                          '"reason": "aisle"}]} hope it helps') is not None
    assert _extract_json("no json at all") is None
    assert _extract_json("broken { not json") is None
    print("PASS json extraction")


def test_objects_format_roundtrip():
    """boxes_objects.json must round-trip with the --boxes input schema:
    save -> load -> same center/size/yaw (within float precision)."""
    import json
    import math
    import tempfile
    import shutil
    from agentic_gts.core.models import (OrientedBox, Scene,
                                         save_boxes_as_objects)

    scene = _scene_with_racks()
    # arbitrary yaw (not axis-aligned) + a device_type-derived name
    boxes = [OrientedBox(center=(k * 0.62, 0.3 * k, 1.0),
                         size=(0.6, 1.1, 2.0), yaw=math.radians(23.5),
                         device_type="rack")
             for k in range(4)]
    out = tempfile.mkdtemp(prefix="obj_rt_")
    try:
        p = os.path.join(out, "boxes_objects.json")
        save_boxes_as_objects(boxes, p)
        data = json.load(open(p, encoding="utf-8"))
        assert "objects" in data and len(data["objects"]) == 4
        # load through the normal CLI path (Scene.load_boxes)
        s2 = Scene(points=scene.points)
        s2.load_boxes(p)
        assert len(s2.boxes) == 4
        for a, b in zip(boxes, s2.boxes):
            assert np.allclose(a.center, b.center, atol=1e-6), "center drifted"
            assert np.allclose(a.size, b.size, atol=1e-6), "size drifted"
            dyaw = (a.yaw - b.yaw + np.pi) % (2 * np.pi) - np.pi
            assert abs(dyaw) < 1e-6, f"yaw drifted {dyaw}"
        print("PASS objects-format roundtrip (center/size/yaw exact)")
    finally:
        shutil.rmtree(out, ignore_errors=True)


def _rack_pts(rng, x_lo, x_hi, n=400, y_depth=1.1, z_h=2.0):
    """Surface points of a rack footprint [x_lo, x_hi]: front/back faces +
    top. Dense enough for profile_cuts min_points."""
    pts = []
    for off in (y_depth / 2, -y_depth / 2):
        u = rng.uniform(x_lo, x_hi, n)
        z = rng.uniform(0, z_h, n)
        pts.append(np.stack([u, np.full(n, off), z], axis=1))
    u = rng.uniform(x_lo, x_hi, n // 2)
    v = rng.uniform(-y_depth / 2, y_depth / 2, n // 2)
    pts.append(np.stack([u, v, np.full(n // 2, z_h)], axis=1))
    return np.vstack(pts)


def test_profile_cuts_gap_and_tail():
    """profile_cuts must find: the empty aisle between two racks (gap cut),
    the sparse fading end of a half-observed device (tail truncation), and
    nothing in a uniform dense box."""
    from agentic_gts.core.models import OrientedBox
    from agentic_gts.tools import geometry as geo

    rng = np.random.default_rng(5)
    # two racks with a 0.1m aisle, one box over both
    pts = np.vstack([_rack_pts(rng, 0.0, 0.6), _rack_pts(rng, 0.7, 1.3)])
    scene = Scene(points=pts)
    box = OrientedBox(center=(0.65, 0, 1), size=(1.3, 1.1, 2.0), yaw=0.0)
    prof = geo.profile_cuts(scene, box)
    assert len(prof["gaps"]) == 1, f"aisle gap not found: {prof}"
    assert abs(prof["gaps"][0]) < 0.06, "cut should sit at the aisle middle"

    # one rack + a sparse fading half-device tail
    pts = np.vstack([
        _rack_pts(rng, 0.0, 0.6),
        _rack_pts(rng, 0.6, 0.9, n=30),   # ~8x sparser
    ])
    scene = Scene(points=pts)
    box = OrientedBox(center=(0.45, 0, 1), size=(0.9, 1.1, 2.0), yaw=0.0)
    prof = geo.profile_cuts(scene, box)
    assert not prof["gaps"], "fading tail must not read as an aisle gap"
    assert prof["tails"][1] is not None, "fading tail not detected"
    assert 0.1 < prof["tails"][1] < 0.25, f"tail cut {prof['tails']} misplaced"

    # uniform dense wide device: neither gaps nor tails
    scene = Scene(points=_rack_pts(rng, 0.0, 0.9, n=600))
    box = OrientedBox(center=(0.45, 0, 1), size=(0.9, 1.1, 2.0), yaw=0.0)
    prof = geo.profile_cuts(scene, box)
    assert not prof["gaps"] and prof["tails"] == (None, None)
    print("PASS profile_cuts (gap / tail / clean)")


def test_ply_artifacts():
    """Output PLYs: boxes_only.ply (no cloud) + cloud_with_boxes.ply
    (height-tinted when no GS, SH-DC colored when GS available)."""
    import tempfile
    import shutil
    from agentic_gts.core.models import OrientedBox
    from agentic_gts.output.visualize import export_boxes_ply, export_ply

    scene = _scene_with_racks()
    scene.boxes = [OrientedBox(center=(k * 0.62, 0, 1),
                               size=(0.6, 1.1, 2.0), yaw=0.0)
                   for k in range(3)]
    out = tempfile.mkdtemp(prefix="ply_art_")
    try:
        # boxes-only: small file, no cloud points
        p1 = os.path.join(out, "boxes_only.ply")
        export_boxes_ply(scene, p1)
        n1 = _ply_point_count(p1)
        # 3 boxes: wireframe only (12 edges, ~1480 pts each), no faces/cloud
        assert 3 * 1000 < n1 < 3 * 4000, f"boxes_only point count {n1} off"
        # mixed: cloud points must dominate
        p2 = os.path.join(out, "cloud_with_boxes.ply")
        export_ply(scene, p2, gs_ply=None)
        n2 = _ply_point_count(p2)
        assert n2 > len(scene.points), "cloud points missing from mixed PLY"
        assert _ply_has_colors(p2), "mixed PLY must carry per-point colors"
        # GS-colored: SH DC -> colors, count = gaussians (capped)
        gs_path = _tiny_gs_ply(out)
        export_ply(scene, p2, gs_ply=gs_path)
        n3 = _ply_point_count(p2)
        assert n3 > 50, "GS cloud should be present"
        assert _ply_has_colors(p2)
        print(f"PASS ply artifacts (boxes_only={n1}, mixed={n2}, gs={n3} pts)")
    finally:
        shutil.rmtree(out, ignore_errors=True)


def _ply_point_count(path: str) -> int:
    """Parse 'element vertex N' from the PLY header."""
    with open(path, "rb") as f:
        head = f.read(4096).decode("ascii", errors="ignore")
    import re
    m = re.search(r"element vertex (\d+)", head)
    assert m, f"no vertex count in PLY header of {path}"
    return int(m.group(1))


def _ply_has_colors(path: str) -> bool:
    with open(path, "rb") as f:
        head = f.read(4096).decode("ascii", errors="ignore")
    return ("red" in head and "green" in head and "blue" in head)


def _tiny_gs_ply(out: str) -> str:
    """Minimal binary 3DGS PLY (60 gaussians) for the coloring path."""
    import struct
    n = 60
    cols = ("property float x\nproperty float y\nproperty float z\n"
            "property float f_dc_0\nproperty float f_dc_1\n"
            "property float f_dc_2\nproperty float opacity\n"
            "property float scale_0\nproperty float scale_1\n"
            "property float scale_2\nproperty float rot_0\n"
            "property float rot_1\nproperty float rot_2\nproperty float rot_3\n")
    hdr = (f"ply\nformat binary_little_endian 1.0\n"
           f"element vertex {n}\n{cols}end_header\n")
    rng = np.random.default_rng(1)
    rows = []
    for i in range(n):
        rows.append(struct.pack(
            "<14f",
            rng.uniform(-3, 3), rng.uniform(-3, 3), rng.uniform(0, 2),
            rng.uniform(-1, 1), rng.uniform(-1, 1), rng.uniform(-1, 1),
            0.9, -3.0, -3.0, -3.0, 1.0, 0.0, 0.0, 0.0))
    p = os.path.join(out, "tiny_gs.ply")
    with open(p, "wb") as f:
        f.write(hdr.encode("ascii"))
        f.write(b"".join(rows))
    return p


def test_fit_box_to_points_keep_depth():
    """keep_depth=True must preserve the seed's cross extent even when
    the interior is hollow and only ONE face has points (the split piece's
    span is trusted knowledge, not point support); without it the
    percentile refit collapses the box back to the observed face shell."""
    from agentic_gts.tools import geometry as geo
    rng = np.random.default_rng(7)
    # ONE observed face (single-side scan): points only at y ~ +0.55
    u = rng.uniform(-0.28, 0.28, 500)
    z = rng.uniform(0.1, 1.9, 500)
    c = 0.55 + rng.uniform(-0.03, 0.03, 500)
    scene = Scene(points=np.stack([u, c, z], axis=1))
    refit = geo.fit_box_to_points(scene, (0.0, 0.0), (0.6, 1.1, 2.0), 0.0,
                                  keep_height=True, keep_depth=True)
    assert refit is not None
    assert abs(refit.size[1] - 1.1) < 0.05, \
        f"keep_depth must preserve the span, got {refit.size[1]}"
    collapsed = geo.fit_box_to_points(scene, (0.0, 0.0), (0.6, 1.1, 2.0),
                                      0.0, keep_height=True)
    assert collapsed.size[1] < 0.5, \
        f"without keep_depth the refit must collapse, got {collapsed.size[1]}"
    print(f"PASS fit_box_to_points keep_depth "
          f"(keep={refit.size[1]:.2f} collapse={collapsed.size[1]:.2f})")


if __name__ == "__main__":
    fns = [v for k, v in list(globals().items()) if k.startswith("test_")]
    passed = 0
    for fn in fns:
        try:
            fn()
            passed += 1
        except AssertionError as e:
            print(f"FAIL  {fn.__name__}: {e}")
        except Exception as e:
            print(f"ERROR {fn.__name__}: {type(e).__name__}: {e}")
    print(f"\n{passed}/{len(fns)} tests passed")
