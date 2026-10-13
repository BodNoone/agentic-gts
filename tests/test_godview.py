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
def test_output_format():
    """Name-keyed output preserves geometry, rotations and confidence."""
    import json
    import math
    import tempfile
    import shutil
    from agentic_gts.core.models import (OrientedBox, Scene,
                                          save_boxes_output)
    scene = _scene_with_racks()
    # arbitrary yaw (not axis-aligned) + a device_type-derived name
    boxes = [OrientedBox(center=(k * 0.62, 0.3 * k, 1.0),
                         size=(0.6, 1.1, 2.0), yaw=math.radians(23.5),
                         device_type="rack")
             for k in range(4)]
    out = tempfile.mkdtemp(prefix="obj_rt_")
    try:
        p = os.path.join(out, "output.json")
        save_boxes_output(boxes, p)
        data = json.load(open(p, encoding="utf-8"))
        assert list(data) == ["rack", "rack_2", "rack_3", "rack_4"]
        for box, entry in zip(boxes, data.values()):
            assert set(entry) == {"center", "extent", "rotation", "confidence"}
            assert np.allclose(entry["center"], box.center)
            assert np.allclose(entry["extent"], box.size)
            assert np.allclose(entry["rotation"], box.rotation)
            assert entry["confidence"] == box.confidence.value
    finally:
        shutil.rmtree(out, ignore_errors=True)
def test_output_names_preserve_custom_names_and_avoid_collisions(tmp_path):
    import json
    from agentic_gts.core.models import (Confidence, DeviceType, OrientedBox,
                                        save_boxes_output)

    boxes = [OrientedBox(center=(0, 0, 1), size=(1, 1, 2),
                         device_type=DeviceType.RACK,
                         confidence=Confidence.LOW),
             OrientedBox(center=(1, 0, 1), size=(1, 1, 2)),
             OrientedBox(center=(2, 0, 1), size=(1, 1, 2),
                         meta={"object_name": "rack_2"}),
             OrientedBox(center=(3, 0, 1), size=(1, 1, 2),
                         device_type=DeviceType.RACK)]
    path = tmp_path / "output.json"
    save_boxes_output(boxes, str(path))
    data = json.loads(path.read_text(encoding="utf-8"))
    assert list(data) == ["rack", "unknown", "rack_2", "rack_3"]
    assert data["rack"]["confidence"] == "low"
    save_boxes_output([], str(path))
    assert json.loads(path.read_text(encoding="utf-8")) == {}


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
