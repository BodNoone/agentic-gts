"""End-to-end and unit tests. Run: python -m pytest tests/ -q  (or python tests/test_pipeline.py)"""
from __future__ import annotations

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np

from agentic_gts.core.models import OrientedBox, DeviceType, Scene
from agentic_gts.synth.generator import SynthConfig, generate
from agentic_gts.tools import geometry as geo
from agentic_gts.pipeline import run_pipeline
from agentic_gts.eval.metrics import evaluate


def test_oriented_box_basic():
    b = OrientedBox(center=(1, 2, 1), size=(0.6, 1.1, 2.0), yaw=0.0)
    pts = np.array([[1.0, 2.0, 1.0], [5.0, 5.0, 5.0]])
    inside = b.contains(pts)
    assert inside[0] and not inside[1]
    corners = b.corners_2d()
    assert corners.shape == (4, 2)


def test_iou_identity():
    b = OrientedBox(center=(0, 0, 1), size=(0.6, 1.1, 2.0), yaw=0.0)
    assert b.iou_2d(b) > 0.9


def test_synth_generation():
    scene, gt, corrupt = generate(SynthConfig(seed=1))
    assert len(scene.points) > 10000
    assert len(gt) >= 10
    assert len(corrupt) >= 5
    # some corruption must exist
    kinds = {b.meta.get("corruption") for b in corrupt}
    assert len(kinds) > 1


def test_center_field_detects_merged():
    scene, gt, corrupt = generate(SynthConfig(seed=42))
    scene.boxes = corrupt
    merged = [b for b in corrupt if b.meta.get("corruption") == "merged"]
    assert merged, "seed 42 should contain merged boxes"
    for b in merged:
        est, _, _ = geo.center_field_clusters(scene, b)
        assert est >= 2, f"merged box should be detected as >=2 racks, got {est}"
    clean = [b for b in corrupt if b.meta.get("corruption") is None
             and b.device_type == DeviceType.RACK]
    for b in clean[:5]:
        est, _, _ = geo.center_field_clusters(scene, b)
        assert est == 1, f"clean rack detected as {est} racks"


def test_split_box():
    scene, gt, corrupt = generate(SynthConfig(seed=42))
    merged = [b for b in corrupt if b.meta.get("corruption") == "merged"][0]
    subs = geo.split_box(scene, merged, 2, width_unit=0.6)
    assert len(subs) == 2
    assert abs(subs[0].size[0] - 0.6) < 0.05


def test_pipeline_improves_layout():
    scene, gt, corrupt = generate(SynthConfig(seed=42))
    scene.boxes = corrupt
    before = evaluate(corrupt, gt, edge_threshold_m=0.05)
    res = run_pipeline(scene, gt_boxes=gt, use_coarse_seg=False,
                       vlm_backend="mock", out_dir="runs/test_tmp")
    after = evaluate(scene.boxes, gt, edge_threshold_m=0.05)
    assert after.recall >= before.recall
    assert after.edge_accuracy > before.edge_accuracy
    assert after.edge_accuracy > 0.85
    assert after.recall > 0.9


def test_eval_edge_error():
    gt = [OrientedBox(center=(0, 0, 1), size=(0.6, 1.1, 2.0), yaw=0.0)]
    ok = [OrientedBox(center=(0.01, 0, 1), size=(0.6, 1.1, 2.0), yaw=0.0)]
    bad = [OrientedBox(center=(0.2, 0, 1), size=(0.6, 1.1, 2.0), yaw=0.0)]
    r_ok = evaluate(ok, gt, edge_threshold_m=0.05)
    r_bad = evaluate(bad, gt, edge_threshold_m=0.05, match_iou=0.1)
    assert r_ok.edge_accuracy == 1.0
    assert r_bad.edge_accuracy < r_ok.edge_accuracy


if __name__ == "__main__":
    fns = [v for k, v in list(globals().items()) if k.startswith("test_")]
    passed = 0
    for fn in fns:
        try:
            fn()
            print(f"PASS  {fn.__name__}")
            passed += 1
        except AssertionError as e:
            print(f"FAIL  {fn.__name__}: {e}")
        except Exception as e:
            print(f"ERROR {fn.__name__}: {type(e).__name__}: {e}")
    print(f"\n{passed}/{len(fns)} tests passed")
