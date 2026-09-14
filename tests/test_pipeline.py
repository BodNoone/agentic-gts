"""End-to-end and unit tests. Run: python -m pytest tests/ -q  (or python tests/test_pipeline.py)"""
from __future__ import annotations

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np

from agentic_gts.core.models import OrientedBox, Scene
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


def test_split_box():
    scene, gt, corrupt = generate(SynthConfig(seed=42))
    merged = [b for b in corrupt if b.meta.get("corruption") == "merged"][0]
    subs = geo.split_box(scene, merged, 2, width_unit=0.6)
    assert len(subs) == 2
    assert abs(subs[0].size[0] - 0.6) < 0.05


def test_pipeline_mock_smoke():
    """New-flow smoke test: run_pipeline on synth data with the mock VLM
    must complete without raising (the old 'improves layout with mock'
    assertions belonged to the removed rule-repair loop; with mock the
    agent stages no-op, only the deterministic rules run)."""
    scene, gt, corrupt = generate(SynthConfig(seed=42))
    scene.boxes = corrupt
    run_pipeline(scene, gt_boxes=gt, use_coarse_seg=False,
                 vlm_backend="mock", out_dir="runs/test_tmp")
    assert len(scene.boxes) > 0, "pipeline dropped every box"
    print(f"PASS pipeline mock smoke ({len(scene.boxes)} boxes out)")


def test_eval_edge_error():
    gt = [OrientedBox(center=(0, 0, 1), size=(0.6, 1.1, 2.0), yaw=0.0)]
    ok = [OrientedBox(center=(0.01, 0, 1), size=(0.6, 1.1, 2.0), yaw=0.0)]
    bad = [OrientedBox(center=(0.2, 0, 1), size=(0.6, 1.1, 2.0), yaw=0.0)]
    r_ok = evaluate(ok, gt, edge_threshold_m=0.05)
    r_bad = evaluate(bad, gt, edge_threshold_m=0.05, match_iou=0.1)
    assert r_ok.edge_accuracy == 1.0
    assert r_bad.edge_accuracy < r_ok.edge_accuracy


def test_yaw_estimation_rotated_scene():
    import math
    from agentic_gts.segment.orientation import estimate_yaw
    for deg in (0, 15, 30, 60):
        scene, _, _ = generate(SynthConfig(seed=42, room_yaw_deg=deg))
        est = estimate_yaw(scene.points)
        true = math.remainder(math.radians(deg), math.pi / 2)
        if true >= math.pi / 4:
            true -= math.pi / 2
        err = abs(math.degrees(true - est))
        err = min(err, 90 - err)
        assert err < 3.0, f"yaw error {err:.1f}deg for input {deg}deg"


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
