"""End-to-end and unit tests. Run: python -m pytest tests/ -q"""
from __future__ import annotations
import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import numpy as np
from agentic_gts.core.models import OrientedBox, Scene
from agentic_gts.synth.generator import SynthConfig, generate
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
def test_pipeline_local_empty_reply_smoke(monkeypatch, tmp_path):
    """No-hint flow smoke test without loading a model or using a network."""
    from agentic_gts.agent.judge import VLMJudge
    monkeypatch.setattr(VLMJudge, "_local_image_call",
                        lambda self, png, prompt, max_new_tokens=64: "")
    scene, gt, corrupt = generate(SynthConfig(seed=42))
    run_pipeline(scene, gt_boxes=gt,
                 vlm_backend="local", out_dir=str(tmp_path))
    assert len(scene.boxes) > 0, \
        "the pipeline should retain geometry-bootstrap boxes when VLM is empty"
    assert (tmp_path / "grounded.png").is_file()
    print(f"PASS pipeline local empty-reply smoke ({len(scene.boxes)} boxes out)")
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
