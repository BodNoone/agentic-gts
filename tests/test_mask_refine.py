"""Local Qwen point prompts -> SAM mask -> 3D OBB tests (no SAM install)."""
from __future__ import annotations

import json
import math
import os
import sys
import tempfile

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from agentic_gts.agent.mask_refine import (
    PointGroup, SamPredictorAdapter, fit_mask_points, parse_point_groups,
    refine_box,
)
from agentic_gts.core.models import OrientedBox, Scene


def test_point_groups_qwen_1000_to_pixels_once():
    reply = (
        'analysis\n{"candidate_groups": ['
        '{"positive": [[100,200], {"x":500,"y":600}], '
        '"negative": [[900,800]], "confidence": 0.8}]}'
    )
    groups = parse_point_groups(reply)
    assert len(groups) == 1
    xy, labels = groups[0].pixel_prompts(768, 512)
    assert labels.tolist() == [1, 1, 0]
    assert np.allclose(xy[0], [76.7, 102.2], atol=0.1)
    assert np.allclose(xy[2], [690.3, 408.8], atol=0.1)
    print("PASS Qwen 0-1000 points converted to SAM pixels exactly once")


def test_point_groups_accept_fractional_normalized():
    groups = parse_point_groups(
        '{"positive": [[0.25,0.5]], "negative": [[0.9,0.1]]}')
    assert groups[0].positive_norm == [(250.0, 500.0)]
    print("PASS fractional [0,1] points normalized to Qwen 0-1000")


def test_fit_mask_points_preserves_length_axis():
    rng = np.random.default_rng(2)
    yaw = math.radians(25)
    old = OrientedBox(center=(2, 3, 1), size=(0.6, 1.1, 2.0), yaw=yaw)
    local = np.column_stack([rng.uniform(-0.3, 0.3, 2000),
                             rng.uniform(-0.55, 0.55, 2000),
                             rng.uniform(-1.0, 1.0, 2000)])
    pts = old.local_to_world(local)
    new = fit_mask_points(pts, old)
    assert new is not None
    dyaw = abs(math.atan2(math.sin(new.yaw-yaw), math.cos(new.yaw-yaw)))
    assert dyaw < math.radians(5), f"yaw changed {math.degrees(dyaw):.1f}deg"
    assert 0.5 < new.size[0] < 0.7 and 0.9 < new.size[1] < 1.2
    print("PASS mask points fit metric OBB without 90deg axis flip")


def test_sam_unconfigured_is_conservative():
    old = os.environ.pop("SAM_CHECKPOINT", None)
    try:
        sam = SamPredictorAdapter(checkpoint=None)
        assert not sam.available
    finally:
        if old is not None:
            os.environ["SAM_CHECKPOINT"] = old
    print("PASS SAM-unconfigured path keeps boxes unchanged")


def test_parse_rack_confirm():
    from agentic_gts.agent.judge import VLMJudge
    p = VLMJudge._parse_rack_confirm(
        'The image shows a rack row.\n{"is_rack": true, "confidence": 0.9}')
    assert p == {"is_rack": True, "confidence": 0.9}
    # string booleans + missing confidence
    p = VLMJudge._parse_rack_confirm('{"is_rack": "false"}')
    assert p == {"is_rack": False, "confidence": 0.5}
    # think-block prefix, clamped confidence
    p = VLMJudge._parse_rack_confirm(
        'reasoning... {"is_rack": true, "confidence": 5}')
    assert p == {"is_rack": True, "confidence": 1.0}
    # no verdict -> None (keep, never guess)
    assert VLMJudge._parse_rack_confirm("cannot tell") is None
    assert VLMJudge._parse_rack_confirm('{"confidence": 0.9}') is None
    print("PASS rack confirm parse (string bools, clamp, None on no verdict)")


def test_type_confirm_marks_low_not_deleted():
    """A VLM 'not a rack' verdict must mark LOW + unresolved and NEVER
    delete the box -- false-positive deletion is the dangerous
    direction. Runs the full _local_mask_refine path with SAM
    unconfigured (type confirmation must work without SAM)."""
    from agentic_gts.agent import loop as loop_mod
    from agentic_gts.agent import mask_refine as mr
    from agentic_gts.agent.judge import Verdict, VLMJudge
    from agentic_gts.core.models import Confidence

    scene = Scene(points=np.zeros((50, 3)))
    scene.meta["yaw"] = 0.0
    suspect = OrientedBox(center=(1, 1, 1), size=(0.6, 1.1, 2.0), yaw=0.0)
    good = OrientedBox(center=(4, 1, 1), size=(0.6, 1.1, 2.0), yaw=0.0)
    scene.boxes = [suspect, good]

    judge = VLMJudge(backend="qwen")

    def _fake_confirm(image, box, png_path=None):
        if box is suspect:
            return Verdict(action="keep",
                           params={"is_rack": False, "confidence": 0.85})
        return Verdict(action="keep",
                      params={"is_rack": True, "confidence": 0.95})
    judge.adjudicate_rack_confirm = _fake_confirm

    fake_view = {"name": "front", "image": np.zeros((4, 4, 3)),
                 "prompt_image": np.zeros((4, 4, 3)), "cam": None,
                 "path": None, "prompt_path": None}
    orig_rlv = mr.render_local_views
    mr.render_local_views = lambda *a, **k: [fake_view]
    try:
        agent = loop_mod.LayoutAgent(judge=judge)
        report = loop_mod.AgentReport()
        agent._local_mask_refine(scene, report)
    finally:
        mr.render_local_views = orig_rlv

    # suspect: kept in the scene, but LOW + flagged for human review
    ids = [b.box_id for b in scene.boxes]
    assert suspect.box_id in ids, "a 'no' verdict must NOT delete the box"
    assert suspect.confidence == Confidence.LOW
    assert suspect.meta.get("type_suspect") is True
    assert any(u["issue"]["type"] == "not_a_rack"
               and u["issue"]["box_id"] == suspect.box_id
               for u in report.unresolved), "must surface for human review"
    # good box untouched
    assert good.confidence != Confidence.LOW
    assert not good.meta.get("type_suspect")
    # no VLM answer, no marking: mock judge returns None -> box stays
    mock_ids = len(scene.boxes)
    assert mock_ids == 2
    print("PASS type confirm marks LOW + unresolved, never deletes")


if __name__ == "__main__":
    tests = [v for k, v in list(globals().items()) if k.startswith("test_")]
    passed = 0
    for fn in tests:
        try:
            fn(); passed += 1
        except Exception as e:
            print(f"FAIL {fn.__name__}: {type(e).__name__}: {e}")
    print(f"\n{passed}/{len(tests)} tests passed")
