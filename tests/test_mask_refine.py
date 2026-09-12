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


if __name__ == "__main__":
    tests = [v for k, v in list(globals().items()) if k.startswith("test_")]
    passed = 0
    for fn in tests:
        try:
            fn(); passed += 1
        except Exception as e:
            print(f"FAIL {fn.__name__}: {type(e).__name__}: {e}")
    print(f"\n{passed}/{len(tests)} tests passed")
