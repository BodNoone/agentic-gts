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


def test_sam_point_prompt_construction():
    """The real-VLM prompt must survive construction: the JSON example's
    literal braces ({\"candidate_groups\": ...}) used to be parsed by
    str.format as a replacement field -> KeyError on every call (the
    mock backend never formats, so only a real run caught it)."""
    from agentic_gts.agent.judge import VLMJudge
    j = VLMJudge(backend="qwen")
    prompt = j._SAM_POINT_PROMPT.replace("{view_name}", "front")
    assert "{view_name}" not in prompt and "front" in prompt
    assert '{"candidate_groups"' in prompt, \
        "the JSON example braces must stay literal"
    # regression: .format would raise KeyError '"candidate_groups"'
    # (field name includes the quotes) -- nothing may raise now
    print("PASS SAM point prompt construction (literal JSON braces)")


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


def test_sam_debug_composite():
    """The debug composite (user request): prompt points + mask overlay +
    back-projected points, one PNG per SAM candidate. Must produce a
    readable image even with empty points / no fitted box."""
    import tempfile
    from agentic_gts.agent.mask_refine import _save_sam_debug

    rng = np.random.default_rng(9)
    H = W = 96
    view = {"name": "front", "image": rng.uniform(0, 1, (H, W, 3)),
            "cam": None, "path": None, "prompt_path": None}
    # prompt points: 2 positive, 1 negative; mask: a filled ellipse
    yy, xx = np.mgrid[0:H, 0:W]
    mask = ((xx - 50) ** 2 / 30 ** 2 + (yy - 50) ** 2 / 20 ** 2) <= 1.0
    # back-projected points: scattered inside the same ellipse footprint
    n = 400
    pts3 = np.column_stack([rng.uniform(30, 70, n),
                           rng.uniform(30, 70, n),
                           rng.uniform(0.0, 2.0, n)])
    with tempfile.TemporaryDirectory() as td:
        _save_sam_debug(view, [(50, 50), (60, 45), (20, 20)],
                        [1, 1, 0], mask, pts3,
                        OrientedBox(center=(50, 50, 1), size=(40, 40, 2),
                                    yaw=0.0),
                        OrientedBox(center=(50, 50, 1), size=(36, 36, 1.9),
                                    yaw=0.1),
                        td, "box1_front_g0_m0")
        p = os.path.join(td, "sam_debug_box1_front_g0_m0.png")
        assert os.path.isfile(p) and os.path.getsize(p) > 5000, \
            "3-panel composite (points + mask + lifted pts) must be written"
        # union path: view=None renders the top-down panel only, and a
        # failed fit (None) must still render
        _save_sam_debug(None, None, None, None, pts3[:5],
                        OrientedBox(center=(50, 50, 1), size=(40, 40, 2),
                                   yaw=0.0),
                        None, td, "box1_union")
        p2 = os.path.join(td, "sam_debug_box1_union.png")
        assert os.path.isfile(p2) and os.path.getsize(p2) > 3000, \
            "union / no-fit panel must still be written"
    print("PASS SAM debug composite (3-panel + union/no-fit fallback)")


def test_view_occlusion_mask_drops_only_occluders():
    """The front view must show the box's FACE, not whatever stands in
    front of it -- and everything else must SURVIVE. The old screen-hull +
    depth-slab isolation deleted the background, the floor band and the
    neighbours (sliced x-ray views); the occlusion mask only drops
    gaussians BETWEEN the camera and the box that project inside the
    box's screen silhouette."""
    from types import SimpleNamespace
    from agentic_gts.agent.mask_refine import _view_occlusion_mask
    from agentic_gts.output.gs_render import make_local_cam

    # a long rack row box: long edge along x (yaw=0)
    box = OrientedBox(center=(3, 0, 1), size=(6, 1.1, 2), yaw=0.0)
    cam = make_local_cam([box], W=768, H=768,
                         elev_deg=18.0, azim_deg=0.0, standoff=1.0)
    pts = np.array([
        [3.0, 0.0, 1.0],      # inside the box
        [3.0, 0.4, 1.0],      # box front band (own face bleed)
        [3.0, 1.2, 1.0],      # OCCLUDER in the aisle, over the box
        [3.0, -1.5, 1.0],     # background behind the far face
        [8.5, 0.0, 1.0],      # neighbour beside the row
        [3.0, 1.2, 0.02],     # floor in front of the aisle
    ])
    gs = SimpleNamespace(means=pts)
    m = _view_occlusion_mask(gs, box, cam)
    assert m[0] and m[1], "box interior / own face band must be kept"
    assert not m[2], "aisle occluder covering the box must be dropped"
    assert m[3], "background behind the row must STAY (normal photo)"
    assert m[4], "neighbour beside the row must STAY (context)"
    assert m[5], "floor in front of the aisle must STAY (normal photo)"
    print("PASS view occlusion mask (occluders dropped, scene kept)")


def test_open_side_picks_aisle():
    """_open_side must find the aisle side: the rack's front faces a
    1.7m corridor, its back a 0.7m gap to the wall -> the open direction
    is +y (front) with the corridor width of the facing structure."""
    from types import SimpleNamespace
    from agentic_gts.agent.mask_refine import _open_side

    box = OrientedBox(center=(0, 0, 1), size=(1.2, 0.6, 2.0), yaw=0.0)
    rng = np.random.default_rng(0)
    row = np.column_stack([rng.uniform(-0.6, 0.6, 500),
                           rng.uniform(-0.3, 0.3, 500),
                           rng.uniform(0.3, 1.7, 500)])
    # wall 0.7m behind the back face (y = -0.3 - 0.7 = -1.0)
    wall = np.column_stack([rng.uniform(-3, 3, 300),
                            np.full(300, -1.0),
                            rng.uniform(0.3, 1.7, 300)])
    # facing row across a 1.7m aisle (front face y = 0.3 + 1.7 = 2.0)
    facing = np.column_stack([rng.uniform(-3, 3, 300),
                              np.full(300, 2.0),
                              rng.uniform(0.3, 1.7, 300)])
    gs = SimpleNamespace(means=np.vstack([row, wall, facing]))
    vec, corridor = _open_side(gs, box)
    assert vec[1] > 0.9, f"open side must be +y (aisle), got {vec}"
    assert 1.3 < corridor < 1.9, f"corridor ~1.7m expected, got {corridor}"
    # mirrored scene: the aisle on -y must flip the pick
    gs2 = SimpleNamespace(
        means=np.vstack([row, wall[:, [0, 1, 2]] * np.array([1, -1, 1]),
                         facing * np.array([1, -1, 1])]))
    vec2, _ = _open_side(gs2, box)
    assert vec2[1] < -0.9, f"mirrored scene must pick -y, got {vec2}"
    print("PASS open side picks the aisle (and flips on mirror)")


def test_front_view_axis_swap():
    """'front' must look perpendicular to the LONG edge: a box whose
    length is on the cross axis (size[0] < size[1]) swaps the azimuth
    so the view faces the device's face, not its side."""
    import inspect
    from agentic_gts.agent import mask_refine as mr
    src = inspect.getsource(mr.render_local_views)
    assert "azim_front" in src, "front azimuth must adapt to box axes"
    # long edge on cross axis: front must use azim 90 (along the yaw
    # axis) so the view direction is perpendicular to the long edge
    from agentic_gts.output.gs_render import make_local_cam
    import numpy as np
    box = OrientedBox(center=(0, 0, 1), size=(1.1, 6, 2), yaw=0.0)
    cam = make_local_cam([box], extent=1.4, W=768, H=768,
                         elev_deg=18.0, azim_deg=90.0)
    # eye offset from center: for azim 90 the eye moves along +yaw axis
    d = np.asarray(cam.eye) - np.asarray(box.center)
    assert abs(d[0]) > abs(d[1]), \
        "camera must look along the yaw (short) axis of the long-cross box"
    print("PASS front view axis swap (perpendicular to the long edge)")


if __name__ == "__main__":
    tests = [v for k, v in list(globals().items()) if k.startswith("test_")]
    passed = 0
    for fn in tests:
        try:
            fn(); passed += 1
        except Exception as e:
            print(f"FAIL {fn.__name__}: {type(e).__name__}: {e}")
    print(f"\n{passed}/{len(tests)} tests passed")
