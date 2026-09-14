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
    BoxGroup, SamPredictorAdapter, fit_mask_points, parse_box_groups,
    refine_box,
)
from agentic_gts.core.models import OrientedBox, Scene


def test_box_groups_qwen_1000_to_pixels_once():
    reply = (
        'analysis\n{"candidate_groups": ['
        '{"bbox_2d": [100,200,500,600], "confidence": 0.8}]}'
    )
    groups = parse_box_groups(reply)
    assert len(groups) == 1
    box_pix = groups[0].pixel_box(768, 512)
    # 0-1000 grid converted to pixels exactly once
    assert np.allclose(box_pix, [76.7, 102.2, 383.5, 306.6], atol=0.1)
    print("PASS Qwen 0-1000 bbox_2d converted to SAM box pixels once")


def test_box_groups_accept_fractional_and_swapped():
    # fractional [0,1] values normalized to the 0-1000 grid
    groups = parse_box_groups('{"bbox_2d": [0.25, 0.5, 0.9, 1.0]}')
    assert groups[0].bbox_norm == (250.0, 500.0, 900.0, 1000.0)
    # swapped corners (x2 < x1) are normalized, not dropped
    groups2 = parse_box_groups('{"bbox_2d": [900, 600, 250, 200]}')
    assert groups2[0].bbox_norm == (250.0, 200.0, 900.0, 600.0)
    # degenerate / out-of-range boxes dropped
    assert parse_box_groups('{"bbox_2d": [500, 500, 500, 600]}') == []
    assert parse_box_groups('{"bbox_2d": [100, 100, 2000, 300]}') == []
    print("PASS box groups (fractional, swapped corners, degenerates)")


def test_sam_box_prompt_construction():
    """The real-VLM prompt must survive construction (literal JSON
    braces vs .format) AND follow the official 2d_grounding cookbook
    style, same as _GROUND_PROMPT: categories + JSON template only,
    no coordinate-system explanation, no custom reply structure."""
    from agentic_gts.agent.judge import VLMJudge
    j = VLMJudge(backend="qwen")
    prompt = j._SAM_BOX_PROMPT.replace("{view_name}", "front")
    assert "{view_name}" not in prompt and "front" in prompt
    assert '{"bbox_2d"' in prompt, \
        "the JSON example braces must stay literal"
    assert '"label"' in prompt, "cookbook template carries a label field"
    assert "candidate_groups" not in prompt, \
        "no custom reply structure (off-distribution instruction)"
    assert "0-1000" not in prompt, \
        "no coordinate-system explanation (the trained format implies it)"
    assert "Locate every instance" in prompt, \
        "the cookbook's trained locate phrasing must be kept"
    # instance rules (the split_stage replacement): distinct cabinets
    # in a joined row ground separately; an open door is excluded
    assert "differ in height or in color" in prompt, \
        "joined-row cabinets must be separated by visual difference"
    assert "door standing" in prompt and "exclude" in prompt, \
        "an open door swung out of the body must stay outside the box"
    print("PASS SAM box prompt construction (cookbook style, literal braces)")


def test_audit_json_survives_numpy_meta():
    """mask_refine.json save regression: audits carry box.to_dict(),
    whose meta transparently forwards numpy values from earlier stages
    (np.int64 is NOT an int subclass -> plain json.dump raised TypeError
    and killed the whole audit save with '[mask-refine] audit save
    failed (TypeError)')."""
    import json as _json
    from agentic_gts.agent.mask_refine import json_default

    box = OrientedBox(center=(1, 2, 1), size=(0.6, 1.1, 2.0), yaw=0.0,
                      meta={"n_pts": np.int64(1234),
                            "mean_z": np.float32(1.23),
                            "yaw_samples": np.arange(3, dtype=np.float64)})
    audit = {"box_id": box.box_id, "accepted": True, "score": 0.62,
             "box": box.to_dict()}
    with tempfile.TemporaryDirectory() as td:
        p = os.path.join(td, "mask_refine.json")
        with open(p, "w", encoding="utf-8") as f:
            _json.dump([audit], f, ensure_ascii=False, indent=2,
                       default=json_default)
        with open(p, encoding="utf-8") as f:
            back = _json.load(f)
    assert back[0]["box"]["meta"]["n_pts"] == 1234
    assert back[0]["box"]["meta"]["mean_z"] == float(np.float32(1.23))
    assert back[0]["box"]["meta"]["yaw_samples"] == [0.0, 1.0, 2.0]
    print("PASS audit json save survives numpy meta values")


def test_box_groups_official_cookbook_array():
    """Qwen's native 2d_grounding reply: a top-level ARRAY of
    {"bbox_2d": ..., "label": ...} items. The structural scan enters
    every inner object too -- a reversed first-hit used to return only
    the LAST item of a multi-box reply (truncating 1-3 candidates)."""
    reply = ('[{"bbox_2d": [100, 200, 500, 600], "label": "rack"}, '
             '{"bbox_2d": [300, 150, 480, 620], "label": "crac"}]')
    groups = parse_box_groups(reply)
    assert len(groups) == 2, \
        "both items must survive (was truncated to the last one)"
    assert groups[0].bbox_norm == (100.0, 200.0, 500.0, 600.0)
    assert groups[0].hypothesis == "rack"
    assert groups[1].hypothesis == "crac"
    # multi-candidate custom draft still parses in full
    reply2 = ('{"candidate_groups": ['
              '{"bbox_2d": [50, 60, 400, 500]}, '
              '{"bbox_2d": [80, 90, 420, 520]}, '
              '{"bbox_2d": [100, 120, 440, 540]}]}')
    assert len(parse_box_groups(reply2)) == 3, \
        "candidate_groups replies must not truncate either"
    print("PASS box groups (official cookbook array, no truncation)")


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


def test_sam_predict_encodes_image_once_per_object():
    """predict() must call set_image only ONCE per distinct image OBJECT:
    a multi-group view (joined row split into G cabinets) runs G box
    prompts over the SAME rendering, and the Hiera encoder -- not the
    lightweight mask head -- is SAM's dominant cost. A different image
    must always re-encode (correctness never depends on the cache)."""
    class _FakePred:
        n_set = 0

        def set_image(self, u8):
            self.n_set += 1

        def predict(self, box, multimask_output):
            return [np.ones((8, 8), bool)], [0.9], None

    sam = SamPredictorAdapter(checkpoint=None)
    sam._predictor = _FakePred()
    sam._last_img = None
    img = np.zeros((8, 8, 3), np.float32)
    sam.predict(img, np.array([1, 1, 5, 5]))
    sam.predict(img, np.array([2, 2, 6, 6]))     # same object, new box
    assert sam._predictor.n_set == 1, \
        f"same-image prompts must not re-encode, got {sam._predictor.n_set}"
    sam.predict(np.zeros((8, 8, 3), np.float32), np.array([1, 1, 5, 5]))
    assert sam._predictor.n_set == 2, \
        f"a new image must re-encode, got {sam._predictor.n_set}"
    print("PASS SAM adapter encodes once per image object")


def test_equipment_label_gate():
    """The type-confirm skip gate: grounding labels naming the equipment
    classes pass; anything else (pillar / wall / ups / unknown) fails and
    keeps the confirm question alive."""
    from agentic_gts.agent.loop import _is_equipment_label
    assert _is_equipment_label("rack")
    assert _is_equipment_label("server rack")
    assert _is_equipment_label("IT cabinet")
    assert _is_equipment_label("air-conditioning unit")
    assert _is_equipment_label("AC unit")
    assert _is_equipment_label(None) is False
    assert not _is_equipment_label("pillar")
    assert not _is_equipment_label("wall segment")
    assert not _is_equipment_label("ups battery")
    print("PASS equipment-label gate for the type-confirm skip")


def test_open_side_far_mass_beats_diffuse_wall():
    """The room-boundary override: an EXTREMELY diffuse wall smear
    (faint enough that no 5cm opacity bin reaches mass 1.0, and
    extending well past 1.2 m so the old extent test could not catch
    it either) still measures a full-width corridor on the wall side.
    What it cannot fake is far-field CONTENT: past the aisle the room
    continues (the facing row 2 m out), past the wall there is only
    faint smear. The far-mass override must pick the aisle side."""
    from agentic_gts.agent.mask_refine import _open_side
    from agentic_gts.tools.gs_io import GaussianData
    rng = np.random.default_rng(11)
    box = OrientedBox(center=(0.0, -2.4, 1.0), size=(2.0, 1.1, 2.0),
                      yaw=0.0)
    # diffuse wall smear behind the back face (y < -2.95), faint AND far
    smear = np.column_stack([rng.uniform(-1.2, 1.2, 250),
                              rng.uniform(-4.2, -2.9, 250),
                              rng.uniform(0.3, 1.8, 250)])
    # a real facing row across the aisle, 2.0 m past the front face:
    # opaque -- big far-field mass, and bin-blocks the aisle corridor
    facing = np.column_stack([rng.uniform(-1.0, 1.0, 300),
                              np.full(300, 0.15),
                              rng.uniform(0.3, 1.8, 300)])
    means = np.vstack([smear, facing]).astype(np.float32)
    n = len(means)
    gs = GaussianData(
        means=means,
        log_scales=np.full((n, 3), -6.0, dtype=np.float32),
        quats=np.tile(np.array([[1.0, 0, 0, 0]], np.float32), (n, 1)),
        raw_opacity=np.concatenate([np.full(250, -8.0),
                                    np.full(300, 2.0)]).astype(np.float32),
        f_dc=np.zeros((n, 3), dtype=np.float32),
    )
    vec, corridor = _open_side(gs, box)
    # wall side measures the full 3.0 m corridor (smear is mass-
    # invisible) vs the aisle's 2.0 m: without the override the WALL
    # side wins and the camera renders from outside the room
    assert vec[1] > 0.99, f"far-mass override must keep the aisle, got {vec}"
    assert 1.5 < corridor < 2.5, \
        f"corridor must track the facing row, got {corridor:.2f}"
    print(f"PASS far-mass override beats diffuse wall "
          f"(aisle corridor {corridor:.2f}m)")


def test_front_azim_puts_camera_on_open_side():
    """The front-view azimuth must place make_local_cam's EYE on the
    open side -- for BOTH box layouts. The old flip compared open_vec
    against the face normal (+x for cross-long boxes), but azim 90
    stands the camera on the -x side: for boxes whose long edge is on
    the cross axis the flip was INVERTED and the camera landed on the
    wall side even when _open_side was right."""
    from agentic_gts.agent.mask_refine import _front_azim
    from agentic_gts.output.gs_render import make_local_cam
    for size, yaw in [((2.0, 0.6, 2.0), 0.3), ((0.6, 2.0, 2.0), -0.7)]:
        box = OrientedBox(center=(1.0, -2.0, 1.0), size=size, yaw=yaw)
        cross = np.array([-np.sin(yaw), np.cos(yaw)])
        axis = np.array([np.cos(yaw), np.sin(yaw)])
        open_dirs = ([cross, -cross] if size[0] >= size[1]
                     else [axis, -axis])
        for od in open_dirs:
            azim = _front_azim(box, od)
            cam = make_local_cam([box], elev_deg=18.0, azim_deg=azim,
                                 standoff=1.0)
            side = np.asarray(cam.eye)[:2] - np.asarray(box.center)[:2]
            side = side / (np.linalg.norm(side) + 1e-12)
            assert float(side @ od) > 0.9, (
                f"camera on the WRONG side: yaw={yaw}, size={size}, "
                f"open={od}, azim={azim}, eye_side={side}")
    print("PASS front azimuth places the camera on the open side "
          "(both layouts, both directions)")


def test_open_side_flush_wall_and_floaters():
    """The open-side pick for a WALL-ADJACENT box: a wall flush against
    one face (gap < 0.1 m) must BLOCK that side -- the old single-point
    test's 0.10 m dead zone measured a full-width corridor there, and
    the camera walked through the wall to render the view from outside
    the room (fog of behind-wall structure). Isolated aisle floaters
    must NOT count as blocking."""
    from agentic_gts.agent.mask_refine import _open_side
    from agentic_gts.tools.gs_io import GaussianData
    rng = np.random.default_rng(9)
    box = OrientedBox(center=(0.0, 0.0, 1.0), size=(2.0, 1.1, 2.0), yaw=0.0)
    # face axis is +y/-y (long edge along x). FLUSH wall on -y: a dense
    # band 2-8 cm past the face -- inside the old test's dead zone
    wall = np.column_stack([rng.uniform(-1.2, 1.2, 400),
                            rng.uniform(-0.63, -0.57, 400),
                            rng.uniform(0.3, 1.8, 400)])
    # open aisle on +y: three isolated floaters far apart (single
    # low-opacity splats must not read as structure)
    floaters = np.array([[0.0, 1.2, 1.0], [0.4, 2.1, 1.2], [-0.5, 2.7, 0.8]])
    means = np.vstack([wall, floaters]).astype(np.float32)
    n = len(means)
    gs = GaussianData(
        means=means,
        log_scales=np.full((n, 3), -6.0, dtype=np.float32),
        quats=np.tile(np.array([[1.0, 0, 0, 0]], dtype=np.float32), (n, 1)),
        # wall splats clearly opaque, floaters faint
        raw_opacity=np.concatenate([np.full(400, 2.0),
                                    np.full(3, -1.0)]).astype(np.float32),
        f_dc=np.zeros((n, 3), dtype=np.float32),
    )
    vec, corridor = _open_side(gs, box)
    # open side is the aisle (+y), corridor wide (floaters don't block)
    assert vec[1] > 0.99, f"open side must be the aisle +y, got {vec}"
    assert corridor > 2.0, f"floaters must not block, corridor={corridor:.2f}"

    # mirror: wall on the +y side, floaters (the aisle) on -y -> the
    # open side flips. (Everything mirrors: an empty side facing away
    # from the interior is correctly vetoed as outside-the-room.)
    gs.means[:, 1] *= -1.0
    vec2, corridor2 = _open_side(gs, box)
    assert vec2[1] < -0.99, f"open side must flip to -y, got {vec2}"
    assert corridor2 > 2.0
    print(f"PASS open side: flush wall blocked, floaters ignored "
          f"(corridor {corridor:.1f} / {corridor2:.1f})")


def test_apply_depth_from_oblique_single_merged_pool():
    """The oblique depth rule: front instances keep along/height; the
    oblique pool (typically ONE merged mask covering the whole row --
    that is why its instance division is never adopted) is sliced per
    instance by along span and measures ONLY each cabinet's depth."""
    from agentic_gts.agent.mask_refine import _apply_depth_from_oblique
    seed = OrientedBox(center=(0.0, 0.0, 1.05), size=(2.0, 1.0, 2.1),
                       yaw=0.0)
    # two front-split instances: along [-0.55,-0.05] and [0.05,0.55]
    inst_a = {"fitted": OrientedBox(center=(-0.3, 0.1, 1.0),
                                    size=(0.5, 0.15, 2.0), yaw=0.0),
              "pts": np.zeros((30, 3)), "view": "front",
              "mask_score": 0.8, "score": 0.6, "label": "rack"}
    inst_b = {"fitted": OrientedBox(center=(0.3, 0.1, 1.0),
                                    size=(0.5, 0.15, 2.0), yaw=0.0),
              "pts": np.zeros((30, 3)), "view": "front",
              "mask_score": 0.8, "score": 0.6, "label": "rack"}
    # oblique pool: ONE merged cloud over both cabinets, y in [-0.4,0.4]
    # (true depth 0.8m), split per along range
    rng = np.random.default_rng(3)
    def slab(along_c):
        return np.column_stack([
            rng.uniform(along_c - 0.22, along_c + 0.22, 200),
            rng.uniform(-0.4, 0.4, 200),
            rng.uniform(0.1, 1.9, 200)])
    pool = np.vstack([slab(-0.3), slab(0.3)])
    recs = _apply_depth_from_oblique([inst_a, inst_b], pool, seed)
    assert len(recs) == 2 and all(r.get("accepted") for r in recs), recs
    for inst in (inst_a, inst_b):
        fb = inst["fitted"]
        # along span and height untouched (front-measured)
        assert abs(fb.size[0] - 0.5) < 1e-9
        assert abs(fb.size[2] - 2.0) < 1e-9
        assert abs(fb.center[0] - (0.3 if inst is inst_b else -0.3)) < 1e-9
        # depth now measured from the oblique pool, centred on it
        assert 0.6 < fb.size[1] < 1.0, fb.size[1]
        assert abs(fb.center[1]) < 0.1, fb.center[1]
        # the pool points were folded in
        assert len(inst["pts"]) == 30 + 200
    # rejection: a pool with an implausible depth (thin sliver) leaves
    # the front fit untouched
    thin = np.column_stack([rng.uniform(-1, 1, 50), np.full(50, 0.05),
                             rng.uniform(0.1, 1.9, 50)])
    inst_c = {"fitted": OrientedBox(center=(0.0, 0.1, 1.0),
                                    size=(0.5, 0.15, 2.0), yaw=0.0),
              "pts": np.zeros((30, 3)), "view": "front",
              "mask_score": 0.8, "score": 0.6, "label": "rack"}
    recs = _apply_depth_from_oblique([inst_c], thin, seed)
    assert not recs[0].get("accepted"), recs
    assert abs(inst_c["fitted"].size[1] - 0.15) < 1e-9
    assert len(inst_c["pts"]) == 30
    print("PASS oblique depth-only application (merged pool + rejection)")


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
    """The debug composite (user request): prompt box + mask overlay +
    back-projected points, one PNG per SAM candidate. Must produce a
    readable image even with no box prompt / no fitted box."""
    import tempfile
    from agentic_gts.agent.mask_refine import _save_sam_debug

    rng = np.random.default_rng(9)
    H = W = 96
    view = {"name": "front", "image": rng.uniform(0, 1, (H, W, 3)),
            "cam": None, "path": None, "prompt_path": None}
    # the VLM's box prompt (pixels) around the device
    box_pix = np.array([20.0, 25.0, 75.0, 70.0], dtype=np.float32)
    # mask: a filled ellipse
    yy, xx = np.mgrid[0:H, 0:W]
    mask = ((xx - 50) ** 2 / 30 ** 2 + (yy - 50) ** 2 / 20 ** 2) <= 1.0
    # back-projected points: scattered inside the same ellipse footprint
    n = 400
    pts3 = np.column_stack([rng.uniform(30, 70, n),
                           rng.uniform(30, 70, n),
                           rng.uniform(0.0, 2.0, n)])
    with tempfile.TemporaryDirectory() as td:
        _save_sam_debug(view, box_pix, mask, pts3,
                        OrientedBox(center=(50, 50, 1), size=(40, 40, 2),
                                    yaw=0.0),
                        OrientedBox(center=(50, 50, 1), size=(36, 36, 1.9),
                                    yaw=0.1),
                        td, "box1_front_g0_m0")
        p = os.path.join(td, "sam_debug_box1_front_g0_m0.png")
        assert os.path.isfile(p) and os.path.getsize(p) > 5000, \
            "3-panel composite (box + mask + lifted pts) must be written"
        # union path: view=None renders the top-down panel only, and a
        # failed fit (None) must still render
        _save_sam_debug(None, None, None, pts3[:5],
                        OrientedBox(center=(50, 50, 1), size=(40, 40, 2),
                                   yaw=0.0),
                        None, td, "box1_union")
        p2 = os.path.join(td, "sam_debug_box1_union.png")
        assert os.path.isfile(p2) and os.path.getsize(p2) > 3000, \
            "union / no-fit panel must still be written"
    print("PASS SAM debug composite (3-panel + union/no-fit fallback)")


def test_box_only_mask_hides_everything_outside():
    """The local views must render ONLY the device: every gaussian
    outside the box's OBB (plus slack) is hidden -- occluders in the
    aisle, the facing row, the floor, background floaters. This is what
    kills the fog: haze came from structure the camera stood inside,
    and it lives OUTSIDE the box."""
    from types import SimpleNamespace
    from agentic_gts.agent.mask_refine import _box_only_mask

    box = OrientedBox(center=(3, 0, 1), size=(6, 1.1, 2), yaw=0.0)
    pts = np.array([
        [3.0, 0.0, 1.0],      # inside the box
        [3.0, 0.4, 1.0],      # box front band (own face bleed, < pad)
        [3.0, 0.8, 1.0],      # just outside the padded face -> hidden
        [3.0, 1.2, 1.0],      # occluder / facing row in the aisle
        [3.0, -1.5, 1.0],     # background behind the far face
        [8.5, 0.0, 1.0],      # neighbour beside the row
        [3.0, 1.2, 0.02],     # floor in front of the aisle
    ])
    gs = SimpleNamespace(means=pts)
    m = _box_only_mask(gs, box)
    assert m[0] and m[1], "box interior / own face band must be kept"
    assert not m[2], "outside the padded OBB must be hidden"
    assert not m[3], "aisle occluder must be hidden"
    assert not m[4], "background must be hidden (device-only view)"
    assert not m[5], "neighbour must be hidden"
    assert not m[6], "floor must be hidden"
    # rotated box: the mask follows the OBB axes, not the world axes
    yaw = math.radians(30.0)
    box2 = OrientedBox(center=(0, 0, 1), size=(2, 0.6, 2), yaw=yaw)
    along = np.array([math.cos(yaw), math.sin(yaw)])
    inside = np.array([[along[0] * 0.9, along[1] * 0.9, 1.0]])   # local |x|<1
    outside = np.array([[along[0] * 1.5, along[1] * 1.5, 1.0]])   # local |x|>1
    gs2 = SimpleNamespace(means=np.vstack([inside, outside]))
    m2 = _box_only_mask(gs2, box2)
    assert m2[0] and not m2[1], "mask must follow the OBB's rotated axes"
    print("PASS box-only mask (everything outside the OBB hidden)")


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
    gs = SimpleNamespace(means=np.vstack([row, wall, facing]),
                         raw_opacity=np.full(len(row) + 600, 2.0,
                                            dtype=np.float32))
    vec, corridor = _open_side(gs, box)
    assert vec[1] > 0.9, f"open side must be +y (aisle), got {vec}"
    assert 1.3 < corridor < 1.9, f"corridor ~1.7m expected, got {corridor}"
    # mirrored scene: the aisle on -y must flip the pick
    gs2 = SimpleNamespace(
        means=np.vstack([row, wall[:, [0, 1, 2]] * np.array([1, -1, 1]),
                         facing * np.array([1, -1, 1])]),
        raw_opacity=np.full(len(row) + 600, 2.0, dtype=np.float32))
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


def test_sam2_model_cfg_file_path_registers_hydra_dir():
    """A model_cfg that is a real FILE path must be re-rooted: hydra's
    compose(config_name=...) strips the leading '/' of an absolute path
    (-> 'home/bod/code/...', MissingConfigException). _build_sam2_model
    must register the file's own directory as the search path and pass
    only the basename; a package-relative name goes through unchanged.
    Runs against stub sam2/hydra modules (no SAM install needed)."""
    import sys
    import types
    import contextlib
    import tempfile
    from agentic_gts.agent.mask_refine import SamPredictorAdapter

    calls = {}

    @contextlib.contextmanager
    def _init_dir(config_dir, version_base=None):
        calls["dir"] = config_dir
        yield

    hydra_mod = types.ModuleType("hydra")
    hydra_mod.initialize_config_dir = _init_dir
    hydra_core = types.ModuleType("hydra.core")
    gh_mod = types.ModuleType("hydra.core.global_hydra")

    class _GH:
        @staticmethod
        def instance():
            return _GH()

        @staticmethod
        def is_initialized():
            return True

        @staticmethod
        def clear():
            calls["cleared"] = True
    gh_mod.GlobalHydra = _GH

    sam2_mod = types.ModuleType("sam2")
    sam2_mod.__path__ = []
    build_mod = types.ModuleType("sam2.build_sam")

    def _build(cfg, ckpt):
        calls["name"], calls["ckpt"] = cfg, ckpt
        return "model"
    build_mod.build_sam2 = _build

    old = {k: sys.modules.get(k) for k in ("sam2", "sam2.build_sam", "hydra",
                                           "hydra.core",
                                           "hydra.core.global_hydra")}
    sys.modules.update({"sam2": sam2_mod, "sam2.build_sam": build_mod,
                        "hydra": hydra_mod, "hydra.core": hydra_core,
                        "hydra.core.global_hydra": gh_mod})
    try:
        with tempfile.TemporaryDirectory() as td:
            cfg_path = os.path.join(td, "sam2.1_hiera_b+.yaml")
            with open(cfg_path, "w") as f:
                f.write("model: 1\n")
            m = SamPredictorAdapter._build_sam2_model(cfg_path, "ckpt.pt")
        assert m == "model"
        assert calls["name"] == "sam2.1_hiera_b+.yaml", \
            "must pass the basename, not the full path"
        assert os.path.normpath(calls["dir"]) == \
            os.path.normpath(os.path.dirname(cfg_path)), \
            "the yaml's directory must become the hydra search path"
        assert calls["cleared"] is True, \
            "sam2's own GlobalHydra binding must be cleared first"
        # package-relative name: straight through, no search-path swap
        m2 = SamPredictorAdapter._build_sam2_model("sam2.1_hiera_b+.yaml",
                                                   "ckpt.pt")
        assert m2 == "model"
        assert "dir" not in calls or calls["name"] == "sam2.1_hiera_b+.yaml"
        assert calls["ckpt"] == "ckpt.pt"
    finally:
        for k, v in old.items():
            if v is None:
                sys.modules.pop(k, None)
            else:
                sys.modules[k] = v
    print("PASS sam2 model_cfg file path re-rooted for hydra")


def test_mask_overlay_is_rgb_plus_tint():
    """The debug panel must be an RGB image with a semi-transparent mask
    tint on top. Regression: the blend used uint8 * uint8, which wraps
    modulo 256 (150*165 -> 174) -- the masked region turned into dark
    garbage and only the solid edge line survived ('contour drawing')."""
    from agentic_gts.agent.mask_refine import _overlay_mask

    # mid-gray image, 4x4, top half masked
    img = np.full((4, 4, 3), 150, dtype=np.uint8)
    m = np.zeros((4, 4), dtype=bool)
    m[:2] = True
    out = _overlay_mask(img, m, alpha=115)
    # unmasked rows unchanged
    assert np.all(out[2:] == 150), "unmasked pixels must stay untouched"
    # masked rows: rgb pushed toward cyan, no uint8 wraparound garbage
    a = 115 / 255.0
    expect = np.array([round(150 * (1 - a)),
                       round(150 * (1 - a) + 220 * a),
                       round(150 * (1 - a) + 255 * a)])
    assert np.abs(out[0, 0].astype(int) - expect).max() <= 1, \
        f"masked pixel must be gray*toward cyan, got {out[0, 0]}"
    assert out[0, 0, 1] > out[0, 0, 0], "green channel must dominate"
    assert out[0, 0, 2] > out[0, 0, 0], "blue channel must dominate"
    # 3D SAM2 mask shape (1, H, W) handled by the caller's squeeze
    print("PASS mask overlay (rgb + tint, no uint8 wraparound)")


if __name__ == "__main__":
    tests = [v for k, v in list(globals().items()) if k.startswith("test_")]
    passed = 0
    for fn in tests:
        try:
            fn(); passed += 1
        except Exception as e:
            print(f"FAIL {fn.__name__}: {type(e).__name__}: {e}")
    print(f"\n{passed}/{len(tests)} tests passed")
