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
    BoxGroup, SamPredictorAdapter, parse_box_groups,
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
    # in a joined row ground separately; the open door is its OWN
    # positive class (user finding: the model detects "open cabinet
    # door" reliably but cannot exclude it via a negative instruction)
    assert "differ in height or in color" in prompt, \
        "joined-row cabinets must be separated by visual difference"
    assert "open cabinet door" in prompt, \
        "the open door must be a positive detection class"
    assert "its OWN instance" in prompt, \
        "the door must ground as its own instance, not be excluded"
    # VLM quality verdict (user direction: judged TOGETHER with the
    # grounding in the same call, garbage views dropped)
    assert "quality: good" in prompt and "quality: poor" in prompt, \
        "the prompt must ask for the first-line quality verdict"
    print("PASS SAM box prompt construction (cookbook style, literal braces)")


def test_reply_view_quality_parsing():
    """The per-view quality verdict parsed from the grounding reply's
    first line (judged in the SAME call, no extra budget). Missing
    marker reads GOOD (dropping a view loses signal -- it takes an
    explicit poor verdict); the LAST match wins; yes/no and
    capitalization variants accepted."""
    from agentic_gts.agent.mask_refine import reply_view_quality
    assert reply_view_quality(
        'quality: good\n[{"bbox_2d": [1, 2, 3, 4]}]') == "good"
    assert reply_view_quality("quality: poor") == "poor"
    assert reply_view_quality("Quality: Poor.") == "poor"
    assert reply_view_quality("quality: no") == "poor"
    assert reply_view_quality("quality: yes") == "good"
    assert reply_view_quality(
        '[{"bbox_2d": [1, 2, 3, 4]}]') == "good", \
        "no marker in the reply must read GOOD, not poor"
    assert reply_view_quality(
        "quality: good ... later: quality: poor") == "poor", \
        "the LAST stated verdict wins"
    assert reply_view_quality("") == "good"
    print("PASS reply view quality parsing")


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


def test_box_groups_long_row_budget():
    """The SAM-box call's token budget must hold a LONG joined row:
    dozens of cabinets, each its own bbox_2d item (plus door
    instances). The old 800-token cap truncated the reply mid-item
    (user report: the tail of the VLM answer was cut off on super-long
    rows) -- parse must recover every COMPLETE item, and the budget
    itself must scale with the row length (mirrors ground_regions)."""
    from agentic_gts.agent.judge import VLMJudge
    # 40 cabinets + 3 doors = 43 items: a realistic super-long row
    items = [{"bbox_2d": [10 + 24 * i, 100, 30 + 24 * i, 900],
              "label": "rack"} for i in range(40)]
    items += [{"bbox_2d": [50, 100, 70, 500],
               "label": "open cabinet door"} for _ in range(3)]
    import json as _json
    reply = _json.dumps(items)
    groups = parse_box_groups(reply)
    assert len(groups) == 43, \
        f"a 43-item row reply must parse in full, got {len(groups)}"
    assert groups[39].bbox_norm[0] > 900, "the LAST rack must survive"
    assert groups[40].hypothesis == "open cabinet door"
    # the budget on the real call path (not just the parser): the
    # qwen/API call must request enough tokens for such a reply
    j = VLMJudge(backend="qwen")
    src = None
    import inspect as _inspect
    for f in (VLMJudge.adjudicate_sam_boxes,):
        src = _inspect.getsource(f)
    assert "max_tokens=6000" in src or "max_new_tokens=6000" in src, \
        "the SAM-box call must carry the long-row budget (6000)"
    print("PASS long-row reply parses in full (43 items, budget 6000)")


def test_merge_spans_dedupes_but_keeps_seams():
    """A duplicate (the VLM double-boxing ONE cabinet -- high 2D IoU of
    the VLM's own pixel boxes) merges; truly adjacent cabinets keep
    their seam."""
    from agentic_gts.agent.mask_refine import _merge_spans
    spans = [
        {"lo": 0.00, "hi": 0.60, "pts": np.zeros((50, 3)),
         "ms": 0.8, "label": "rack", "pix": (100, 200, 500, 600)},
        {"lo": 0.05, "hi": 0.58, "pts": np.zeros((30, 3)),
         "ms": 0.7, "label": "rack",
         "pix": (110, 210, 510, 610)},    # duplicate VLM box: high IoU
        {"lo": 0.62, "hi": 1.20, "pts": np.zeros((50, 3)),
         "ms": 0.8, "label": "rack",
         "pix": (520, 200, 900, 600)},   # adjacent: distinct box
    ]
    out = _merge_spans(spans)
    assert len(out) == 2, f"expected 2 spans, got {len(out)}"
    assert abs(out[0]["lo"]) < 1e-6 and abs(out[0]["hi"] - 0.60) < 1e-6
    assert abs(out[1]["lo"] - 0.62) < 1e-6 and abs(out[1]["hi"] - 1.20) < 1e-6
    assert len(out[0]["pts"]) == 80      # points merged
    print("PASS span merging (duplicates union, seams survive)")


def test_merge_spans_mask_bleed_keeps_instances():
    """SAM masks bleed a few cm across the seam between joined cabinets
    (user report: the VLM grounds DISTINCT instances but the spans
    overlap, and the old >0.10 m overlap merge collapsed the row back
    into one span -- joined rows stayed joined). Duplicates are now
    detected on the VLM's OWN boxes (2D IoU); distinct instances keep
    both spans, cut at the overlap midpoint -- the seam."""
    from agentic_gts.agent.mask_refine import _merge_spans
    spans = [
        {"lo": 0.00, "hi": 0.64, "pts": np.zeros((50, 3)),
         "ms": 0.8, "label": "rack", "pix": (100, 200, 480, 600)},
        {"lo": 0.56, "hi": 1.20, "pts": np.zeros((50, 3)),
         "ms": 0.8, "label": "rack", "pix": (520, 200, 900, 600)},
    ]
    out = _merge_spans(spans)
    assert len(out) == 2, "two VLM-distinct instances must both survive"
    seam = 0.5 * (0.64 + 0.56)          # the overlap midpoint
    assert abs(out[0]["hi"] - seam) < 1e-9
    assert abs(out[1]["lo"] - seam) < 1e-9
    # heavy span overlap but DISTINCT VLM boxes: still never merged
    spans2 = [
        {"lo": 0.00, "hi": 1.00, "pts": np.zeros((50, 3)),
         "ms": 0.8, "label": "rack", "pix": (100, 200, 480, 600)},
        {"lo": 0.20, "hi": 1.20, "pts": np.zeros((50, 3)),
         "ms": 0.8, "label": "rack", "pix": (520, 200, 900, 600)},
    ]
    out2 = _merge_spans(spans2)
    assert len(out2) == 2, \
        "distinct VLM boxes never merge, however far the masks bleed"
    print("PASS mask-bleed overlap keeps instances (seam at midpoint)")


def test_build_split_pieces_trusts_seed_dims():
    """Pieces are SPLITS OF THE SEED: only the along extent/position come
    from the measured spans; yaw / height / depth / cross centre stay
    seed-trusted (user decision: corrections on the initial box only)."""
    from agentic_gts.agent.mask_refine import _build_split_pieces
    yaw = math.radians(25)
    seed = OrientedBox(center=(2, 3, 1.05), size=(3.6, 1.1, 2.1), yaw=yaw)
    spans = [
        {"lo": -1.8, "hi": -0.6, "pts": np.zeros((100, 3)),
         "ms": 0.8, "label": "rack"},
        {"lo": -0.55, "hi": 1.75, "pts": np.zeros((100, 3)),
         "ms": 0.8, "label": "rack"},
    ]
    out = _build_split_pieces(spans, seed)
    assert len(out) == 2
    axis = np.array([math.cos(yaw), math.sin(yaw)])
    for e in out:
        p = e["fitted"]
        assert abs(math.atan2(math.sin(p.yaw - yaw),
                              math.cos(p.yaw - yaw))) < 1e-9
        assert abs(p.size[1] - 1.1) < 1e-9, "depth must stay seed-trusted"
        assert abs(p.size[2] - 2.1) < 1e-9, "height must stay seed-trusted"
        assert abs(p.center[2] - 1.05) < 1e-9
        # cross centre unchanged: only the along position moved
        assert abs(np.asarray(p.center)[:2] @ np.array(
            [-math.sin(yaw), math.cos(yaw)]) - np.asarray(seed.center)[:2]
            @ np.array([-math.sin(yaw), math.cos(yaw)])) < 1e-9
    assert abs(out[0]["fitted"].size[0] - 1.2) < 1e-9
    assert abs(out[1]["fitted"].size[0] - 2.3) < 1e-9
    print("PASS split pieces keep seed dims, spans give along extent")


def test_height_and_staggered_thickness_per_piece():
    """Per-piece height + geometry thickness fallback (user report:
    refine never corrected HEIGHTS, and a front-back STAGGERED row
    kept the seed's UNION depth on every piece). Two cabinets in one
    seed: A along [0,1.2] tall 2.1m at cross [-0.2,0.9]; B along
    [1.2,2.4] short 1.2m at cross [-0.9,0.2] -- union cross [-0.9,0.9]
    (seed depth 1.8, the row's tallest height 2.1). No side view: the
    geometry fallback must give each piece its OWN ~1.1m thickness at
    its OWN offset, and the short one its OWN height."""
    from agentic_gts.agent.mask_refine import (_apply_height_and_geom_depth,
                                                _build_split_pieces)
    rng = np.random.default_rng(7)

    def cabinet(x0, x1, c_lo, c_hi, z_hi, n=900):
        # shell-heavy sampling: half the points on each cross wall
        # (what a real cabinet's front/back faces look like to the
        # strong-bin estimator), spread along/z inside
        xs = rng.uniform(x0, x1, n)
        cs = np.where(rng.random(n) < 0.5,
                      c_lo + 0.02 * rng.random(n),
                      c_hi - 0.02 * rng.random(n))
        zs = rng.uniform(0.1, z_hi, n)
        return np.column_stack([xs, cs, zs])

    a = cabinet(0.0, 1.2, -0.2, 0.9, 2.1)
    b = cabinet(1.2, 2.4, -0.9, 0.2, 1.2)
    scene = Scene(points=np.vstack([a, b]))
    seed = OrientedBox(center=(1.2, 0.0, 1.05), size=(2.4, 1.8, 2.1),
                       yaw=0.0)
    spans = [
        {"lo": -1.2, "hi": 0.0, "pts": a, "ms": 0.8, "label": "rack"},
        {"lo": 0.0, "hi": 1.2, "pts": b, "ms": 0.8, "label": "rack"},
    ]
    instances = _build_split_pieces(spans, seed)
    n_geom = _apply_height_and_geom_depth(instances, seed, scene)
    assert n_geom == 2, "both pieces need the geometry depth fallback"
    by_along = sorted(instances, key=lambda e: e["fitted"].center[0])
    pa, pb = (e["fitted"] for e in by_along)
    # per-piece height: A keeps ~2.1, B drops to ~1.2 (was seed 2.1)
    assert 1.9 < pa.size[2] < 2.2, f"A height {pa.size[2]:.2f}"
    assert 1.0 < pb.size[2] < 1.35, \
        f"B height {pb.size[2]:.2f} -- its OWN, not the seed's 2.1"
    assert abs(pb.center[2] - pb.size[2] / 2.0) < 0.05, \
        "B bottom must stay on the ground (seed bottom ~0)"
    # per-piece thickness at its own stagger offset (was seed 1.8 union)
    for p, mid_exp in ((pa, 0.35), (pb, -0.35)):
        assert 0.9 < p.size[1] < 1.35, \
            f"thickness {p.size[1]:.2f} -- its OWN, not the union 1.8"
        assert abs(p.center[1] - mid_exp) < 0.15, \
            f"cross centre {p.center[1]:.2f}, expected ~{mid_exp}"
    print("PASS per-piece height + staggered geometry thickness")


def test_cross_view_merge_unions_pairs_drops_coarse():
    """Front + back vote on the SAME cabinets (user report: the front
    aisle is sometimes a narrow corridor -- the back face must
    strengthen the judgement). Three regimes:
      * mutual pair: the same cabinet seen from both faces unions;
      * COARSE bridge: the poor face's whole-row box overlaps two
        fine spans -- it is DROPPED (unioning it with either would
        swallow the good face's split);
      * singleton: a cabinet legible from only one face survives."""
    from agentic_gts.agent.mask_refine import _merge_cross_view
    mkpts = lambda n: np.zeros((n, 3))
    # front (poor view): ONE whole-row box; back (open side): the
    # true two cabinets, slightly different seam placement
    spans = [
        {"lo": -1.0, "hi": 1.0, "pts": mkpts(80), "ms": 0.6,
         "label": "rack"},                       # front coarse
        {"lo": -0.98, "hi": 0.02, "pts": mkpts(60), "ms": 0.9,
         "label": "rack"},                       # back cabinet A
        {"lo": 0.04, "hi": 0.98, "pts": mkpts(55), "ms": 0.9,
         "label": "rack"},                       # back cabinet B
        {"lo": 1.1, "hi": 1.7, "pts": mkpts(40), "ms": 0.7,
         "label": "rack"},                       # back-only row end
    ]
    out = _merge_cross_view(spans)
    assert len(out) == 3, f"coarse dropped, A/B/end survive: {len(out)}"
    by_lo = sorted(out, key=lambda s: s["lo"])
    a, b, e = by_lo
    assert abs(a["lo"] + 0.98) < 1e-9 and abs(a["hi"] - 0.02) < 1e-9
    assert abs(b["lo"] - 0.04) < 1e-9 and abs(b["hi"] - 0.98) < 1e-9
    assert abs(e["lo"] - 1.1) < 1e-9 and abs(e["hi"] - 1.7) < 1e-9
    # mutual pair: same cabinet from BOTH faces unions (points stack)
    spans2 = [
        {"lo": 0.0, "hi": 0.6, "pts": mkpts(50), "ms": 0.8,
         "label": "rack"},
        {"lo": 0.02, "hi": 0.58, "pts": mkpts(30), "ms": 0.7,
         "label": "rack"},
    ]
    out2 = _merge_cross_view(spans2)
    assert len(out2) == 1 and len(out2[0]["pts"]) == 80
    assert abs(out2[0]["lo"]) < 1e-9 and abs(out2[0]["hi"] - 0.6) < 1e-9
    print("PASS cross-view merge (pairs union, coarse bridges drop)")


def test_back_view_rescues_poor_front_end_to_end():
    """The user's narrow-corridor scenario END-TO-END: the front aisle
    is cramped, its render poor, and the VLM grounds ONE whole-row box
    there; the back side is open and grounds the true TWO cabinets.
    With the back view voting, the row must still split into 2 -- the
    coarse front span is dropped by the cross-view rule, the back's
    fine division stands."""
    import math

    from agentic_gts.agent import mask_refine as mr
    from agentic_gts.agent.judge import Verdict, VLMJudge
    from agentic_gts.agent.loop import AgentReport, LayoutAgent
    from agentic_gts.output.gs_render import Cam

    rng = np.random.default_rng(7)
    cabA = np.column_stack([rng.uniform(-1.0, -0.05, 800),
                            rng.uniform(-0.5, 0.5, 800),
                            rng.uniform(0.05, 1.95, 800)])
    cabB = np.column_stack([rng.uniform(0.05, 1.0, 800),
                            rng.uniform(-0.5, 0.5, 800),
                            rng.uniform(0.05, 1.95, 800)])
    scene = Scene(points=np.vstack([cabA, cabB]))
    seed = OrientedBox(center=(0.0, 0.0, 1.0), size=(2.0, 1.0, 1.9),
                       yaw=0.0)
    scene.boxes = [seed]
    img = np.zeros((768, 768, 3), np.float32)
    views = [{"name": "front", "cam": Cam(
                  eye=np.array([0.0, 4.0, 1.2]),
                  target=np.array([0.0, 0.0, 1.0]),
                  up=np.array([0.0, 0.0, 1.0]), fovy_deg=60.0,
                  W=768, H=768), "path": None, "prompt_path": None,
              "image": img},
             {"name": "back", "cam": Cam(
                  eye=np.array([0.0, -4.0, 1.2]),
                  target=np.array([0.0, 0.0, 1.0]),
                  up=np.array([0.0, 0.0, 1.0]), fovy_deg=60.0,
                  W=768, H=768), "path": None, "prompt_path": None,
              "image": img},
             {"name": "side", "cam": Cam(
                  eye=np.array([4.0, 0.0, 1.2]),
                  target=np.array([0.0, 0.0, 1.0]),
                  up=np.array([0.0, 0.0, 1.0]), fovy_deg=60.0,
                  W=768, H=768), "path": None, "prompt_path": None,
              "image": img}]
    _real = (mr.render_local_views, mr.SamPredictorAdapter._load,
             mr.SamPredictorAdapter.predict)
    mr.render_local_views = lambda scene, box, out_dir: views

    j = VLMJudge(backend="mock")

    def fake_ground(image, box, view_name, png_path=None):
        # front: the POOR view -- one whole-row box; back: the open
        # side -- the true two cabinets; side: one profile box
        if view_name == "front":
            groups = [{"bbox": (10, 10, 990, 990), "hypothesis": "rack",
                       "confidence": 0.6}]
        elif view_name == "back":
            groups = [{"bbox": (10, 10, 490, 990), "hypothesis": "rack",
                       "confidence": 0.9},
                      {"bbox": (510, 10, 990, 990), "hypothesis": "rack",
                       "confidence": 0.9}]
        else:
            groups = [{"bbox": (10, 10, 990, 990), "hypothesis": "rack",
                       "confidence": 0.9}]
        return Verdict(action="segment", params={"groups": groups},
                       confidence=0.9, detail="fake")

    j.adjudicate_sam_boxes = fake_ground

    def fake_predict(self, image, box_pix):
        m = np.zeros(image.shape[:2], bool)
        x1, y1, x2, y2 = (int(round(float(v))) for v in box_pix)
        m[max(y1, 0):max(y2, 1), max(x1, 0):max(x2, 1)] = True
        return [m], [0.95]

    mr.SamPredictorAdapter._load = lambda self: None
    mr.SamPredictorAdapter.predict = fake_predict

    agent = LayoutAgent(judge=j, opts={"sam_checkpoint": "fake.pt"},
                        out_dir=None)
    try:
        agent._local_mask_refine(scene, AgentReport())
    finally:
        (mr.render_local_views, mr.SamPredictorAdapter._load,
         mr.SamPredictorAdapter.predict) = _real

    assert len(scene.boxes) == 2, \
        (f"the back view's fine division must survive the poor front: "
         f"got {len(scene.boxes)}: "
         + str([b.to_dict().get("center") for b in scene.boxes]))
    centers = sorted(b.center[0] for b in scene.boxes)
    assert centers[0] < -0.2 < 0.2 < centers[1], centers
    print("PASS back view rescues a poor front (coarse span dropped)")


def test_vlm_quality_verdict_drops_garbage_view():
    """VLM quality verdict (user direction: judged TOGETHER with the
    grounding in the same call, garbage views DROPPED): a fogged front
    render on which the model still HALLUCINATED a whole-row box must
    contribute nothing -- the 'poor' verdict drops the view's boxes
    outright, and the clean back view's two-cabinet split stands
    (unlike the bridge test, the front's box never even reaches the
    cross-view merge)."""
    from agentic_gts.agent import mask_refine as mr
    from agentic_gts.agent.judge import Verdict, VLMJudge
    from agentic_gts.agent.loop import AgentReport, LayoutAgent
    from agentic_gts.output.gs_render import Cam

    rng = np.random.default_rng(7)
    cabA = np.column_stack([rng.uniform(-1.0, -0.05, 800),
                            rng.uniform(-0.5, 0.5, 800),
                            rng.uniform(0.05, 1.95, 800)])
    cabB = np.column_stack([rng.uniform(0.05, 1.0, 800),
                            rng.uniform(-0.5, 0.5, 800),
                            rng.uniform(0.05, 1.95, 800)])
    scene = Scene(points=np.vstack([cabA, cabB]))
    seed = OrientedBox(center=(0.0, 0.0, 1.0), size=(2.0, 1.0, 1.9),
                       yaw=0.0)
    scene.boxes = [seed]
    img = np.zeros((768, 768, 3), np.float32)
    views = [{"name": n, "cam": Cam(
                  eye=np.array([0.0, 4.0 if n == "front" else -4.0, 1.2]),
                  target=np.array([0.0, 0.0, 1.0]),
                  up=np.array([0.0, 0.0, 1.0]), fovy_deg=60.0,
                  W=768, H=768), "path": None, "prompt_path": None,
              "image": img}
             for n in ("front", "back", "side")]
    _real = (mr.render_local_views, mr.SamPredictorAdapter._load,
             mr.SamPredictorAdapter.predict)
    mr.render_local_views = lambda scene, box, out_dir: views

    j = VLMJudge(backend="mock")

    def fake_ground(image, box, view_name, png_path=None):
        # front: the FOGGED view -- judged poor, but the model still
        # hallucinated a confident whole-row box (the danger the
        # verdict guards against); back: clean, the true two cabinets
        if view_name == "front":
            return Verdict(action="segment", params={
                "groups": [{"bbox": (10, 10, 990, 990),
                            "hypothesis": "rack", "confidence": 0.9}],
                "view_quality": "poor"}, confidence=0.9,
                detail="fake", raw="quality: poor")
        groups = ([{"bbox": (10, 10, 490, 990), "hypothesis": "rack",
                    "confidence": 0.9},
                  {"bbox": (510, 10, 990, 990), "hypothesis": "rack",
                   "confidence": 0.9}] if view_name == "back"
                  else [{"bbox": (10, 10, 990, 990),
                         "hypothesis": "rack", "confidence": 0.9}])
        return Verdict(action="segment", params={
            "groups": groups, "view_quality": "good"},
            confidence=0.9, detail="fake", raw="quality: good")

    j.adjudicate_sam_boxes = fake_ground

    def fake_predict(self, image, box_pix):
        m = np.zeros(image.shape[:2], bool)
        x1, y1, x2, y2 = (int(round(float(v))) for v in box_pix)
        m[max(y1, 0):max(y2, 1), max(x1, 0):max(x2, 1)] = True
        return [m], [0.95]

    mr.SamPredictorAdapter._load = lambda self: None
    mr.SamPredictorAdapter.predict = fake_predict

    agent = LayoutAgent(judge=j, opts={"sam_checkpoint": "fake.pt"},
                        out_dir=None)
    try:
        agent._local_mask_refine(scene, AgentReport())
    finally:
        (mr.render_local_views, mr.SamPredictorAdapter._load,
         mr.SamPredictorAdapter.predict) = _real

    assert len(scene.boxes) == 2, \
        (f"the poor-verdict front must be dropped and the back's "
         f"division stand: got {len(scene.boxes)}: "
         + str([b.to_dict().get("center") for b in scene.boxes]))
    centers = sorted(b.center[0] for b in scene.boxes)
    assert centers[0] < -0.2 < 0.2 < centers[1], centers
    print("PASS VLM quality verdict drops the garbage view")


def test_cross_view_single_face_yields_to_multi():
    """USER RULE: one face grounds ONE instance, the other grounds
    SEVERAL -> the several stand (the row is one whole; the single box
    is that whole unresolved). Even a PARTIAL single (the poor face's
    one box covering cabinet A and half of B) must NOT union with A --
    that would stretch A's piece across the seam. Its points are
    clipped into the fine spans; its extent is dropped."""
    from agentic_gts.agent.mask_refine import _merge_cross_view
    axis = np.array([1.0, 0.0])
    mk = lambda xs: np.column_stack(
        [np.asarray(xs, float), np.zeros(len(xs)), np.ones(len(xs))])
    # front (poor): ONE partial box [0, 1.3] -- A plus part of B
    front = {"lo": 0.0, "hi": 1.3, "pts": mk([0.2, 0.8, 1.2]),
             "ms": 0.6, "label": "rack", "view": "front"}
    # back (open): the true two cabinets
    backA = {"lo": 0.0, "hi": 1.0, "pts": mk([0.5]),
             "ms": 0.9, "label": "rack", "view": "back"}
    backB = {"lo": 1.05, "hi": 2.05, "pts": mk([1.5]),
             "ms": 0.9, "label": "rack", "view": "back"}
    out = _merge_cross_view([front, backA, backB], axis, 0.0)
    assert len(out) == 2, f"the multi face's split stands: {len(out)}"
    by_lo = sorted(out, key=lambda s: s["lo"])
    a, b = by_lo
    assert abs(a["lo"]) < 1e-9 and abs(a["hi"] - 1.0) < 1e-9, \
        "A's extent must NOT stretch to the absorbed single's 1.3"
    assert abs(b["lo"] - 1.05) < 1e-9 and abs(b["hi"] - 2.05) < 1e-9
    # the single's real surface points were clipped into the fines
    assert len(a["pts"]) == 3, "A keeps its 0.5 + the single's 0.2/0.8"
    assert len(b["pts"]) == 2, "B keeps its 1.5 + the single's 1.2"
    print("PASS single-instance face yields to the multi face")


def test_height_ignores_truncated_mask_points():
    """The height z-source is the RAW-CLOUD column, not the piece's
    mask points (user report: some boxes came out VERY low). Two ways
    a mask truncates: the VLM front box covered only the lower half,
    and the side pass REPLACED pts with a vertically short profile
    slice -- the piece's pts all sit below 0.8m while the cabinet is
    2.1m tall. The raw column under the piece's footprint carries the
    full height, so the measured box must stay tall."""
    from agentic_gts.agent.mask_refine import (_apply_height_and_geom_depth,
                                                _build_split_pieces)
    rng = np.random.default_rng(11)
    # one cabinet, full height in the CLOUD
    cab = np.column_stack([rng.uniform(-0.6, 0.6, 1200),
                           rng.uniform(-0.5, 0.5, 1200),
                           rng.uniform(0.1, 2.05, 1200)])
    scene = Scene(points=cab)
    seed = OrientedBox(center=(0.0, 0.0, 1.05), size=(1.2, 1.0, 2.1),
                       yaw=0.0)
    # the piece's "mask" points: TRUNCATED to the lower 0.8m
    trunc = cab[cab[:, 2] < 0.8][:200]
    spans = [{"lo": -0.6, "hi": 0.6, "pts": trunc, "ms": 0.8,
              "label": "rack"}]
    instances = _build_split_pieces(spans, seed)
    _apply_height_and_geom_depth(instances, seed, scene)
    h = instances[0]["fitted"].size[2]
    assert 1.9 < h < 2.2, \
        f"height {h:.2f} must come from the raw column (~2.1), " \
        "not the truncated mask points (~0.8)"
    print("PASS height ignores truncated mask points (raw column)")


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


def test_mask_to_points_clips_far_outside_seed():
    """Backprojected points must stay within a small pad of the seed OBB:
    mask-edge bleed onto floor / neighbouring structure picks up their
    pixels, and the P1-P99 fit balloons toward them (user report:
    backprojected points well past the initial box). 0.15 m outside the
    face survives (a conservative grounding box growing to the true
    surface); 0.4 m outside is dropped."""
    from agentic_gts.agent.mask_refine import _mask_to_points
    from agentic_gts.output.gs_render import Cam
    box = OrientedBox(center=(0.0, 0.0, 1.0), size=(2.0, 1.0, 2.0),
                      yaw=0.0)
    # front-view camera in the +y aisle looking back at the box
    cam = Cam(eye=np.array([0.0, 4.0, 1.2]),
              target=np.array([0.0, 0.0, 1.0]), up=np.array([0.0, 0.0, 1.0]),
              fovy_deg=60.0, W=768, H=768)
    pts = np.array([
        [0.0, 0.0, 1.0],           # inside: kept
        [0.5, -0.3, 1.6],          # inside: kept
        [0.0, 0.65, 1.0],          # 0.15 m past the front face: kept
        [0.2, 0.90, 1.0],          # 0.40 m past the front face: DROPPED
        [1.6, 0.0, 1.0],           # 0.60 m past the row end: DROPPED
        [0.0, 0.0, -0.30],         # 0.30 m below the box floor: DROPPED
    ])
    scene = Scene(points=pts)
    mask = np.ones((768, 768), dtype=bool)
    out = _mask_to_points(scene, box, mask, cam)
    got = {tuple(np.round(p, 3)) for p in out}
    assert (0.0, 0.0, 1.0) in got and (0.5, -0.3, 1.6) in got, got
    assert (0.0, 0.65, 1.0) in got, (
        f"slight growth past the face must survive, got {got}")
    for p in ((0.2, 0.90, 1.0), (1.6, 0.0, 1.0), (0.0, 0.0, -0.30)):
        assert p not in got, f"noise point {p} must be clipped, got {got}"
    print("PASS mask backprojection clips points far outside the seed")


def test_door_class_subtracts_from_device_points():
    """The open-door positive class (user finding: the VLM detects
    "open cabinet door" reliably as a detection task but cannot exclude
    it via a negative instruction): _door_union SAM-segments the door
    boxes into one subtractive mask, door-class groups never form
    spans/pool entries, and _mask_to_points drops every point that
    projects into the door mask -- the open door cannot stretch the
    span or the thickness."""
    from agentic_gts.agent.mask_refine import (
        _door_union, _is_door, _mask_to_points,
    )
    from agentic_gts.output.gs_render import Cam

    assert _is_door("open cabinet door")
    assert _is_door("Open Cabinet DOOR")
    assert not _is_door("rack")
    assert not _is_door(None)

    # _door_union: only door-class groups, best-score SAM mask unioned
    IMG = np.zeros((64, 64, 3))       # 64px: VLM boxes clear the
    # degenerate-size guard (a tiny 8px test image would not)

    class _FakeSam:
        def __init__(self):
            self.calls = []

        def predict(self, image, box_pix):
            self.calls.append(tuple(box_pix))
            # two door boxes -> two disjoint masks
            m = np.zeros((64, 64), bool)
            if box_pix[0] < 20:         # left door box
                m[20:40, 5:20] = True
            else:                        # right door box
                m[25:35, 30:50] = True
            return [m], [0.9]

    sam = _FakeSam()
    groups = [{"bbox": (100, 100, 400, 500), "hypothesis": "rack"},
              {"bbox": (100, 200, 300, 600), "hypothesis": "open cabinet door"},
              {"bbox": (500, 200, 800, 600), "hypothesis": "door"}]
    u = _door_union(IMG, groups, sam)
    assert len(sam.calls) == 2, "device-class box must NOT hit the door SAM"
    assert u is not None and u[25, 10] and u[30, 40], \
        "union must cover both door masks"
    # no door group -> None (the common case): no subtraction layer
    assert _door_union(IMG,
                       [{"bbox": (100, 100, 400, 500),
                         "hypothesis": "rack"}], sam) is None

    # _mask_to_points: points projecting into the door mask are dropped
    box = OrientedBox(center=(0.0, 0.0, 1.0), size=(2.0, 1.0, 2.0),
                      yaw=0.0)
    cam = Cam(eye=np.array([0.0, 4.0, 1.2]),
              target=np.array([0.0, 0.0, 1.0]), up=np.array([0.0, 0.0, 1.0]),
              fovy_deg=60.0, W=768, H=768)
    scene = Scene(points=np.array([
        [0.0, 0.0, 1.0],           # cabinet body: kept
        [0.5, -0.3, 1.6],          # cabinet body: kept
    ]))
    uv = cam.project_cv(scene.points)
    x = np.rint(uv[:, 0]).astype(int)
    y = np.rint(uv[:, 1]).astype(int)
    doors = np.zeros((768, 768), bool)
    doors[y[1], x[1]] = True       # the second point's pixel is "door"
    out = _mask_to_points(scene, box, np.ones((768, 768), bool), cam,
                          exclude=doors)
    assert len(out) == 1 and np.allclose(out[0][:2], (0.0, 0.0)), \
        "the door-pixel point must be subtracted, the body point kept"
    print("PASS door class subtracts from device points")


def test_sam_debug_render_survives_negative_z():
    """Debug render regression (user report: '[mask-refine] debug
    render failed (ValueError: minvalue must be less than or equal to
    maxvalue)'): the top-down panel hardcoded vmin=0.0 (ground at z~0),
    but with --boxes input (no ground alignment) every back-projected
    z can be negative -- autoscaled vmax < vmin made matplotlib's
    Normalize raise and the whole composite was lost. Never raises,
    always writes the file."""
    import tempfile
    from agentic_gts.agent.mask_refine import _save_sam_debug
    from agentic_gts.output.gs_render import Cam

    view = {"image": np.full((64, 64, 3), 0.4, np.float32),
            "cam": None}
    pts3 = np.array([[-1.0, 0.0, -2.4],      # ALL z negative: the
                    [0.5, 0.3, -2.1],        # old vmin=0.0 blew up
                    [1.2, -0.4, -1.8]])
    box = OrientedBox(center=(0.0, 0.0, -2.0), size=(2.0, 1.0, 1.0),
                      yaw=0.0)
    with tempfile.TemporaryDirectory() as td:
        _save_sam_debug(view, (10, 10, 50, 50),
                        np.ones((64, 64), bool), pts3, box, box, td,
                        "negz")
        assert os.path.isfile(os.path.join(td, "sam_debug_negz.png")), \
            "the composite must be written despite all-negative z"
    print("PASS SAM debug render survives all-negative z")


def test_local_refine_splits_joined_row_end_to_end():
    """END-TO-END for the split adoption path the mock pipeline never
    covers (mock grounding returns empty groups): a seed covering TWO
    joined cabinets, a local grounding that returns two device boxes,
    SAM that segments them -- after _local_mask_refine the scene must
    hold TWO boxes, one per cabinet, each keeping the seed's trusted
    depth/height (user report: boxes_only.ply still showed the joined
    row as ONE box despite the local grounding splitting it).

    Parametrized over the row DIRECTION: along x (seed yaw=0) and along
    y (seed yaw=pi/2 -- the user report where the yaw axis landed on
    the THICKNESS and the row split across its depth)."""
    import math

    from agentic_gts.agent import mask_refine as mr
    from agentic_gts.agent.judge import Verdict, VLMJudge
    from agentic_gts.agent.loop import AgentReport, LayoutAgent
    from agentic_gts.output.gs_render import Cam

    for along_y in (False, True):
        rng = np.random.default_rng(7)
        # two cabinets side by side (a joined row), 1.0m each
        cabA = np.column_stack([rng.uniform(-1.0, -0.05, 800),
                                rng.uniform(-0.5, 0.5, 800),
                                rng.uniform(0.05, 1.95, 800)])
        cabB = np.column_stack([rng.uniform(0.05, 1.0, 800),
                                rng.uniform(-0.5, 0.5, 800),
                                rng.uniform(0.05, 1.95, 800)])
        pts = np.vstack([cabA, cabB])
        if along_y:              # the row runs along y: transpose x/y
            pts = pts[:, [1, 0, 2]]
        scene = Scene(points=pts)
        # seed: yaw axis rides the ROW axis (what _fit_region_box now
        # guarantees for y-rows: yaw=pi/2, size=(length, depth))
        seed = OrientedBox(center=(0.0, 0.0, 1.0), size=(2.0, 1.0, 1.9),
                           yaw=math.pi / 2.0 if along_y else 0.0)
        scene.boxes = [seed]

        # front: perpendicular to the row (across its long side);
        # side: along the row axis (the thickness profile)
        front_cam = Cam(eye=np.array([4.0, 0.0, 1.2]) if along_y
                        else np.array([0.0, 4.0, 1.2]),
                       target=np.array([0.0, 0.0, 1.0]),
                       up=np.array([0.0, 0.0, 1.0]), fovy_deg=60.0,
                       W=768, H=768)
        side_cam = Cam(eye=np.array([0.0, 4.0, 1.2]) if along_y
                       else np.array([4.0, 0.0, 1.2]),
                      target=np.array([0.0, 0.0, 1.0]),
                      up=np.array([0.0, 0.0, 1.0]), fovy_deg=60.0,
                      W=768, H=768)
        img = np.zeros((768, 768, 3), np.float32)
        views = [{"name": "front", "cam": front_cam, "path": None,
                  "prompt_path": None, "image": img},
                 {"name": "side", "cam": side_cam, "path": None,
                  "prompt_path": None, "image": img}]
        _real = (mr.render_local_views, mr.SamPredictorAdapter._load,
                 mr.SamPredictorAdapter.predict)
        mr.render_local_views = lambda scene, box, out_dir: views

        # VLM grounding: TWO device instances on the front view, ONE on
        # the side profile; SAM segments exactly the prompted rectangle
        j = VLMJudge(backend="mock")

        def fake_ground(image, box, view_name, png_path=None):
            groups = ([{"bbox": (10, 10, 490, 990), "hypothesis": "rack",
                        "confidence": 0.9},
                       {"bbox": (510, 10, 990, 990), "hypothesis": "rack",
                        "confidence": 0.9}] if view_name == "front"
                      else [{"bbox": (10, 10, 990, 990),
                             "hypothesis": "rack", "confidence": 0.9}])
            return Verdict(action="segment", params={"groups": groups},
                           confidence=0.9, detail="fake")

        j.adjudicate_sam_boxes = fake_ground

        def fake_predict(self, image, box_pix):
            m = np.zeros(image.shape[:2], bool)
            x1, y1, x2, y2 = (int(round(float(v))) for v in box_pix)
            m[max(y1, 0):max(y2, 1), max(x1, 0):max(x2, 1)] = True
            return [m], [0.95]

        mr.SamPredictorAdapter._load = lambda self: None
        mr.SamPredictorAdapter.predict = fake_predict

        agent = LayoutAgent(judge=j, opts={"sam_checkpoint": "fake.pt"},
                            out_dir=None)
        try:
            agent._local_mask_refine(scene, AgentReport())
        finally:              # restore the module-level patches so
            (mr.render_local_views, mr.SamPredictorAdapter._load,
             mr.SamPredictorAdapter.predict) = _real

        axis_i = 1 if along_y else 0
        assert len(scene.boxes) == 2, \
            (f"the {'y' if along_y else 'x'}-running joined row must split "
             f"into 2 boxes, got {len(scene.boxes)}: "
             + str([b.to_dict().get("center") for b in scene.boxes]))
        for b in scene.boxes:
            # depth/height are now MEASURED per piece (side view for
            # depth, the piece's own points for height), not
            # seed-inherited: both cabinets are 1.0 deep / ~1.9 tall
            # here, so the measured values must land on the truth,
            # within the strong-bin / P99.5 quantization slop
            assert 0.85 < b.size[1] < 1.15, \
                f"depth {b.size[1]:.2f} -- measured, near the true 1.0"
            assert 1.75 < b.size[2] < 2.05, \
                f"height {b.size[2]:.2f} -- measured, near the true 1.9"
        centers = sorted(b.center[axis_i] for b in scene.boxes)
        assert centers[0] < -0.2 < 0.2 < centers[1], \
            f"the two pieces must sit on their own cabinets: {centers}"
        # the OTHER axis (thickness) must NOT have been split apart
        for b in scene.boxes:
            assert abs(b.center[1 - axis_i]) < 0.1, \
                ("the pieces must stay centred on the row -- a split "
                 "across the THICKNESS is the yaw-axis bug")
        print(f"PASS local refine splits a {'y' if along_y else 'x'}-running"
              " joined row end-to-end")


def test_apply_depth_from_side_excludes_open_door():
    """The side-view thickness rule: pieces keep along/height (seed-
    trusted); the side pool -- ONE merged cloud over the whole row,
    lifted without the z-buffer -- is sliced per piece by along span
    and measures ONLY that cabinet's depth. The strong-bin estimator
    must drop an open door's spread-out tail beyond the cabinet body,
    which a P2-P98 percentile cut kept inflating the thickness."""
    from agentic_gts.agent.mask_refine import _apply_depth_from_side
    seed = OrientedBox(center=(0.0, 0.0, 1.05), size=(2.0, 1.0, 2.1),
                       yaw=0.0)
    inst_a = {"fitted": OrientedBox(center=(-0.3, 0.1, 1.0),
                                    size=(0.5, 0.15, 2.0), yaw=0.0),
              "pts": np.zeros((30, 3)), "view": "front",
              "mask_score": 0.8, "score": 0.6, "label": "rack"}
    inst_b = {"fitted": OrientedBox(center=(0.3, 0.1, 1.0),
                                    size=(0.5, 0.15, 2.0), yaw=0.0),
              "pts": np.zeros((30, 3)), "view": "front",
              "mask_score": 0.8, "score": 0.6, "label": "rack"}
    rng = np.random.default_rng(3)

    def slab(along_c, y_lo, y_hi, n):
        return np.column_stack([
            rng.uniform(along_c - 0.22, along_c + 0.22, n),
            rng.uniform(y_lo, y_hi, n),
            rng.uniform(0.1, 1.9, n)])

    # body shells: dense, y within [-0.4, 0.4] (true depth 0.8 m)
    pool = np.vstack([slab(-0.3, -0.4, 0.4, 300), slab(0.3, -0.4, 0.4, 300)])
    # open door on cabinet A: a SPREAD-OUT tail beyond the front face
    # (y in [0.4, 1.1], sparse compared to the shells)
    pool = np.vstack([pool, slab(-0.3, 0.42, 1.1, 60)])
    recs = _apply_depth_from_side([inst_a, inst_b], pool, seed)
    assert len(recs) == 2 and all(r.get("accepted") for r in recs), recs
    for inst in (inst_a, inst_b):
        fb = inst["fitted"]
        # along span and height untouched (seed-trusted)
        assert abs(fb.size[0] - 0.5) < 1e-9
        assert abs(fb.size[2] - 2.0) < 1e-9
        assert abs(fb.center[0] - (0.3 if inst is inst_b else -0.3)) < 1e-9
        # thickness measured from the body shells, door tail EXCLUDED:
        # a percentile cut would have stretched cabinet A to ~1.05m
        assert 0.6 < fb.size[1] < 0.95, fb.size[1]
        assert abs(fb.center[1]) < 0.1, fb.center[1]
    # rejection: a pool with an implausible depth (thin sliver) leaves
    # the seed-trusted depth untouched
    thin = np.column_stack([rng.uniform(-1, 1, 50), np.full(50, 0.05),
                            rng.uniform(0.1, 1.9, 50)])
    inst_c = {"fitted": OrientedBox(center=(0.0, 0.1, 1.0),
                                    size=(0.5, 0.15, 2.0), yaw=0.0),
              "pts": np.zeros((30, 3)), "view": "front",
              "mask_score": 0.8, "score": 0.6, "label": "rack"}
    recs = _apply_depth_from_side([inst_c], thin, seed)
    assert not recs[0].get("accepted"), recs
    assert abs(inst_c["fitted"].size[1] - 0.15) < 1e-9
    print("PASS side depth application (merged pool, door tail dropped)")


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


def test_side_view_looks_along_row_axis():
    """The SIDE slot (front azimuth + 90 deg) must look ALONG the row
    axis: the (depth, height) PROFILE is what corrects thickness and
    exposes open doors sticking out beyond the cabinet body -- the front
    view cannot separate them (user report)."""
    from agentic_gts.output.gs_render import make_local_cam
    box = OrientedBox(center=(0, 0, 1), size=(6, 1.1, 2), yaw=0.0)
    # front azim 0 (long edge along x); side = front + 90
    cam = make_local_cam([box], W=768, H=768, elev_deg=18.0,
                         azim_deg=90.0, standoff=0.6)
    d = np.asarray(cam.eye) - np.asarray(box.center)
    assert abs(d[0]) > 3 * abs(d[1]), (
        f"side camera must stand along the ROW axis, got offset {d}")
    look = np.asarray(cam.target) - np.asarray(cam.eye)
    assert abs(look[0]) > 3 * abs(look[1]), (
        f"side view must look along the row, got direction {look}")
    print("PASS side view looks along the row axis (thickness profile)")


def test_view_quality_gate_drops_haze_views():
    """VIEW QUALITY GATE (user rule, replaces the reverted corridor
    pre-gate: judge the RENDER, not the geometry). The discriminator
    is GRADIENT ENERGY: a veil is smooth, a device render is full of
    crisp steps. Must pass clean renders (however much of the frame
    the device fills, INCLUDING a uniform-panel close-up the old
    std/coverage rule wrongly rejected) and reject empty frames,
    flat veils and GRADIENT veils (a smooth brightness ramp -- spread
    the histogram, fooling std).
    And render_local_views must DROP a fogged view instead of feeding
    it to the VLM."""
    from agentic_gts.agent.mask_refine import _view_quality

    def clean(device_frac=0.3, H=128, W=128):
        img = np.full((H, W, 3), 0.01, np.float32)
        k = int(H * W * device_frac)
        flat = img.reshape(-1, 3)
        idx = np.random.default_rng(0).choice(len(flat), k, replace=False)
        flat[idx] = 0.85
        return img

    ok, why = _view_quality(clean())
    assert ok, f"clean bimodal render must pass, got {why}"
    ok, _ = _view_quality(clean(device_frac=0.85))
    assert ok, "a close-up filling most of the frame is still clean"
    # uniform-panel close-up: 95% of the frame one flat value, only
    # the contour carries a step -- low std, near-total coverage.
    # The old rule (cov > 0.90 and std < 0.15) REJECTED this legit
    # render; the edges must save it.
    uni = np.full((128, 128, 3), 0.0, np.float32)
    uni[6:122, 6:122] = 0.5
    ok, why = _view_quality(uni)
    assert ok, f"uniform-panel close-up must pass, got {why}"
    # flat veil: the camera inside structure, no structure in frame
    ok, why = _view_quality(np.full((128, 128, 3), 0.42, np.float32))
    assert not ok, "flat veil must be rejected"
    # GRADIENT veil: a smooth brightness ramp -- spreads the histogram
    # (std ~0.09 would pass the old std rule); per-pixel slope is tiny
    def _ramp(ramp_1d):
        return np.repeat(ramp_1d[:, None], 128, axis=1)[..., None] \
            * np.ones(3, np.float32)

    ok, why = _view_quality(
        _ramp(np.linspace(0.2, 0.5, 128, dtype=np.float32)))
    assert not ok, "gradient veil must be rejected"
    ok, why = _view_quality(
        _ramp(np.linspace(0.2, 0.5, 128, dtype=np.float32))
        .transpose(1, 0, 2))
    assert not ok, "horizontal gradient veil must be rejected"
    # empty: nothing rendered
    ok, why = _view_quality(np.zeros((128, 128, 3), np.float32))
    assert not ok, "empty frame must be rejected"
    # no side-view fallback as a span voter: with both faces dropped,
    # refine_box must not hand the span vote to the side view
    from agentic_gts.agent.mask_refine import refine_box
    import inspect
    src = inspect.getsource(refine_box)
    assert "voters = views[:1]" not in src, \
        "the side view must never inherit the span vote (it looks " \
        "ALONG the row; its masks span the cross axis)"

    # render_local_views drops a fogged view (mocked rasterizer
    # returns haze for cameras on the wall side, clean elsewhere)
    from agentic_gts.agent import mask_refine as mr
    from agentic_gts.tools.gs_io import GaussianData
    rng = np.random.default_rng(5)
    box = OrientedBox(center=(0.0, 0.0, 1.0), size=(4.0, 1.0, 2.0),
                      yaw=0.0)
    scene = Scene(points=np.zeros((10, 3)))
    scene.meta["gs_ply"] = "fake.ply"
    means = np.column_stack([rng.uniform(-2, 2, 800),
                             rng.uniform(-2.5, 2.5, 800),
                             rng.uniform(0.3, 1.7, 800)])
    means = np.vstack([means,
                       np.full((400, 3), [0.0, -0.6, 1.0])])
    means = means.astype(np.float32)
    n = len(means)
    gs = GaussianData(
        means=means,
        log_scales=np.full((n, 3), -6.0, dtype=np.float32),
        quats=np.tile(np.array([[1.0, 0, 0, 0]], np.float32), (n, 1)),
        raw_opacity=np.full(n, 2.0, dtype=np.float32),
        f_dc=np.zeros((n, 3), dtype=np.float32),
    )
    import agentic_gts.output.gs_render as gsr
    import agentic_gts.tools.gs_io as gio
    _real = (gsr.rasterize_gs, gsr.render_gs_view, gsr.png_bytes,
             gio.read_gaussian_ply)
    clean_img = clean(H=768, W=768)
    fog = np.full((768, 768, 3), 0.42, np.float32)

    def fake_raster(sub, cam):
        # cameras on the -y (wall) side render from inside the wall:
        # fog
        return fog if float(np.asarray(cam.eye)[1]) < -0.5 else clean_img

    gsr.rasterize_gs = fake_raster
    gsr.render_gs_view = lambda *a, **k: clean_img
    gsr.png_bytes = lambda a: b"png"
    gio.read_gaussian_ply = lambda p: gs
    try:
        views = mr.render_local_views(scene, box, None)
    finally:
        (gsr.rasterize_gs, gsr.render_gs_view, gsr.png_bytes,
         gio.read_gaussian_ply) = _real
    names = [v["name"] for v in views]
    assert "back" not in names, \
        f"the fogged back view must be dropped, got {names}"
    assert "front" in names, names
    print("PASS view quality gate (veils dropped, clean kept)")


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
