"""God-view pass tests: rendering, JSON parsing, and issue injection."""
from __future__ import annotations

import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np

from agentic_gts.core.models import Scene
from agentic_gts.agent.judge import VLMJudge, render_godview_png, _extract_json
from agentic_gts.agent.loop import LayoutAgent


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


def test_godview_render_produces_png():
    scene = _scene_with_racks()
    boxes = scene.boxes
    png = render_godview_png(scene.points, boxes)
    assert png[:8] == b"\x89PNG\r\n\x1a\n" and len(png) > 20_000
    print(f"PASS godview render: {len(png) // 1024} KB png")


def test_ceiling_autocut():
    """A separated dense top slab (ceiling) must be cut; a bare scene with
    no ceiling must not be."""
    from agentic_gts.agent.judge import _auto_ceiling_z
    rng = np.random.default_rng(1)
    floor = np.zeros(4000)                     # dense floor z=0
    racks = rng.uniform(0, 2.0, 8000)          # devices 0..2m
    no_ceiling = np.concatenate([floor, racks])
    assert not np.isfinite(_auto_ceiling_z(no_ceiling)), \
        "bare scene wrongly cut"
    ceiling = rng.uniform(4.0, 4.3, 50000)     # dense slab at 4m (big gap)
    with_ceiling = np.concatenate([no_ceiling, ceiling])
    cut = _auto_ceiling_z(with_ceiling)
    assert np.isfinite(cut) and 1.9 < cut < 2.3, f"cut={cut}"
    print(f"PASS ceiling autocut: cut at z={cut:.1f} (ceiling 4.0-4.3 kept out)")


def test_godview_render_drops_ceiling():
    """With boxes given, the cut is the tallest box top + margin: the dense
    4m ceiling slab must be excluded, the 2m racks kept."""
    from agentic_gts.agent.judge import _render_cut_z
    from agentic_gts.core.models import OrientedBox
    scene = _scene_with_racks()
    rng = np.random.default_rng(2)
    cx = rng.uniform(-1, 4, 60000)
    cy = rng.uniform(-1, 7, 60000)
    cz = rng.uniform(4.0, 4.2, 60000)          # dense ceiling slab at 4m
    pts = np.vstack([scene.points, np.stack([cx, cy, cz], axis=1)])
    boxes = [OrientedBox(center=(k * 0.62, r * 2.4, 1.0),
                         size=(0.6, 1.1, 2.0), yaw=0.0)
             for r in range(3) for k in range(4)]
    cut = _render_cut_z(pts, boxes)
    assert 2.0 < cut < 2.5, f"cut={cut} (expected just above box tops)"
    kept = pts[pts[:, 2] < cut]
    assert (kept[:, 2] < 2.5).all(), "ceiling points leaked into the render"
    assert len(kept) < len(pts), "nothing was cut"
    png = render_godview_png(pts, boxes)
    assert png[:8] == b"\x89PNG\r\n\x1a\n" and len(png) > 20_000
    print(f"PASS ceiling cut from box heights: cut={cut:.1f}, "
          f"{len(pts) - len(kept)} of {len(pts)} points removed")


def test_extract_json_variants():
    assert _extract_json('{"suspicious": []}') == {"suspicious": []}
    assert _extract_json('Here it is:\n{"suspicious": [{"index": 2, '
                          '"reason": "aisle"}]} hope it helps') is not None
    assert _extract_json("no json at all") is None
    assert _extract_json("broken { not json") is None
    print("PASS json extraction")


def test_mock_backend_godview_is_noop():
    """Mock judge must return no godview issues (pipeline unaffected)."""
    scene = _scene_with_racks()
    agent = LayoutAgent(judge=VLMJudge(backend="mock"))
    issues = agent.godview_pass(scene)
    assert issues == []
    print("PASS mock godview noop")


def test_godview_bad_reply_is_contained():
    """A VLM backend that returns garbage must not crash the loop."""
    class BrokenJudge(VLMJudge):
        def adjudicate_godview(self, scene, boxes):
            raise RuntimeError("network down")

    scene = _scene_with_racks()
    agent = LayoutAgent(judge=BrokenJudge(backend="qwen"))
    issues = agent.godview_pass(scene)
    assert issues == []
    print("PASS broken backend contained")


def test_godview_flag_becomes_issue():
    """A judge flagging box 0 must produce a FALSE_POSITIVE issue for it."""
    class FlaggingJudge(VLMJudge):
        def adjudicate_godview(self, scene, boxes):
            return [{"index": 0, "reason": "in aisle"}]

    scene = _scene_with_racks()
    from agentic_gts.core.models import OrientedBox
    scene.boxes = [OrientedBox(center=(0, 0, 1), size=(0.6, 1.1, 2.0), yaw=0.0)
                   for _ in range(3)]
    agent = LayoutAgent(judge=FlaggingJudge(backend="qwen"))
    issues = agent.godview_pass(scene)
    assert len(issues) == 1
    assert issues[0].box_ids == [scene.boxes[0].box_id]
    assert "godview" in issues[0].detail
    print("PASS godview flag -> issue")


def test_local_evidence_saved():
    """The per-box evidence (three-view composite) must be persisted during
    the repair loop -- by the JUDGE, before the VLM call, so it is saved
    even when the backend call itself fails."""
    import glob
    import tempfile
    import shutil
    class FlaggingJudge(VLMJudge):
        def adjudicate_godview(self, scene, boxes):
            return [{"index": 0, "reason": "in aisle"}]

    scene = _scene_with_racks()
    from agentic_gts.core.models import OrientedBox
    scene.boxes = [OrientedBox(center=(0, 0, 1), size=(0.6, 1.1, 2.0), yaw=0.0)
                   for _ in range(3)]
    out = tempfile.mkdtemp(prefix="godview_ev_")
    try:
        judge = FlaggingJudge(backend="qwen")
        # enable the judge-side evidence dir (as pipeline.py does)
        judge.set_record(os.path.join(out, "vlm_records.jsonl"))
        agent = LayoutAgent(judge=judge, out_dir=out)
        agent.run(scene)
        ev = glob.glob(os.path.join(out, "evidence_*.png"))
        assert ev, f"no evidence png saved to {out}"
        assert os.path.getsize(ev[0]) > 1_000
        print(f"PASS local evidence saved: {os.path.basename(ev[0])}")
    finally:
        shutil.rmtree(out, ignore_errors=True)


def test_final_godview_qa_flags_low_confidence():
    """The post-repair god-view QA must NOT delete late-flagged boxes --
    it marks them LOW confidence and reports them unresolved."""
    import tempfile
    import shutil
    class LateFlaggingJudge(VLMJudge):
        """First god-view: clean. Final QA (2nd call): flag box 0."""
        def __init__(self, *a, **kw):
            super().__init__(*a, **kw)
            self.calls = 0
        def adjudicate_godview(self, scene, boxes):
            self.calls += 1
            if self.calls >= 2:
                return [{"index": 0, "reason": "off every row"}]
            return []

    scene = _scene_with_racks()
    from agentic_gts.core.models import OrientedBox, Confidence
    # boxes placed ON the racks (no overlap issues: those would be fixed
    # by resolve_overlap and change the count this test asserts on)
    scene.boxes = [OrientedBox(center=(k * 0.62, 0, 1),
                               size=(0.6, 1.1, 2.0), yaw=0.0)
                   for k in range(3)]
    out = tempfile.mkdtemp(prefix="godview_qa_")
    try:
        judge = LateFlaggingJudge(backend="qwen")
        agent = LayoutAgent(judge=judge, out_dir=out)
        report = agent.run(scene)
        # box 0 still exists (no late deletion) ...
        assert len(scene.boxes) == 3, "final QA must not delete boxes"
        # ... but is flagged LOW and unresolved for human review
        assert scene.boxes[0].confidence == Confidence.LOW
        assert any("final godview" in str(e) for e in report.unresolved), \
            "late flag must surface as unresolved"
        assert os.path.exists(os.path.join(out, "godview_final.png")), \
            "final godview render must be persisted"
        print("PASS final godview QA: flags LOW, no deletion")
    finally:
        shutil.rmtree(out, ignore_errors=True)


def test_low_confidence_empty_verdict_not_deleted():
    """A low-confidence 'empty space' verdict must NOT delete the box --
    deletion requires confidence >= 0.6 (irreversible action)."""
    import tempfile
    import shutil
    class UnsureJudge(VLMJudge):
        def adjudicate_godview(self, scene, boxes):
            return [{"index": 0, "reason": "in aisle"}]
        def adjudicate_box(self, scene, box, question, options):
            return type("V", (), {"action": "answer",
                                  "params": {"choice": "empty space"},
                                  "confidence": 0.3, "detail": "unsure",
                                  "raw": ""})()

    scene = _scene_with_racks()
    from agentic_gts.core.models import OrientedBox
    scene.boxes = [OrientedBox(center=(k * 0.62, 0, 1),
                               size=(0.6, 1.1, 2.0), yaw=0.0)
                   for k in range(3)]
    agent = LayoutAgent(judge=UnsureJudge(backend="qwen"))
    agent.run(scene)
    assert len(scene.boxes) == 3, \
        "low-confidence empty-space verdict must not delete"
    print("PASS low-confidence verdict does not delete")


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
