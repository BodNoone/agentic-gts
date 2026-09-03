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
    """The per-box local crop must be persisted during the repair loop."""
    import glob
    import tempfile
    import shutil
    class FlaggingJudge(VLMJudge):
        def adjudicate_godview(self, scene, boxes):
            return [{"index": 0, "reason": "in aisle"}]
        def adjudicate_box(self, scene, box, question, options):
            return type("V", (), {"action": "keep", "params": {"choice": "real device"},
                                  "confidence": 0.8, "detail": "stub", "raw": ""})()

    scene = _scene_with_racks()
    from agentic_gts.core.models import OrientedBox
    scene.boxes = [OrientedBox(center=(0, 0, 1), size=(0.6, 1.1, 2.0), yaw=0.0)
                   for _ in range(3)]
    out = tempfile.mkdtemp(prefix="godview_ev_")
    try:
        agent = LayoutAgent(judge=FlaggingJudge(backend="qwen"), out_dir=out)
        agent.run(scene)
        ev = glob.glob(os.path.join(out, "evidence_*.png"))
        assert ev, f"no evidence png saved to {out}"
        assert os.path.getsize(ev[0]) > 1_000
        print(f"PASS local evidence saved: {os.path.basename(ev[0])}")
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
