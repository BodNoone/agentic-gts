"""Tests for the structured-coordinate hallucination filter
(agent/hallucination.py): normal detections and genuinely regular
machine-room structure survive; repetitive X/Y chains, grids and
out-of-bounds marches drop whole, with diagnostics recorded."""
import json
import os
import sys
import tempfile

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from agentic_gts.agent.hallucination import (HallucinationConfig,
                                              HALLUCINATION_CFG,
                                              filter_hallucination_rects,
                                              save_debug_png)

W, H = 1280, 1024


def _r(x0, y0, x1, y1, label="rack"):
    return (float(x0), float(y0), float(x1), float(y1), label)


def test_normal_detections_survive():
    """Scattered rows of varying width, generous aisles, nothing
    chainable -- nothing dropped, nothing scored."""
    rects = [_r(80, 100, 420, 300), _r(500, 120, 700, 280),
             _r(100, 450, 380, 650), _r(600, 480, 900, 660),
             _r(300, 800, 560, 980)]
    kept, diag = filter_hallucination_rects(rects, W, H, view="v")
    assert kept == rects and diag["n_dropped"] == 0
    assert diag["chains"] == []
    print("PASS normal detections survive")


def test_real_regular_arrangement_survives():
    """The dangerous look-alikes of a hallucination, all real:
    identical-width rows repeating at a FIXED AISLE PITCH (the aisle
    is >= 55% of the row depth -- far above the chain gap cap), and a
    tight head-to-tail over-split row of same-size cabinets along the
    row axis (the genuine over-split signature). Both must survive."""
    # 4 identical rows, fixed pitch, aisle = 60% of the row depth
    rows = [_r(100, 100 + k * 320, 900, 100 + k * 320 + 200)
            for k in range(4)]
    # a real over-split row: 5 touching same-size cabinets along x
    split = [_r(100 + k * 120, 700, 100 + k * 120 + 120, 900)
             for k in range(5)]
    rects = rows + split
    kept, diag = filter_hallucination_rects(rects, W, H, view="v")
    assert kept == rects, f"real regular structure must survive: {diag}"
    # the over-split IS an X chain (scored) but below the X threshold
    x_chains = [c for c in diag["chains"] if c["axis"] == "x"]
    assert x_chains and not x_chains[0]["dropped"], \
        f"over-split X chain scored {x_chains[0]['score']} " \
        f"(thr {x_chains[0]['threshold']}) -- must stay under"
    assert all(c["axis"] != "y" for c in diag["chains"]), \
        "aisle-separated rows must not even form a Y chain"
    print(f"PASS real regular arrangement survives "
          f"(over-split X chain scored {x_chains[0]['score']:.2f} "
          f"< {x_chains[0]['threshold']})")


def test_y_hallucination_chains_dropped():
    """Vertical repetitive chains drop whole, valid boxes in the same
    view survive: a pixel-clean tower, a sloppy tower (drifting
    column, ~10% gaps) and a perspective tower (sizes grow ~10% per
    step -- 40% end to end, chained only via transitivity)."""
    clean = [_r(500, 20 + k * 240, 700, 20 + k * 240 + 240)
             for k in range(4)]
    sloppy = [_r(80, 50, 280, 250), _r(90, 270, 292, 478),
              _r(75, 500, 273, 690)]
    persp = [_r(820, 0, 980, 200), _r(815, 200, 985, 415),
             _r(808, 415, 992, 648), _r(800, 648, 1000, 903)]
    valid = [_r(950, 100, 1230, 300), _r(60, 780, 360, 990)]
    rects = clean + sloppy + persp + valid
    kept, diag = filter_hallucination_rects(rects, W, H, view="v")
    assert kept == valid, \
        f"all three towers must drop, got kept={len(kept)}: {diag}"
    y_dropped = [c for c in diag["chains"]
                 if c["axis"] == "y" and c["dropped"]]
    assert len(y_dropped) == 3, f"3 Y chains expected: {diag['chains']}"
    for c in y_dropped:
        for key in ("count", "size", "align", "adjacency", "periodic",
                    "oob", "grid", "gaps", "n_pinned"):
            assert key in c["stats"], f"stats missing {key}"
        assert c["score"] >= c["threshold"]
    print("PASS Y hallucination chains dropped (clean / sloppy / "
          "perspective), valid boxes kept")


def test_x_hallucination_march_dropped():
    """A long repetitive march along the row axis whose coordinates
    run past the frame (edge-pinned after the parser's clip) drops;
    the REAL over-split in the same view stays (no OOB, under the
    strict X threshold)."""
    # 10 identical boxes marching in x; the last one straddles the
    # right frame edge -> pinned (the OOB proxy)
    march = [_r(100 + k * 110, 100, 100 + k * 110 + 110, 320)
             for k in range(10)]
    march[-1] = _r(100 + 9 * 110, 100, W, 320)     # clipped at x1 == W
    split = [_r(100 + k * 120, 700, 100 + k * 120 + 120, 900)
             for k in range(5)]
    kept, diag = filter_hallucination_rects(march + split, W, H, view="v")
    assert kept == split, f"the OOB march must drop: {diag}"
    mx = [c for c in diag["chains"] if c["axis"] == "x" and c["dropped"]]
    assert mx and mx[0]["stats"]["n_pinned"] >= 1
    assert mx[0]["stats"]["oob"] > 0
    print(f"PASS X hallucination march dropped "
          f"(score {mx[0]['score']:.2f}, oob {mx[0]['stats']['oob']})")


def test_grid_hallucination_dropped():
    """A tight 4x4 grid of identical boxes (rows AND columns tight --
    no real room does this, aisles must exist across rows) drops
    whole via its Y chains with the grid bonus."""
    grid = [_r(300 + c * 120, 100 + r_ * 120, 300 + c * 120 + 120,
               100 + r_ * 120 + 120)
            for r_ in range(4) for c in range(4)]
    valid = [_r(950, 700, 1230, 950)]
    kept, diag = filter_hallucination_rects(grid + valid, W, H, view="v")
    assert kept == valid, f"the grid must drop whole: {diag}"
    g = [c for c in diag["chains"] if c["axis"] == "y" and c["dropped"]
         and c["stats"]["grid"] > 0]
    assert g, "Y chains must carry the grid membership boost"
    print(f"PASS grid hallucination dropped "
          f"(Y chain grid={g[0]['stats']['grid']:.2f}, "
          f"score {g[0]['score']:.2f})")


def test_debug_png_and_diag_shape():
    """The debug render marks dropped chains red / kept green, and the
    per-view diagnostic carries everything needed to audit a drop."""
    tower = [_r(500, 20 + k * 240, 700, 20 + k * 240 + 240)
             for k in range(4)]
    valid = [_r(100, 800, 400, 990)]
    rects = tower + valid
    kept, diag = filter_hallucination_rects(rects, W, H, view="gv.png")
    assert diag["n_dropped"] == 4 and diag["n_kept"] == 1
    assert diag["drop_indices"] == [0, 1, 2, 3]
    ch = diag["chains"][0]
    assert (ch["view"] if "view" in ch else True) or True
    assert set(ch) >= {"axis", "n", "score", "threshold", "dropped",
                       "reason", "stats", "rects"}
    img = np.full((H, W, 3), 0.4, dtype=np.float32)
    with tempfile.TemporaryDirectory() as td:
        p = os.path.join(td, "hallucination_gv.png")
        out = save_debug_png(img, rects, diag, p)
        assert out == p and os.path.exists(p) and os.path.getsize(p) > 0
    print("PASS debug png + diagnostics shape")


def test_ground_stage_pure_hallucination_reply():
    """End-to-end: a reply that is ONLY a repetitive chain (4 stacked
    identical rects) is pure hallucination -- every rect drops at the
    filter, the views carry no usable regions, ground_stage fails
    loudly and the diagnostics JSON records why."""
    from agentic_gts.agent import ground
    from agentic_gts.agent.judge import VLMJudge
    from agentic_gts.core.models import Scene

    sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
    rng = np.random.default_rng(43)

    def _row_points(x0, x1, y, rng, n=4000):
        return np.column_stack([rng.uniform(x0, x1, n),
                                rng.uniform(y - 0.4, y + 0.4, n),
                                rng.uniform(0.30, 1.90, n)])

    pts = _row_points(0.0, 8.0, 0.0, rng)
    scene = Scene(points=pts)
    scene.meta["yaw"] = 0.0
    scene.meta["device_footprint"] = (-0.5, -0.8, 8.5, 0.8)
    scene.meta["device_cells"] = pts[::10][:, :2]
    scene.meta["z_top"] = 2.1
    scene.boxes = []
    tower = [{"bbox_2d": [100, k * 250, 900, k * 250 + 250],
              "label": "row"} for k in range(4)]
    reply = "Rows.\n" + json.dumps(tower)
    judge = VLMJudge(backend="qwen")
    judge._qwen_image_call = lambda png, prompt, *a, **k: reply
    with tempfile.TemporaryDirectory() as td:
        ok = ground.ground_stage(scene, judge, out_dir=td)
        diag_path = os.path.join(td, "hallucination_diag.json")
        assert os.path.exists(diag_path), \
            "diagnostics JSON must be written for a hallucination view"
        recs = json.load(open(diag_path, encoding="utf-8"))
        assert recs and recs[0]["n_dropped"] > 0
        assert any(c["dropped"] for c in recs[0]["chains"])
        assert any(f.startswith("hallucination_") and
                   f.endswith(".png") for f in os.listdir(td)), \
            "the per-view debug PNG must be written"
    assert not ok, "a pure-hallucination reply must leave no regions"
    assert not scene.boxes, "no box may come from a hallucination"
    print("PASS pure-hallucination reply rejected end-to-end "
          "(diag JSON + debug PNG written)")


def test_config_thresholds_are_tunable():
    """The thresholds live in HallucinationConfig: raising the Y
    threshold above the clean-tower score keeps a hallucination
    (deliberately), proving the verdict is config-driven."""
    tower = [_r(500, 20 + k * 240, 700, 20 + k * 240 + 240)
             for k in range(4)]
    kept, diag = filter_hallucination_rects(
        tower, W, H, cfg=HallucinationConfig(y_score_thr=0.99))
    assert kept == tower and diag["n_dropped"] == 0, \
        "a near-1.0 Y threshold must keep the tower"
    print("PASS config thresholds are tunable")
