"""Tests for the per-box local-view + VLM-verdict HTML report."""
from __future__ import annotations

import json
import os
import shutil
import sys
import tempfile

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np

from agentic_gts.core.models import OrientedBox, Scene


def _tiny_png(path, color=(200, 30, 30)):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    arr = np.zeros((40, 60, 3))
    arr[..., 0], arr[..., 1], arr[..., 2] = [c / 255 for c in color]
    plt.imsave(path, arr)


def test_build_report_groups_verdicts_per_box():
    from agentic_gts.output.report import build_report

    rng = np.random.default_rng(11)
    pts = rng.uniform(-2, 2, (500, 3))
    scene = Scene(points=pts)
    b1 = OrientedBox(center=(0, 0, 1), size=(0.6, 1.1, 2.0), yaw=0.1)
    b2 = OrientedBox(center=(1.5, 0, 1), size=(0.6, 1.1, 2.0), yaw=0.1)
    scene.boxes = [b1, b2]

    out = tempfile.mkdtemp(prefix="report_")
    try:
        scene.save_boxes(os.path.join(out, "boxes.json"))
        # evidence images: a fit image for box1, a pair image for both boxes
        _tiny_png(os.path.join(out, f"fit_evidence_{b1.box_id[:8]}.png"))
        _tiny_png(os.path.join(out, f"evidence_{b1.box_id[:8]}.png"))
        _tiny_png(os.path.join(out,
                  f"pair_evidence_{b1.box_id[:8]}_{b2.box_id[:8]}.png"))
        # an orphan (deleted-box) record with its image
        _tiny_png(os.path.join(out, "evidence_deadbeef.png"))
        recs = [
            {"kind": "fit", "prompt": "fit question", "answer": '{"dl": 0.1}',
             "choice": "", "confidence": 0.8, "detail": "",
             "image": f"fit_evidence_{b1.box_id[:8]}.png"},
            {"kind": "box", "prompt": "real device?", "answer": "real device",
             "choice": "real device", "confidence": 0.9, "detail": "",
             "image": f"evidence_{b1.box_id[:8]}.png"},
            {"kind": "pair", "prompt": "same rack?", "answer": "separate",
             "choice": "two separate racks", "confidence": 0.7, "detail": "",
             "image": f"pair_evidence_{b1.box_id[:8]}_{b2.box_id[:8]}.png"},
            {"kind": "box", "prompt": "empty?", "answer": "empty space",
             "choice": "empty space", "confidence": 0.9, "detail": "",
             "image": "evidence_deadbeef.png"},
        ]
        with open(os.path.join(out, "vlm_records.jsonl"), "w",
                  encoding="utf-8") as f:
            for r in recs:
                f.write(json.dumps(r) + "\n")

        html_path = build_report(out, points=pts)
        doc = open(html_path, encoding="utf-8").read()
        # both final boxes got their own card with a local view render
        assert b1.box_id[:8] in doc and b2.box_id[:8] in doc
        # local views (2) + verdict evidence: box1 gets 3 (fit/box/pair),
        # box2 gets the pair verdict too, orphan gets 1 -> 7 total. The pair
        # record is deliberately attached to BOTH boxes it is about.
        assert doc.count("data:image/png;base64,") == 7, \
            f"expected 7 inlined images, got {doc.count('data:image/png;base64,')}"
        # verdict content present
        assert "real device" in doc and "two separate racks" in doc
        assert "已删除 / 不在最终结果中的候选" in doc  # orphan section
        assert "deadbeef" in doc
        print("PASS report groups verdicts per box + orphans")
    finally:
        shutil.rmtree(out, ignore_errors=True)


def test_build_report_without_records():
    """A run dir with boxes.json only must still produce a valid report
    (local views, 'no verdicts' notes) -- mock runs are not an error."""
    from agentic_gts.output.report import build_report

    rng = np.random.default_rng(12)
    pts = rng.uniform(-2, 2, (300, 3))
    scene = Scene(points=pts)
    scene.boxes = [OrientedBox(center=(0, 0, 1), size=(0.6, 1.1, 2.0))]
    out = tempfile.mkdtemp(prefix="report2_")
    try:
        scene.save_boxes(os.path.join(out, "boxes.json"))
        html_path = build_report(out, points=pts)
        doc = open(html_path, encoding="utf-8").read()
        assert "没有 VLM 判定记录" in doc
        assert "无判定记录（未触发任何 issue）" in doc
        assert doc.count("data:image/png;base64,") >= 1  # the local view
        assert "已删除 / 不在最终结果中的候选" not in doc  # no orphan section
        print("PASS report handles record-less run dirs")
    finally:
        shutil.rmtree(out, ignore_errors=True)


if __name__ == "__main__":
    test_build_report_groups_verdicts_per_box()
    test_build_report_without_records()
    print("ALL report tests passed")
