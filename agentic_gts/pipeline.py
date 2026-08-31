"""End-to-end pipeline orchestration."""
from __future__ import annotations

import json
import math
import os
import time
from dataclasses import dataclass

import numpy as np

from agentic_gts.core.models import OrientedBox, Scene
from agentic_gts.agent.judge import VLMJudge
from agentic_gts.agent.loop import LayoutAgent
from agentic_gts.eval.metrics import EvalResult, evaluate
from agentic_gts.output.render import boxes_to_png, boxes_to_svg
from agentic_gts.rules.rules import apply_rules
from agentic_gts.segment.coarse import coarse_segment


@dataclass
class PipelineResult:
    scene: Scene
    stage_evals: dict
    agent_report: dict
    out_dir: str


def load_point_cloud(path: str) -> np.ndarray:
    """Load PLY/PCD/NPY point cloud."""
    ext = os.path.splitext(path)[1].lower()
    if ext == ".npy":
        return np.load(path)
    import open3d as o3d
    pcd = o3d.io.read_point_cloud(path)
    return np.asarray(pcd.points)


def run_pipeline(scene: Scene,
                 gt_boxes: list[OrientedBox] | None = None,
                 use_coarse_seg: bool = True,
                 vlm_backend: str = "mock",
                 vlm_api_base: str | None = None,
                 opts: dict | None = None,
                 out_dir: str = "runs/latest",
                 edge_threshold_m: float = 0.05) -> PipelineResult:
    """Run stages A -> B -> C, render outputs, and (optionally) evaluate."""
    opts = opts or {}
    os.makedirs(out_dir, exist_ok=True)
    evals: dict = {}
    t0 = time.time()

    # --- stage 0: dominant orientation estimation ---
    # The pipeline reasons in a row-aligned frame. If the caller didn't pin a
    # yaw (opts or scene.meta), estimate it from the point cloud so arbitrary
    # oriented scans work (no axis-aligned assumption).
    if "yaw" not in opts and "yaw" not in scene.meta:
        from agentic_gts.segment.orientation import estimate_yaw
        yaw = estimate_yaw(scene.points)
        scene.meta["yaw"] = yaw
        print(f"[stage0] estimated dominant yaw = {math.degrees(yaw):.1f} deg")
    opts.setdefault("yaw", float(scene.meta.get("yaw", 0.0)))

    def _eval(tag: str):
        if gt_boxes is not None:
            r = evaluate(scene.boxes, gt_boxes, edge_threshold_m=edge_threshold_m)
            evals[tag] = r.to_dict()
            print(f"[{tag}] {r.summary()}")

    # --- stage A: coarse segmentation (optional; skip if boxes given) ---
    if use_coarse_seg:
        coarse_segment(scene, opts)
        print(f"[stageA] coarse segmentation -> {len(scene.boxes)} candidate boxes")
    else:
        print(f"[stageA] skipped (using {len(scene.boxes)} provided boxes)")
    _eval("stageA")

    # --- stage B: deterministic rules ---
    _, issues = apply_rules(scene, opts)
    print(f"[stageB] rules applied -> {len(scene.boxes)} boxes, {len(issues)} issues noted")
    _eval("stageB")

    # --- stage C: agent loop ---
    judge = VLMJudge(backend=vlm_backend, api_base=vlm_api_base)
    agent = LayoutAgent(judge=judge, opts=opts)
    report = agent.run(scene)
    n_res = len(report.resolved)
    n_unres = len(report.unresolved)
    print(f"[stageC] agent loop -> {n_res} issues resolved, {n_unres} flagged for human review")
    _eval("stageC")

    # --- outputs ---
    scene.save_boxes(os.path.join(out_dir, "boxes.json"))
    with open(os.path.join(out_dir, "layout.svg"), "w", encoding="utf-8") as f:
        f.write(boxes_to_svg(scene.boxes, title="Data-center layout"))
    with open(os.path.join(out_dir, "layout.png"), "wb") as f:
        f.write(boxes_to_png(scene.boxes, title="Data-center layout"))
    # point cloud + boxes overlays (2D PNG + merged PLY for 3D viewers)
    try:
        from agentic_gts.output.visualize import export_ply, overlay_topdown
        with open(os.path.join(out_dir, "overlay.png"), "wb") as f:
            f.write(overlay_topdown(scene, gt_boxes=gt_boxes,
                                    title="point cloud + detected boxes"))
        export_ply(scene, os.path.join(out_dir, "cloud_with_boxes.ply"),
                   gt_boxes=gt_boxes)
    except Exception as e:  # visualization must never break the pipeline
        print(f"[warn] visualization failed: {type(e).__name__}: {e}")
    with open(os.path.join(out_dir, "agent_report.json"), "w", encoding="utf-8") as f:
        json.dump(report.to_dict(), f, ensure_ascii=False, indent=2)
    if evals:
        with open(os.path.join(out_dir, "eval.json"), "w", encoding="utf-8") as f:
            json.dump(evals, f, ensure_ascii=False, indent=2)

    print(f"[done] {time.time()-t0:.1f}s -> outputs in {out_dir}")
    return PipelineResult(scene=scene, stage_evals=evals,
                          agent_report=report.to_dict(), out_dir=out_dir)
