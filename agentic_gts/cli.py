"""CLI entry point.

Usage:
  python -m agentic_gts.cli synth --seed 42 --out runs/synth1
  python -m agentic_gts.cli run   --point-cloud path.ply [--out runs/x]
  python -m agentic_gts.cli run   --point-cloud path.ply --boxes boxes.json
  python -m agentic_gts.cli demo  --out runs/demo      # synth + full pipeline + eval
"""
from __future__ import annotations

import argparse
import json

import numpy as np

from agentic_gts.core.models import OrientedBox, Scene
from agentic_gts.pipeline import load_point_cloud, run_pipeline
from agentic_gts.synth.generator import SynthConfig, generate


def _load_boxes(path: str, scene: Scene) -> None:
    scene.load_boxes(path)


def cmd_synth(args):
    cfg = SynthConfig(seed=args.seed)
    scene, gt, corrupt = generate(cfg)
    import os
    os.makedirs(args.out, exist_ok=True)
    np.save(f"{args.out}/points.npy", scene.points)
    scene.save_boxes(f"{args.out}/corrupted_boxes.json")
    s2 = Scene(points=scene.points, boxes=gt)
    s2.save_boxes(f"{args.out}/gt_boxes.json")
    print(f"[synth] n_points={len(scene.points)} gt={len(gt)} corrupted={len(corrupt)}")


def cmd_run(args):
    pts = load_point_cloud(args.point_cloud)
    if args.boxes or args.gt:
        # external boxes share the cloud's coordinate frame; transforming the
        # cloud alone would desynchronize them. Caller must pre-align.
        print("[diag][ground] external boxes/gt given -> skipping auto ground alignment")
    else:
        from agentic_gts.pipeline import align_to_ground, denoise_cloud
        pts = denoise_cloud(pts)
        pts = align_to_ground(pts)
    scene = Scene(points=pts)
    if args.boxes:
        scene.load_boxes(args.boxes)
    gt_boxes = None
    if args.gt:
        gs = Scene(points=scene.points)
        gs.load_boxes(args.gt)
        gt_boxes = gs.boxes
    opts = {}
    if args.yaw is not None:
        import math as _math
        opts["yaw"] = _math.radians(args.yaw)
        print(f"[cli] yaw pinned by user: {args.yaw} deg (estimation skipped)")
    res = run_pipeline(scene, gt_boxes=gt_boxes,
                       use_coarse_seg=not args.boxes,
                       vlm_backend=args.vlm,
                       vlm_api_base=args.vlm_base,
                       opts=opts,
                       out_dir=args.out,
                       edge_threshold_m=args.edge_thr)
    return res


def cmd_demo(args):
    cfg = SynthConfig(seed=args.seed)
    scene, gt, corrupt = generate(cfg)
    res = run_pipeline(scene, gt_boxes=gt, use_coarse_seg=True,
                       vlm_backend=args.vlm, vlm_api_base=args.vlm_base,
                       out_dir=args.out, edge_threshold_m=args.edge_thr)
    print(json.dumps(res.stage_evals, ensure_ascii=False, indent=2))


def cmd_diagnose(args):
    """Preprocess a cloud, estimate yaw, render a yaw-diagnosis PNG.

    Use this when the pipeline output looks wrong (e.g. axis-aligned boxes
    on a rotated room): the PNG shows the device-band points with all
    candidate yaw arrows and the chosen one, so a hijacked estimate is
    visible at a glance.
    """
    import os
    from agentic_gts.pipeline import align_to_ground, denoise_cloud
    from agentic_gts.segment.orientation import estimate_yaw_detailed
    from agentic_gts.output.visualize import render_yaw_diagnosis

    pts = load_point_cloud(args.point_cloud)
    if len(pts) == 0:
        print("[diagnose] empty point cloud, nothing to do")
        return
    os.makedirs(args.out, exist_ok=True)
    pts = denoise_cloud(pts)
    pts = align_to_ground(pts)
    info = estimate_yaw_detailed(pts)
    png = os.path.join(args.out, "yaw_check.png")
    render_yaw_diagnosis(info["device_pts"], info["candidates"], info["yaw"], png)
    print(f"[diagnose] chosen yaw = {__import__('math').degrees(info['yaw']):.1f} deg")
    print(f"[diagnose] visualization -> {png}")
    print("[diagnose] check: does the RED arrow follow your device rows?")


def cmd_view(args):
    """Open the 3D interactive viewer: point cloud + wireframe boxes."""
    from agentic_gts.output.visualize import view_3d
    scene = Scene(points=load_point_cloud(args.point_cloud))
    if args.boxes:
        scene.load_boxes(args.boxes)
    gt_boxes = None
    if args.gt:
        gs = Scene(points=scene.points)
        gs.load_boxes(args.gt)
        gt_boxes = gs.boxes
    view_3d(scene, gt_boxes=gt_boxes)


def main():
    p = argparse.ArgumentParser(prog="agentic-gts")
    sub = p.add_subparsers(dest="cmd", required=True)

    s = sub.add_parser("synth", help="generate synthetic machine room data")
    s.add_argument("--seed", type=int, default=42)
    s.add_argument("--out", default="runs/synth")
    s.set_defaults(fn=cmd_synth)

    r = sub.add_parser("run", help="run pipeline on point cloud")
    r.add_argument("--point-cloud", required=True)
    r.add_argument("--boxes", default=None, help="optional pre-detected boxes json")
    r.add_argument("--gt", default=None, help="optional ground-truth boxes json")
    r.add_argument("--out", default="runs/latest")
    r.add_argument("--vlm", default="mock", choices=["mock", "qwen"])
    r.add_argument("--vlm-base", default=None)
    r.add_argument("--edge-thr", type=float, default=0.05)
    r.add_argument("--yaw", type=float, default=None,
                   help="pin device row yaw in degrees (skips estimation)")
    r.set_defaults(fn=cmd_run)

    d = sub.add_parser("demo", help="synthetic data -> full pipeline -> eval")
    d.add_argument("--seed", type=int, default=42)
    d.add_argument("--out", default="runs/demo")
    d.add_argument("--vlm", default="mock", choices=["mock", "qwen"])
    d.add_argument("--vlm-base", default=None)
    d.add_argument("--edge-thr", type=float, default=0.05)
    d.set_defaults(fn=cmd_demo)

    g = sub.add_parser("diagnose", help="preprocess + yaw check visualization")
    g.add_argument("--point-cloud", required=True)
    g.add_argument("--out", default="runs/diag")
    g.set_defaults(fn=cmd_diagnose)

    v = sub.add_parser("view", help="open 3D viewer: cloud + boxes")
    v.add_argument("--point-cloud", required=True)
    v.add_argument("--boxes", default=None)
    v.add_argument("--gt", default=None)
    v.set_defaults(fn=cmd_view)

    args = p.parse_args()
    args.fn(args)


if __name__ == "__main__":
    main()
