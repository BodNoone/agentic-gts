"""CLI entry point.

Usage:
  python -m agentic_gts.cli synth --seed 42 --out runs/synth1
  python -m agentic_gts.cli run   --point-cloud gs.ply [--mesh-cloud mesh.ply] [--out runs/x]
"""
from __future__ import annotations

import argparse

import numpy as np

from agentic_gts.core.models import Scene
from agentic_gts.pipeline import load_point_cloud, run_pipeline
from agentic_gts.synth.generator import SynthConfig, generate


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
    # geometry source: the mesh-discretized cloud when given, else the
    # --point-cloud input itself (GS centers for a gaussian ply). The
    # mesh is coordinate-aligned with the 3DGS by contract, and 3DGS
    # renders well but measures poorly (haze, floaters, sparse zones)
    # while a mesh sampling is geometrically exact -- so every stage
    # that MEASURES (yaw/bootstrap, region fits, column heights,
    # thickness fallbacks) runs on the mesh, and every stage that
    # RENDERS (groundview, local evidence views) still splats the GS.
    mesh = getattr(args, "mesh_cloud", None)
    pts = load_point_cloud(mesh) if mesh else load_point_cloud(args.point_cloud)
    if mesh:
        print(f"[cli] mesh cloud given ({len(pts)} pts): geometry stages "
              f"run on the mesh, rendering stays 3DGS")
        scene_is_mesh = True
    if args.gt:
        # ground-truth boxes share the cloud's coordinate frame; transforming
        # the cloud alone would desynchronize them. Caller must pre-align.
        print("[diag][ground] gt boxes given -> skipping auto ground alignment")
    else:
        from agentic_gts.pipeline import align_to_ground, denoise_cloud
        pts = denoise_cloud(pts)
        pts = align_to_ground(pts)
    scene = Scene(points=pts)
    # geometry-source flag: a mesh sampling has no haze / floaters /
    # under-floor diffusion -- every "robust" estimator downstream can
    # take its simple (min / percentile) form when this is set
    if mesh:
        scene.meta["geometry_is_mesh"] = True
    # remember the 3DGS source so VLM evidence renders are TRUE splat renders
    # (needs gsplat / diff_gaussian_rasterization at render time; scatter
    # fallback otherwise)
    try:
        from agentic_gts.tools.gs_io import is_gaussian_ply
        if is_gaussian_ply(args.point_cloud):
            scene.meta["gs_ply"] = args.point_cloud
            print("[cli] 3DGS input: god-view / local evidence will be "
                  "rendered via Gaussian splatting")
    except Exception as e:
        print(f"[cli] GS detection failed ({type(e).__name__}: {e})")
    # COLMAP training poses (optional): pose-based render trust
    if getattr(args, "gs_cams", None):
        from agentic_gts.tools.gs_io import read_colmap_views
        tv = read_colmap_views(args.gs_cams)
        if tv is None:
            print(f"[cli] --gs-cams: no images.txt found under "
                  f"{args.gs_cams} -> pose trust disabled")
        else:
            scene.meta["gs_cams"] = args.gs_cams
            print(f"[cli] COLMAP poses loaded: {len(tv[0])} training "
                  f"cameras -> render trust enabled")
    gt_boxes = None
    if args.gt:
        gs = Scene(points=scene.points)
        gs.load_boxes(args.gt)
        gt_boxes = gs.boxes
    opts = {}
    if args.yaw is not None:
        import math as _math
        opts["yaw"] = _math.radians(args.yaw)
        print(f"[cli] yaw pinned by user: {args.yaw} deg")
    rt = getattr(args, "recall_tilts", "auto")
    if rt != "auto":
        opts["recall_tilts"] = (rt == "on")
        print(f"[cli] recall tilt views forced {rt}")
    if getattr(args, "sam_checkpoint", None):
        opts["sam_checkpoint"] = args.sam_checkpoint
        if getattr(args, "sam_model_cfg", None):
            opts["sam_model_cfg"] = args.sam_model_cfg
        print(f"[cli] local SAM mask refinement enabled: {args.sam_checkpoint}")
    res = run_pipeline(scene, gt_boxes=gt_boxes,
                       vlm_backend=args.vlm,
                       vlm_api_base=args.vlm_base,
                       vlm_model=args.vlm_model,
                       vlm_thinking_model=args.vlm_thinking_model,
                       vlm_thinking_base=args.vlm_thinking_base,
                       opts=opts,
                       out_dir=args.out,
                       edge_threshold_m=args.edge_thr)
    return res


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


def cmd_report(args):
    """Build the per-box local-view + VLM-verdict HTML report for a run.

    Works retroactively on any run directory (needs boxes.json; verdicts
    come from vlm_records.jsonl when present). With --point-cloud every
    final box gets a fresh local three-view render; without it only the
    on-disk evidence images are shown.
    """
    import os
    from agentic_gts.pipeline import load_point_cloud
    from agentic_gts.output.report import build_report

    points = None
    gs_ply = None
    if args.point_cloud:
        points = load_point_cloud(args.point_cloud)
        try:
            from agentic_gts.tools.gs_io import is_gaussian_ply
            if is_gaussian_ply(args.point_cloud):
                gs_ply = args.point_cloud
        except Exception:
            pass
    out = build_report(args.run_dir, out_path=args.out,
                       points=points, gs_ply=gs_ply)
    print(f"[report] {out} (open in a browser)")


def main():
    p = argparse.ArgumentParser(prog="agentic-gts")
    sub = p.add_subparsers(dest="cmd", required=True)

    s = sub.add_parser("synth", help="generate synthetic machine room data")
    s.add_argument("--seed", type=int, default=42)
    s.add_argument("--out", default="runs/synth")
    s.set_defaults(fn=cmd_synth)

    r = sub.add_parser("run", help="run pipeline on point cloud")
    r.add_argument("--point-cloud", required=True)
    r.add_argument("--mesh-cloud", default=None, metavar="PATH",
                   help="optional mesh-discretized point cloud, coordinate-"
                        "aligned with the 3DGS: given, every geometry stage "
                        "(yaw/bootstrap, region fits, heights, thickness "
                        "fallbacks) runs on the mesh while rendering stays "
                        "3DGS; omitted, the point cloud itself is the "
                        "geometry source")
    r.add_argument("--gs-cams", default=None, metavar="PATH",
                   help="COLMAP training poses for render-trust scoring: "
                        "the sparse dir (e.g. sparse/0 containing "
                        "cameras.txt + images.txt) or images.txt itself. "
                        "Candidate view scores then blend the distance to "
                        "the trained ray distribution")
    r.add_argument("--gt", default=None, help="optional ground-truth boxes json")
    r.add_argument("--out", default="runs/latest")
    r.add_argument("--vlm", default="mock", choices=["mock", "qwen", "local"])
    r.add_argument("--vlm-base", default=None,
                   help="OpenAI-compatible API base, e.g. http://127.0.0.1:8000/v1 "
                        "(also env VLM_API_BASE)")
    r.add_argument("--vlm-model", default=None,
                   help="served model name (qwen) or local checkpoint dir (local), "
                        "e.g. Qwen/Qwen3-VL-8B-Instruct or /models/qwen3-vl "
                        "(also env VLM_MODEL)")
    r.add_argument("--vlm-thinking-model", default=None,
                   help="optional thinking checkpoint for hard-case escalation, "
                        "e.g. Qwen/Qwen3-VL-8B-Thinking (also env "
                        "VLM_THINKING_MODEL). Low-quality evidence renders are "
                        "re-asked on it; godview audits run on it directly")
    r.add_argument("--vlm-thinking-base", default=None,
                   help="API base for the thinking model if served separately "
                        "(defaults to --vlm-base, also env VLM_THINKING_API_BASE)")
    r.add_argument("--sam-checkpoint", default=None,
                   help="SAM2/SAM checkpoint for local VLM-point + SAM mask "
                        "refinement (also env SAM_CHECKPOINT)")
    r.add_argument("--sam-model-cfg", default=None,
                   help="SAM2 model config (also env SAM_MODEL_CFG); omitted "
                        "for legacy segment-anything")
    r.add_argument("--edge-thr", type=float, default=0.05)
    r.add_argument("--yaw", type=float, default=None,
                   help="pin device row yaw in degrees (skips estimation)")
    r.add_argument("--recall-tilts", default="auto",
                   choices=["auto", "on", "off"],
                   help="recall tilt views: auto = single-view only (tiled "
                        "layouts skip them -- tiling already renders edge "
                        "devices obliquely); on/off force it for A/B tests")
    r.set_defaults(fn=cmd_run)

    g = sub.add_parser("diagnose", help="preprocess + yaw check visualization")
    g.add_argument("--point-cloud", required=True)
    g.add_argument("--out", default="runs/diag")
    g.set_defaults(fn=cmd_diagnose)

    v = sub.add_parser("view", help="open 3D viewer: cloud + boxes")
    v.add_argument("--point-cloud", required=True)
    v.add_argument("--boxes", default=None)
    v.add_argument("--gt", default=None)
    v.set_defaults(fn=cmd_view)

    rp = sub.add_parser("report", help="per-box local view + VLM verdict HTML")
    rp.add_argument("--run-dir", required=True,
                    help="pipeline output dir (needs boxes.json)")
    rp.add_argument("--point-cloud", default=None,
                    help="optional: re-render a fresh local view for every "
                         "final box")
    rp.add_argument("--out", default=None,
                    help="output html path (default <run-dir>/vlm_report.html)")
    rp.set_defaults(fn=cmd_report)

    args = p.parse_args()
    args.fn(args)


if __name__ == "__main__":
    main()
