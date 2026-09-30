"""CLI entry point.

Usage:
  python -m agentic_gts.cli synth --seed 42 --out runs/synth1
  python -m agentic_gts.cli run   --point-cloud gs.ply [--mesh-cloud mesh.ply] [--out runs/x]
"""
from __future__ import annotations

import argparse

from agentic_gts.core.models import Scene
from agentic_gts.pipeline import load_point_cloud, run_pipeline


def cmd_run(args):
    # geometry source: the mesh-discretized cloud when given, else the
    # --point-cloud input itself -- 3DGS renders well but measures
    # poorly, a mesh sampling is geometrically exact (see --mesh-cloud)
    mesh = getattr(args, "mesh_cloud", None)
    pts = load_point_cloud(mesh) if mesh else load_point_cloud(args.point_cloud)
    if mesh:
        print(f"[cli] mesh cloud given ({len(pts)} pts): geometry stages "
              f"run on the mesh, rendering stays 3DGS")
    from agentic_gts.pipeline import align_to_ground, denoise_cloud
    pts = denoise_cloud(pts)
    pts, align_tf = align_to_ground(pts, return_transform=True)
    scene = Scene(points=pts)
    if align_tf is not None:
        # renders/exports read the RAW gaussian file -- they must apply
        # the SAME transform (see apply_align_transform)
        scene.meta["align_tf"] = align_tf
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
    opts = {}
    if args.yaw is not None:
        import math as _math
        opts["yaw"] = _math.radians(args.yaw)
        print(f"[cli] yaw pinned by user: {args.yaw} deg")
    if getattr(args, "sam_checkpoint", None):
        opts["sam_checkpoint"] = args.sam_checkpoint
        if getattr(args, "sam_model_cfg", None):
            opts["sam_model_cfg"] = args.sam_model_cfg
        print(f"[cli] local SAM mask refinement enabled: {args.sam_checkpoint}")
    res = run_pipeline(scene,
                       vlm_backend=args.vlm,
                       vlm_model=args.vlm_model,
                       opts=opts,
                       out_dir=args.out)
    return res


def main():
    p = argparse.ArgumentParser(prog="agentic-gts")
    sub = p.add_subparsers(dest="cmd", required=True)

    r = sub.add_parser("run", help="run pipeline on point cloud")
    r.add_argument("--point-cloud", required=True)
    r.add_argument("--mesh-cloud", default=None, metavar="PATH",
                   help="optional mesh-discretized point cloud, coordinate-"
                        "aligned with the 3DGS: given, every geometry stage "
                        "(yaw/bootstrap, region fits, heights, thickness "
                        "fallbacks) runs on the mesh while rendering stays "
                        "3DGS; omitted, the point cloud itself is the "
                        "geometry source")
    r.add_argument("--out", default="runs/latest")
    r.add_argument("--vlm", default="local", choices=["local"])
    r.add_argument("--vlm-model", default=None,
                   help="served model name (qwen) or local checkpoint dir (local), "
                        "e.g. Qwen/Qwen3-VL-8B-Instruct or /models/qwen3-vl "
                        "(also env VLM_MODEL)")
    r.add_argument("--sam-checkpoint", default=None,
                   help="SAM2/SAM checkpoint for local VLM-point + SAM mask "
                        "refinement (also env SAM_CHECKPOINT)")
    r.add_argument("--sam-model-cfg", default=None,
                   help="SAM2 model config (also env SAM_MODEL_CFG); omitted "
                        "for legacy segment-anything")
    r.add_argument("--yaw", type=float, default=None,
                   help="pin device row yaw in degrees (skips estimation)")
    r.set_defaults(fn=cmd_run)

    args = p.parse_args()
    args.fn(args)


if __name__ == "__main__":
    main()
