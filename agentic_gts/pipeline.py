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
    """Load PLY/PCD/NPY point cloud.

    3DGS exports (PLY with f_dc/opacity/scale/rot) are parsed with our own
    reader and reduced to their Gaussian centers — open3d may silently
    return 0 points for this PLY variant.
    """
    ext = os.path.splitext(path)[1].lower()
    if ext == ".npy":
        return np.load(path)
    if ext == ".ply":
        try:
            from agentic_gts.tools.gs_io import is_gaussian_ply, read_gaussian_ply
            if is_gaussian_ply(path):
                gs = read_gaussian_ply(path)
                print(f"[diag][load] 3DGS ply detected: {len(gs)} gaussians "
                      f"(means used as point cloud; full attrs kept for "
                      f"true-render passes)")
                return gs.means.astype(np.float64)
        except Exception as e:
            print(f"[diag][load] GS ply parse failed ({type(e).__name__}: {e}) "
                  f"-> falling back to open3d")
    import open3d as o3d
    pcd = o3d.io.read_point_cloud(path)
    pts = np.asarray(pcd.points)
    if len(pts) == 0:
        print(f"[diag][load] WARNING: open3d read 0 points from {path} "
              f"(file unreadable / unsupported PLY variant)")
    return pts


def denoise_cloud(points: np.ndarray, nb_neighbors: int = 20,
                  std_ratio: float = 2.0) -> np.ndarray:
    """Stage -1a: statistical outlier removal for 3DGS reconstruction noise.

    3DGS exports contain floaters near the floor and stray splats. They fill
    the z (0.4, 2.5) device band with diffuse mass, inflating every band in
    the yaw histogram (device direction loses contrast) and polluting Stage A
    row detection. Real surfaces are locally dense; isolated noise is not.
    """
    if len(points) < 1000:
        return points
    import open3d as o3d
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(points)
    try:
        pcd, _ = pcd.remove_statistical_outlier(
            nb_neighbors=nb_neighbors, std_ratio=std_ratio)
    except Exception as e:
        print(f"[diag][denoise] SOR failed ({type(e).__name__}) -> keep raw cloud")
        return points
    kept = np.asarray(pcd.points)
    if len(kept) < 100:
        print("[diag][denoise] SOR removed almost everything -> keep raw cloud")
        return points
    print(f"[diag][denoise] SOR: {len(points)} -> {len(kept)} "
          f"(removed {1.0 - len(kept) / len(points):.1%})")
    return kept


def align_to_ground(points: np.ndarray) -> np.ndarray:
    """Stage -1b: level the cloud so the floor plane is horizontal at z=0.

    Two decoupled steps, because in big 3DGS scenes the *largest* horizontal
    plane is often NOT the floor:

      1. TILT correction from the largest horizontal-ish plane (ceiling,
         rack-top field or floor all share the building tilt, so any of
         them gives the up-direction).
      2. Z OFFSET from the largest plane inside the bottom slice of the
         (now level) cloud — that is the actual floor. Using the global
         largest plane here mislabels e.g. a coplanar field of rack tops
         as "floor", shifting the device height band onto the ceiling.

    Falls back to the 2nd z-percentile as floor when no bottom plane fits.
    """
    if len(points) < 100:
        return points
    import open3d as o3d
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(points)

    # ---- phase 1: tilt from the first horizontal-ish dominant plane ----
    # Real floors/ceilings are level within a few degrees; a plane tilted
    # more than ~10 deg is a diagonal RANSAC fit through noise, and rotating
    # by it would smear the floor across the z histogram.
    tilt_n = None
    for attempt in range(6):
        try:
            (a, b, c, d), inliers = pcd.segment_plane(0.05, 3, 1000)
        except Exception as e:
            print(f"[diag][ground] plane fit failed ({type(e).__name__}) -> skip alignment")
            return points
        if c < 0:  # normal must point up
            a, b, c, d = -a, -b, -c, -d
        tilt_deg = math.degrees(math.acos(min(1.0, abs(c))))
        if tilt_deg <= 10.0:
            tilt_n = np.array([a, b, c], dtype=float)
            tilt_n /= float(np.linalg.norm(tilt_n))
            print(f"[diag][ground] tilt reference plane: inliers={len(inliers)} "
                  f"({len(inliers) / len(points):.0%}), tilt={tilt_deg:.1f} deg")
            break
        print(f"[diag][ground] plane #{attempt} too tilted ({tilt_deg:.1f} deg; "
              f"wall or diagonal noise fit) -> excluding, refitting")
        pcd = pcd.select_by_index(inliers, invert=True)
    if tilt_n is None:
        print("[diag][ground] no horizontal plane found -> skip alignment")
        return points

    # Rodrigues rotation mapping the reference normal to +z
    z_axis = np.array([0.0, 0.0, 1.0])
    v = np.cross(tilt_n, z_axis)
    s = float(np.linalg.norm(v))
    if s < 1e-9:
        R = np.eye(3)
    else:
        k = v / s
        K = np.array([[0.0, -k[2], k[1]],
                      [k[2], 0.0, -k[0]],
                      [-k[1], k[0], 0.0]])
        theta = math.atan2(s, float(tilt_n @ z_axis))
        R = np.eye(3) + math.sin(theta) * K + (1.0 - math.cos(theta)) * (K @ K)
    pts = points @ R.T

    # ---- phase 2: floor z from the lowest density spike in the bottom region ----
    # The floor concentrates at one z (a histogram spike); device sides and
    # walls are ~uniform in z and produce no spike. RANSAC in the bottom
    # region is unreliable there (vertical rack faces and smeared diagonal
    # fits win the inlier count), so detect the spike directly: take the
    # lowest smoothed-z-histogram bin that clearly exceeds the regional
    # median density. Falls back to the 2nd percentile when the floor is
    # not reconstructed at all (walls/racks still extend down to it).
    z = pts[:, 2]
    zb = z[z <= float(np.percentile(z, 25))]
    floor_z = None
    if len(zb) > 500:
        edges = np.arange(zb.min(), zb.max() + 0.05, 0.05)
        hist, _ = np.histogram(zb, bins=edges)
        if len(hist) >= 3:
            ext = np.concatenate([[hist[0]], hist, [hist[-1]]])
            sm = (ext[:-2] + 2.0 * ext[1:-1] + ext[2:]) / 4.0
            pos = sm[sm > 0]
            if len(pos):
                thr = max(50.0, 2.0 * float(np.median(pos)))
                # strongest spike, NOT the lowest one: 3DGS floaters under
                # the floor form marginal low bins that win a bottom-up
                # first-above-threshold scan and shift the whole cloud up
                low_i = int(np.argmax(sm)) if sm.max() >= thr else None
                if low_i is not None:
                    zc = float(edges[low_i])
                    near = zb[(zb >= zc - 0.05) & (zb <= zc + 0.15)]
                    if len(near) > 50:
                        floor_z = float(np.median(near))
                        print(f"[diag][ground] floor spike: z={floor_z:.2f} "
                              f"(bin count {int(sm[low_i])}, thr={thr:.0f})")
    if floor_z is None:
        floor_z = float(np.percentile(z, 2))
        print(f"[diag][ground] no floor spike found -> using z p2={floor_z:.2f} as floor")
    pts[:, 2] -= floor_z
    print(f"[diag][ground] aligned: tilt corrected, floor set to z=0 "
          f"(shift={-floor_z:+.2f} m)")
    return pts


def diag_point_cloud(points: np.ndarray) -> None:
    """Print stats to diagnose coordinate-system / scale / density problems."""
    if len(points) == 0:
        print("[diag][cloud] empty point cloud!")
        return
    lo, hi = points.min(axis=0), points.max(axis=0)
    span = hi - lo
    z = points[:, 2]
    q01, q10, q50, q90, q99 = np.percentile(z, [1, 10, 50, 90, 99])
    dev = ((z > 0.5) & (z < 2.4)).mean()
    n_vox = len(np.unique(np.floor(points[:, :2] / 0.1).astype(np.int64), axis=0))
    print(f"[diag][cloud] n={len(points)}  span={span.round(2)}  "
          f"bbox min={lo.round(2)} max={hi.round(2)}")
    print(f"[diag][cloud] z pct 1/10/50/90/99 = {q01:.2f}/{q10:.2f}/{q50:.2f}/"
          f"{q90:.2f}/{q99:.2f}  |  frac in (0.5,2.4) = {dev:.1%}  |  "
          f"0.1m xy voxels = {n_vox}")
    if span.max() > 100 or 0 < span.min() < 0.5:
        print("[diag][cloud] WARNING: span not meter-scale? (machine room expect 5~30m)")
    if abs(q10) > 1.0:
        print("[diag][cloud] WARNING: floor (z p10) far from 0 -> ground not at z~0")
    if dev < 0.05:
        print("[diag][cloud] WARNING: almost no points in device band (0.5,2.4) -> z-axis/scale suspect")


def _diag_support(scene: Scene) -> None:
    if not scene.boxes:
        return
    from agentic_gts.tools import geometry as geo
    sups = sorted(geo.support_fraction(scene, b) for b in scene.boxes)
    n = len(sups)
    print(f"[diag][support] n={n}  min={sups[0]:.2f}  med={sups[n // 2]:.2f}  max={sups[-1]:.2f}")


def _render_stage(scene: Scene, tag: str, out_dir: str,
                  gt_boxes: list[OrientedBox] | None = None) -> None:
    """Save a top-down overlay PNG of the current scene state (per-stage QA).

    Rendered after every pipeline stage so regressions localize at a glance:
    stage0_align_yaw -> stageA_coarse -> stageB_rules -> stageC_agent.
    """
    try:
        from agentic_gts.output.visualize import overlay_topdown
        path = os.path.join(out_dir, f"{tag}.png")
        with open(path, "wb") as f:
            f.write(overlay_topdown(scene, gt_boxes=gt_boxes, title=tag))
        print(f"[viz] {tag} -> {path}")
    except Exception as e:  # stage renders must never break the pipeline
        print(f"[warn] stage render failed ({tag}): {type(e).__name__}: {e}")


def run_pipeline(scene: Scene,
                 gt_boxes: list[OrientedBox] | None = None,
                 use_coarse_seg: bool = True,
                 vlm_backend: str = "mock",
                 vlm_api_base: str | None = None,
                 vlm_model: str | None = None,
                 opts: dict | None = None,
                 out_dir: str = "runs/latest",
                 edge_threshold_m: float = 0.05) -> PipelineResult:
    """Run stages A -> B -> C, render outputs, and (optionally) evaluate."""
    opts = opts or {}
    os.makedirs(out_dir, exist_ok=True)
    evals: dict = {}
    t0 = time.time()
    diag_point_cloud(scene.points)

    # --- stage 0: dominant orientation estimation ---
    # The pipeline reasons in a row-aligned frame. If the caller didn't pin a
    # yaw (opts or scene.meta), estimate it from the point cloud so arbitrary
    # oriented scans work (no axis-aligned assumption).
    if "yaw" not in opts and "yaw" not in scene.meta:
        from agentic_gts.segment.orientation import estimate_yaw_detailed
        info = estimate_yaw_detailed(scene.points)
        yaw = info["yaw"]
        scene.meta["yaw"] = yaw
        print(f"[stage0] estimated dominant yaw = {math.degrees(yaw):.1f} deg")
        try:
            from agentic_gts.output.visualize import render_yaw_diagnosis
            png = os.path.join(out_dir, "yaw_check.png")
            render_yaw_diagnosis(info["device_pts"], info["candidates"], yaw, png)
            print(f"[stage0] yaw diagnosis -> {png}")
        except Exception as e:  # diagnosis render must never break the run
            print(f"[warn] yaw diagnosis render failed: {type(e).__name__}: {e}")
    opts.setdefault("yaw", float(scene.meta.get("yaw", 0.0)))
    _render_stage(scene, "stage0_align_yaw", out_dir, gt_boxes)

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
    _diag_support(scene)
    _eval("stageA")
    _render_stage(scene, "stageA_coarse", out_dir, gt_boxes)

    # --- stage B: deterministic rules ---
    _, issues = apply_rules(scene, opts)
    print(f"[stageB] rules applied -> {len(scene.boxes)} boxes, {len(issues)} issues noted")
    _diag_support(scene)
    _eval("stageB")
    _render_stage(scene, "stageB_rules", out_dir, gt_boxes)

    # --- stage C: agent loop ---
    judge = VLMJudge(backend=vlm_backend, api_base=vlm_api_base,
                     model=vlm_model)
    # record every adjudication (prompt + answer + choice + confidence) to a
    # JSONL so the user can audit why the agent decided each issue
    try:
        judge.set_record(os.path.join(out_dir, "vlm_records.jsonl"))
    except Exception as e:
        print(f"[warn] record path set failed ({type(e).__name__}: {e})")
    agent = LayoutAgent(judge=judge, opts=opts, out_dir=out_dir)
    report = agent.run(scene)
    n_res = len(report.resolved)
    n_unres = len(report.unresolved)
    print(f"[stageC] agent loop -> {n_res} issues resolved, {n_unres} flagged for human review")
    _diag_support(scene)
    _eval("stageC")
    _render_stage(scene, "stageC_agent", out_dir, gt_boxes)

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
