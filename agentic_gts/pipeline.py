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
    the yaw histogram (device direction loses contrast) and polluting the
    grounding fit. Real surfaces are locally dense; isolated noise is not.
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
    # Pin open3d's global RNG before the RANSAC loop: segment_plane
    # draws random triplets, and on a knife-edged scene (floor partly
    # occluded by racks, several near-coplanar surfaces) different
    # samples find slightly different consensus planes -- a hundredth
    # of a degree of tilt flips a few hundred points across stage0's
    # z-band edges, and the yaw candidate scores are near-tied, so the
    # WINNER flips run to run on the SAME input (user logs: two mesh
    # runs, z-band 1953905 vs 1954303, yaw right once / wrong once).
    o3d.utility.random.seed(0)
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
    stage0_align_yaw -> stageG_ground -> stageC_agent.
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
                 vlm_backend: str = "mock",
                 vlm_api_base: str | None = None,
                 vlm_model: str | None = None,
                 vlm_thinking_model: str | None = None,
                 vlm_thinking_base: str | None = None,
                 opts: dict | None = None,
                 out_dir: str = "runs/latest",
                 edge_threshold_m: float = 0.05) -> PipelineResult:
    """Run the unified no-hint flow, render outputs, and (optionally) evaluate.

    stage0 (yaw + layout bootstrap) -> stageG (global nadir VLM 2D
    grounding) -> stageC (per-box local refine). There is no hint-box
    input anymore: boxes come ONLY from the VLM grounding.
    """
    opts = opts or {}
    os.makedirs(out_dir, exist_ok=True)
    evals: dict = {}
    t0 = time.time()
    diag_point_cloud(scene.points)

    # --- stage 0: dominant orientation + layout bootstrap ---
    # The pipeline reasons in a row-aligned frame. The detailed yaw pass
    # ALWAYS runs: besides the yaw, its byproducts (vertical-surface
    # filter + boundary-cell removal) isolate the device layout --
    # device_footprint (framing) and z_top (ceiling cut) -- which the
    # grounding needs (there are no hint boxes to take them from). A
    # caller-pinned yaw overrides only the ANGLE; the byproducts are
    # yaw-independent.
    from agentic_gts.segment.orientation import estimate_yaw_detailed
    info = estimate_yaw_detailed(scene.points)
    yaw_suspect = False
    if "yaw" in opts:
        yaw = float(opts["yaw"])
        print(f"[stage0] yaw pinned by caller: {math.degrees(yaw):.1f} deg "
              f"(estimation used for layout bootstrap only)")
    else:
        if "yaw" in scene.meta:
            yaw = float(scene.meta["yaw"])
        else:
            yaw = info["yaw"]
            print(f"[stage0] estimated dominant yaw = {math.degrees(yaw):.1f} deg")
        # --- residual self-check: closed-loop hijack detector ---
        # Rotate the cloud by -yaw and re-run the SAME estimator: a
        # correct yaw leaves the rows axis-aligned (residual ~0); a
        # hijacked one (wall / sloped floor pulling the histogram
        # peak) leaves the TRUE rows tilted by the error angle, and
        # the re-estimate returns it. The grounded nadir view is the
        # pipeline's only box producer -- a tilted render makes the
        # VLM draw AABBs over skewed rows, one rect swallowing
        # several neighbouring devices (user report from a scene
        # whose groundview rows were visibly not axis-aligned).
        # Curved OUTER walls do not trip this: their energy spreads
        # evenly over the angle histogram (a floor, not a competing
        # peak). A residual that still exceeds the gate after ONE
        # correction means a genuinely multi-directional layout --
        # warned about, not looped on.
        from agentic_gts.segment.orientation import estimate_residual_yaw
        yaw_suspect = False
        res = estimate_residual_yaw(scene.points, yaw)
        if abs(res) > math.radians(5.0):
            fixed = math.remainder(yaw + res, math.pi / 2)
            if fixed >= math.pi / 4:
                fixed -= math.pi / 2
            elif fixed < -math.pi / 4:
                fixed += math.pi / 2
            print(f"[stage0] residual self-check FAILED "
                  f"(residual {math.degrees(res):.1f} deg) -> "
                  f"yaw corrected {math.degrees(yaw):.1f} -> "
                  f"{math.degrees(fixed):.1f} deg")
            yaw = fixed
            res2 = estimate_residual_yaw(scene.points, yaw)
            if abs(res2) > math.radians(5.0):
                print(f"[stage0] WARNING: residual still "
                      f"{math.degrees(res2):.1f} deg after correction -- "
                      f"multi-directional layout? verify yaw_check.png "
                      f"(kept the corrected yaw; stageG will arbitrate "
                      f"the top candidates by grounding yield)")
                # knife-edged scene flag: the estimator's candidate
                # scores are near-tied and the residual chain
                # oscillates between basins (user logs: correction
                # landed on the truth once, 8 deg off the other time)
                yaw_suspect = True
        else:
            print(f"[stage0] residual self-check passed "
                  f"(residual {math.degrees(res):.1f} deg)")
    scene.meta["yaw"] = yaw
    if info.get("z_top") is not None:
        scene.meta["z_top"] = info["z_top"]
    if info.get("device_footprint") is not None:
        scene.meta["device_footprint"] = info["device_footprint"]
    if info.get("device_cells") is not None:
        scene.meta["device_cells"] = info["device_cells"]
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

    # --- stage G: VLM 2D grounding (the ONLY box producer) ---
    # The VLM outlines every device structure on a top-down nadir view
    # (a joined row = ONE region), geometry turns each region into a
    # full-depth row box: the region IS the whole device extent, so the
    # thin-fragment problem never arises. Failure leaves the scene
    # empty (no fallback boxes exist without hint input).
    judge = VLMJudge(backend=vlm_backend, api_base=vlm_api_base,
                     model=vlm_model,
                     thinking_model=vlm_thinking_model,
                     thinking_api_base=vlm_thinking_base)
    # record every adjudication (prompt + answer + choice + confidence)
    # to a JSONL so the user can audit why the agent decided each issue
    try:
        judge.set_record(os.path.join(out_dir, "vlm_records.jsonl"))
    except Exception as e:
        print(f"[warn] record path set failed ({type(e).__name__}: {e}")
    from agentic_gts.agent.ground import ground_stage
    # --- knife-edged yaw: arbitrate the top candidates by GROUNDING
    # YIELD (user logs: same mesh, one run 5 regions at the true yaw,
    # the next 2 regions at a wrong one; the candidate scores were
    # near-tied, the truth sat at #2-3 by score and never won the
    # argmax, and the blind residual correction landed on the truth
    # once and 8 deg off the other time). The estimator alone cannot
    # break the tie -- but the grounding CAN: render + ground at each
    # top candidate direction, and let the EVIDENCE pick --
    #   1. the fitted boxes' OWN directions (per-seed PCA, the
    #      seed_axis_delta measurement) must AGREE with the render
    #      yaw: at the true yaw the rows come out axis-aligned and the
    #      votes cluster AT it; at every wrong yaw the boxes still
    #      physically point wherever the rows are, so the votes carry
    #      the ERROR angle -- agreement is unique to the truth;
    #   2. most boxes wins among agreeing trials (a straight view
    #      detects more structures than a skewed one: 5 vs 2 in the
    #      user's logs).
    # Fires when the residual chain failed twice OR the top-2 folded
    # candidates are NEAR-TIED (ratio >= 0.85): the self-check measures
    # consistency, not correctness -- a wrong yaw backed by a REAL
    # structure at that direction (a wall, a sub-layout) re-aligns
    # that structure and PASSES (user run 3: yaw -26.7 over a genuine
    # -26.5 structure, residual 0.7, 2 boxes instead of the true
    # yaw's 5). Unpinned yaw only -- stable scenes pay nothing.
    from agentic_gts.segment.orientation import yaw_arbitration_needed
    arbitrated = False
    if ("yaw" not in opts and info.get("candidates")
            and yaw_arbitration_needed(info, yaw_suspect)):
        import shutil
        from agentic_gts.segment.orientation import (pick_yaw_trial,
                                                     seed_axis_delta,
                                                     top_yaw_candidates)
        trials = []
        tried = []
        for cy in [w for w, _s in top_yaw_candidates(info, k=3)] \
                + [float(scene.meta["yaw"])]:
            if any(abs(cy - t) < math.radians(2.0) for t in tried):
                continue                      # same direction, tried
            tried.append(cy)
            scene.meta["yaw"] = cy
            tdir = os.path.join(out_dir,
                                "yaw_trial_%+d" % round(math.degrees(cy)))
            os.makedirs(tdir, exist_ok=True)
            ok = ground_stage(scene, judge, tdir)
            n_boxes = len(scene.boxes) if ok else 0
            d = None
            if ok and len(scene.boxes) >= 2:
                d = seed_axis_delta(
                    scene.boxes, scene.points, cy,
                    top_cut=(float(scene.meta.get("z_top", 2.5) or 2.5)
                             + 0.10))
            print(f"[stageG] yaw trial {math.degrees(cy):+.1f} deg: "
                  f"{n_boxes} boxes, seed-axis delta "
                  + ("none" if d is None
                     else f"{math.degrees(d):+.1f} deg"))
            trials.append({"yaw": cy, "n": n_boxes, "delta": d,
                           "boxes": list(scene.boxes) if ok else []})
        win = pick_yaw_trial(trials)
        if win is not None and win["n"] > 0:
            arbitrated = True
            scene.meta["yaw"] = win["yaw"]
            scene.boxes = win["boxes"]
            print(f"[stageG] yaw arbitration -> "
                  f"{math.degrees(win['yaw']):+.1f} deg "
                  f"({win['n']} boxes)")
            # promote the winner's audit renders to the run root --
            # the trials each wrote their own subdir (no clobbering),
            # and the before/after comparison the user reads expects
            # groundview.png / grounded.png at the root
            for fn in ("groundview.png", "grounded.png",
                       "cluster_check.png"):
                src = os.path.join(
                    out_dir,
                    "yaw_trial_%+d" % round(math.degrees(win["yaw"])),
                    fn)
                if os.path.exists(src):
                    try:
                        shutil.copy2(src, os.path.join(out_dir, fn))
                    except OSError:
                        pass
            _diag_support(scene)
            _eval("stageG")
            _render_stage(scene, "stageG_ground", out_dir, gt_boxes)
    if not arbitrated and ground_stage(scene, judge, out_dir):
        # NOTE: the row SPLIT no longer runs here -- it moved into
        # the agent loop, AFTER the per-box local refinement (SAM).
        # User-directed order: grounding -> refine each region ->
        # split the joined rows. Pre-splitting decided structure
        # membership before the refinement evidence had a vote.
        _diag_support(scene)
        _eval("stageG")
        _render_stage(scene, "stageG_ground", out_dir, gt_boxes)

        # --- grounding feedback: yaw from the seeds' OWN directions ---
        # The pre-render residual self-check only sees what stage0's
        # estimator sees (the WHOLE device band -- hijack soil). The
        # fitted seeds are the purer evidence: each grounded box's
        # direction is MEASURED by PCA on the device-band points
        # inside it (the boxes themselves are axis-aligned in the row
        # frame, so their yaw carries no information), and the votes'
        # weighted median is the layout's true direction -- local
        # per-structure, no histogram to hijack, one stray wall-ish
        # fit cannot drag it. The earlier pool version re-ran the
        # GLOBAL estimator on the union of the boxes' points, a pool
        # carved along the ASSUMED yaw: slanted rows re-confirmed the
        # assumed yaw and residual haze kept stage0's hijack surface
        # (user report: wrong yaw after the feedback). If the median
        # disagrees with the render yaw, the groundview was tilted
        # and the VLM's AABBs over skewed rows are unreliable ->
        # correct the yaw and re-ground ONCE. Never fires when the
        # render was already straight.
        if "yaw" not in opts and len(scene.boxes) >= 2:
            from agentic_gts.segment.orientation import seed_axis_delta
            # top cut = the FIT pool's own (z_top + 0.10): the vote
            # pool must match what the seed fit measured against, or
            # tray / ceiling remnants above the devices pull the PCA
            delta = seed_axis_delta(
                scene.boxes, scene.points, float(scene.meta["yaw"]),
                top_cut=(float(scene.meta.get("z_top", 2.5) or 2.5)
                         + 0.10))
            if delta is not None and abs(delta) > math.radians(3.0):
                new_yaw = math.remainder(
                    float(scene.meta["yaw"]) + delta, math.pi / 2)
                if new_yaw >= math.pi / 4:
                    new_yaw -= math.pi / 2
                elif new_yaw < -math.pi / 4:
                    new_yaw += math.pi / 2
                print(f"[stageG] grounding feedback: yaw "
                      f"{math.degrees(float(scene.meta['yaw'])):.1f} -> "
                      f"{math.degrees(new_yaw):.1f} deg "
                      f"(delta {math.degrees(delta):.1f}, per-seed PCA "
                      f"weighted median) -> re-rendering + re-grounding")
                scene.meta["yaw"] = new_yaw
                opts["yaw"] = new_yaw
                if ground_stage(scene, judge, out_dir):
                    _diag_support(scene)
                    _eval("stageG_reground")
                    _render_stage(scene, "stageG_ground", out_dir,
                                  gt_boxes)

    # --- stage C: agent loop (per-box local refine) ---
    agent = LayoutAgent(judge=judge, opts=opts, out_dir=out_dir)
    report = agent.run(scene)
    n_res = len(report.resolved)
    n_unres = len(report.unresolved)
    print(f"[stageC] agent loop -> {n_res} issues resolved, {n_unres} flagged for human review")
    _diag_support(scene)
    _eval("stageC")
    _render_stage(scene, "stageC_agent", out_dir, gt_boxes)

    # --- stage D: row completion (geometry-only recall fallback) ---
    # VLM grounding is the only box producer; a cabinet it missed
    # (occluded in the nadir view, dim, dropped with a poor-quality
    # view) is lost for good without this pass. Walks the fitted
    # rows' interiors and ends with point-support probes -- the old
    # rules' find_gaps/add_box_at job, seeded from grounded boxes.
    from agentic_gts.tools.geometry import complete_row_gaps
    added = complete_row_gaps(scene)
    if added:
        print(f"[stageD] row completion: +{len(added)} point-supported "
              f"fill(s), Confidence.LOW (human review)")
        _diag_support(scene)
        _eval("stageD")
        _render_stage(scene, "stageD_complete", out_dir, gt_boxes)

    # --- final filter: drop LOW-confidence boxes (user directive) ---
    # The only LOW boxes are the stageD geometry-only row completions
    # (no VLM confirmation); too unreliable to keep in the result.
    from agentic_gts.core.models import Confidence
    n_low = sum(1 for b in scene.boxes if b.confidence == Confidence.LOW)
    if n_low:
        scene.boxes = [b for b in scene.boxes
                       if b.confidence != Confidence.LOW]
        print(f"[out] dropped {n_low} LOW-confidence box(es) "
              f"(geometry-only completions, no VLM confirmation)")

    # --- outputs ---
    scene.save_boxes(os.path.join(out_dir, "boxes.json"))
    # also persist the final layout in the detector-style 'objects' schema
    # (the format the 'view' sub-command accepts as --boxes), so the result
    # feeds downstream viewers directly
    try:
        from agentic_gts.core.models import save_boxes_as_objects
        save_boxes_as_objects(scene.boxes,
                               os.path.join(out_dir, "boxes_objects.json"))
        print(f"[out] {len(scene.boxes)} boxes -> boxes_objects.json "
              f"(input 'objects' format)")
    except Exception as e:  # extra format must never break the run
        print(f"[warn] objects-format save failed: {type(e).__name__}: {e}")
    with open(os.path.join(out_dir, "layout.svg"), "w", encoding="utf-8") as f:
        f.write(boxes_to_svg(scene.boxes, title="Data-center layout"))
    with open(os.path.join(out_dir, "layout.png"), "wb") as f:
        f.write(boxes_to_png(scene.boxes, title="Data-center layout"))
    # point cloud + boxes overlays (2D PNG + merged PLY for 3D viewers)
    try:
        from agentic_gts.output.visualize import (export_boxes_ply,
                                                 export_ply, overlay_topdown)
        with open(os.path.join(out_dir, "overlay.png"), "wb") as f:
            f.write(overlay_topdown(scene, gt_boxes=gt_boxes,
                                    title="point cloud + detected boxes"))
        # colored cloud (3DGS true colors when available) + boxes
        export_ply(scene, os.path.join(out_dir, "cloud_with_boxes.ply"),
                   gt_boxes=gt_boxes, gs_ply=scene.meta.get("gs_ply"))
        # boxes-only artifact: the layout reads clearly without the cloud
        export_boxes_ply(scene, os.path.join(out_dir, "boxes_only.ply"),
                         gt_boxes=gt_boxes)
    except Exception as e:  # visualization must never break the pipeline
        print(f"[warn] visualization failed: {type(e).__name__}: {e}")
    with open(os.path.join(out_dir, "agent_report.json"), "w", encoding="utf-8") as f:
        json.dump(report.to_dict(), f, ensure_ascii=False, indent=2)
    # per-box local-view + VLM-verdict browsable report (self-contained HTML):
    # fresh local render for every final box, verdicts from vlm_records.jsonl
    try:
        from agentic_gts.output.report import build_report
        html_path = build_report(out_dir, points=scene.points,
                                 gs_ply=scene.meta.get("gs_ply"))
        print(f"[out] VLM verdict report -> {html_path}")
    except Exception as e:  # report must never break the pipeline
        print(f"[warn] VLM report failed: {type(e).__name__}: {e}")
    if evals:
        with open(os.path.join(out_dir, "eval.json"), "w", encoding="utf-8") as f:
            json.dump(evals, f, ensure_ascii=False, indent=2)

    print(f"[done] {time.time()-t0:.1f}s -> outputs in {out_dir}")
    return PipelineResult(scene=scene, stage_evals=evals,
                          agent_report=report.to_dict(), out_dir=out_dir)
