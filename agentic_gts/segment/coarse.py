"""Stage A: coarse segmentation -- superpoint over-segmentation + rule merge.

Takes a point cloud, over-segments it into superpoints, then merges neighbors
into candidate device boxes using geometric rules (planarity, height
continuity, row alignment, standard-dimension termination).

High-recall is the goal: we'd rather over-produce candidate boxes and let
stages B/C prune them than miss devices.
"""
from __future__ import annotations

import math

import numpy as np
from scipy.spatial import cKDTree

from agentic_gts.core.models import (
    BoxSource,
    Confidence,
    DeviceType,
    OrientedBox,
    Scene,
)

MERGE_PRIMARY_VOXEL = 0.12


def superpoint_over_segmentation(scene: Scene, voxel: float = MERGE_PRIMARY_VOXEL) \
        -> tuple[np.ndarray, np.ndarray]:
    """Voxelize floor-projected points and return superpoint labels + centroids.

    We only care about candidate *device* points: those above a small height
    (racks are vertical volumes) and off the floor plane. Floor/ground points
    are filtered to keep the row structure clean.
    """
    pts = scene.points
    z = pts[:, 2]
    above = pts[z > 0.05]
    print(f"[diag][A] points z>0.05: {len(above)}/{len(pts)}")
    if len(above) < 50:
        print("[diag][A] too few above-floor points -> 0 superpoints")
        return np.zeros(0, dtype=int), np.zeros((0, 3))
    xy = above[:, :2]
    key = np.floor(xy / voxel).astype(np.int64)
    # collapse voxel keys to compact integer labels
    uniq_keys, inv = np.unique(key, axis=0, return_inverse=True)
    # centroid of each voxel cluster (average member points)
    n_clusters = uniq_keys.shape[0]
    sums = np.zeros((n_clusters, 3))
    cnts = np.zeros(n_clusters)
    np.add.at(sums, inv, above)
    np.add.at(cnts, inv, 1.0)
    centroids = sums / cnts[:, None]
    print(f"[diag][A] superpoint voxels: {len(centroids)}")
    return inv.astype(np.int64), centroids


def merge_to_boxes(scene: Scene, yaw: float = 0.0,
                   min_side: float = 0.4, max_gap_merge: float = 0.2) -> list[OrientedBox]:
    """Merge superpoint voxels into device boxes along row direction.

    Voxels are clustered by (a) cross-axis position (row membership) and
    (b) along-axis contiguity with small gap tolerance. Resulting clusters
    that are wide enough become candidate rack boxes. Conservative: larger
    merge tolerance to favour recall.

    Only device-surface voxels are considered: height in [0.5, 2.4] and
    bounded footprint near the room's device cluster, which excludes walls
    and floor noise.
    """
    labels, centroids = superpoint_over_segmentation(scene)
    if len(centroids) == 0:
        return []
    axis = np.array([math.cos(yaw), math.sin(yaw)])
    cross = np.array([-math.sin(yaw), math.cos(yaw)])
    z = centroids[:, 2]
    # restrict to plausible device heights (racks), dropping walls & floor
    height_ok = (z > 0.5) & (z < 2.4)
    c = centroids[height_ok]
    print(f"[diag][A] centroids in height band (0.5,2.4): {len(c)}/{len(centroids)}")
    if len(c) < 3:
        print("[diag][A] too few device-height voxels -> no candidates")
        return []
    z = z[height_ok]
    along = c[:, :2] @ axis
    crs = c[:, :2] @ cross

    # --- density-based row extraction on the cross axis ---
    # Rows appear as high-density peaks; aisles and walls are low-density.
    bin_w = 0.15
    lo_b, hi_b = crs.min() - bin_w, crs.max() + bin_w
    n_bins = max(4, int((hi_b - lo_b) / bin_w))
    hist, edges = np.histogram(crs, bins=n_bins, range=(lo_b, hi_b))
    thr = max(3, int(0.25 * hist.max()))
    dense = hist >= thr
    # group consecutive dense bins into row bands
    bands: list[tuple[float, float]] = []
    i = 0
    while i < len(dense):
        if dense[i]:
            j = i
            while j + 1 < len(dense) and dense[j + 1]:
                j += 1
            bands.append((edges[i] - 0.05, edges[j + 1] + 0.05))
            i = j + 1
        else:
            i += 1
    print(f"[diag][A] cross-axis dense bands: {len(bands)}  "
          f"ranges={[(round(lo, 2), round(hi, 2)) for lo, hi in bands]}")
    if not bands:
        print("[diag][A] WARNING: no dense bands (point density too low, or wrong yaw/scale)")

    # Each dense band is a candidate device *face* (front or back of a row),
    # or a wall. Pair up bands whose cross gap matches a plausible rack depth
    # (0.6..1.4m) AND whose along-axis extents overlap strongly -> one row.
    # Unpaired bands become thin candidates that later stages can prune.
    band_stats = []
    for (blo, bhi) in bands:
        idx = np.where((crs >= blo) & (crs <= bhi))[0]
        if len(idx) < 3:
            continue
        band_stats.append({
            "lo": blo, "hi": bhi, "idx": idx,
            "along_min": float(along[idx].min()),
            "along_max": float(along[idx].max()),
            "center": (blo + bhi) / 2,
        })

    def _overlap(a, b) -> float:
        lo = max(a["along_min"], b["along_min"])
        hi = min(a["along_max"], b["along_max"])
        if hi <= lo:
            return 0.0
        return (hi - lo) / max(min(a["along_max"] - a["along_min"],
                                    b["along_max"] - b["along_min"]), 1e-6)

    used = set()
    rows: list[np.ndarray] = []
    for i in range(len(band_stats)):
        if i in used:
            continue
        best_j, best_score = None, 0.0
        for j in range(i + 1, len(band_stats)):
            if j in used:
                continue
            gap = band_stats[j]["center"] - band_stats[i]["center"]
            if 0.5 <= gap <= 1.5:
                ov = _overlap(band_stats[i], band_stats[j])
                if ov > 0.6 and ov > best_score:
                    best_j, best_score = j, ov
        if best_j is not None:
            used.update([i, best_j])
            rows.append(np.concatenate([band_stats[i]["idx"], band_stats[best_j]["idx"]]))
        else:
            used.add(i)
            rows.append(band_stats[i]["idx"])
    print(f"[diag][A] row groups after band pairing: {len(rows)} "
          f"(sizes={[len(r) for r in rows]})")

    boxes: list[OrientedBox] = []
    row_id = 0
    for row_idx in rows:
        if len(row_idx) < 1:
            continue
        row_idx = np.asarray(row_idx, dtype=int)
        r_along = along[row_idx]
        r_crs = crs[row_idx]
        r_z = z[row_idx]
        a_order = np.argsort(r_along)
        # split contiguous runs along the row with gap tolerance
        runs: list[np.ndarray] = []
        start = 0
        for k in range(1, len(a_order)):
            if r_along[a_order[k]] - r_along[a_order[k - 1]] > max_gap_merge:
                runs.append(a_order[start:k])
                start = k
        runs.append(a_order[start:])
        for run in runs:
            if len(run) == 0:
                continue
            a_pts = r_along[run]
            cr_pts = r_crs[run]
            z_pts = r_z[run]
            length = float(a_pts.max() - a_pts.min()) + MERGE_PRIMARY_VOXEL
            depth = float(cr_pts.max() - cr_pts.min()) + MERGE_PRIMARY_VOXEL
            height = float(z_pts.max()) + 0.1
            if length < min_side:
                continue
            along_c = float((a_pts.max() + a_pts.min()) / 2)
            crs_c = float((cr_pts.max() + cr_pts.min()) / 2)
            world_xy = axis * along_c + cross * crs_c
            boxes.append(OrientedBox(
                center=(world_xy[0], world_xy[1], height / 2),
                size=(length, max(depth, 0.4), max(height, 0.5)),
                yaw=yaw, device_type=DeviceType.RACK,
                source=BoxSource.COARSE_SEG, confidence=Confidence.LOW,
                row_id=row_id,
            ))
        row_id += 1
    print(f"[diag][A] candidate boxes produced: {len(boxes)} "
          f"(lengths={[round(b.size[0], 2) for b in boxes]})")
    return boxes


def coarse_segment(scene: Scene, opts: dict | None = None) -> list[OrientedBox]:
    """Stage A entry point."""
    opts = opts or {}
    yaw = float(opts.get("yaw", scene.meta.get("yaw", 0.0)))
    axes = merge_to_boxes(scene, yaw=yaw)
    scene.boxes = axes
    return axes
