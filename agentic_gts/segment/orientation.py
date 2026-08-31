"""Dominant orientation (yaw) estimation for a scene.

The whole pipeline reasons in a row-aligned frame; real point clouds come in
arbitrary orientations, so we estimate the dominant horizontal direction of
the device structures before anything else runs.

Method: for points at plausible device heights, compute local 2D surface
directions via neighborhood PCA on a voxel grid, then histogram the angles
modulo 90 deg (Manhattan assumption: rows are parallel/perpendicular) and
take the peak. Falls back to global PCA if the histogram is flat.
"""
from __future__ import annotations

import math

import numpy as np


def estimate_yaw(points: np.ndarray, z_range: tuple[float, float] = (0.4, 2.5),
                 voxel: float = 0.25) -> float:
    """Estimate the dominant row direction (radians, in [-pi/4, pi/4)).

    Returns the yaw angle such that rotating the scene by -yaw aligns device
    rows with the +x axis. Modulo-90deg symmetric: we cannot (and need not)
    distinguish rows along x from rows along y; the row detector handles both
    once the scene is axis-aligned.
    """
    z = points[:, 2]
    m = (z > z_range[0]) & (z < z_range[1])
    pts = points[m][:, :2]
    if len(pts) < 100:
        return 0.0

    # subsample for speed
    if len(pts) > 150_000:
        sel = np.random.default_rng(0).choice(len(pts), 150_000, replace=False)
        pts = pts[sel]

    # voxel-average to build a sparse occupancy set (removes density bias)
    key = np.floor(pts / voxel).astype(np.int64)
    uniq, inv = np.unique(key, axis=0, return_inverse=True)
    n = uniq.shape[0]
    sums = np.zeros((n, 2))
    cnts = np.zeros(n)
    np.add.at(sums, inv, pts)
    np.add.at(cnts, inv, 1.0)
    cells = sums / cnts[:, None]
    if len(cells) < 12:
        return 0.0

    # local direction per cell: PCA over neighboring cells within radius
    from scipy.spatial import cKDTree
    tree = cKDTree(cells)
    pairs = tree.query_pairs(r=voxel * 2.2, output_type="ndarray")
    if len(pairs) < 10:
        return _global_pca_yaw(cells)
    d = cells[pairs[:, 1]] - cells[pairs[:, 0]]
    ang = np.arctan2(d[:, 1], d[:, 0])          # [-pi, pi]
    ang = np.mod(ang, math.pi / 2)              # fold to [0, pi/2): Manhattan
    # weight long edges slightly higher (structure > noise)
    w = np.linalg.norm(d, axis=1)

    nbins = 90
    hist, edges = np.histogram(ang, bins=nbins, range=(0, math.pi / 2), weights=w)
    # smooth circularly (folded space wraps at 0 == pi/2)
    kernel = np.array([1, 2, 3, 2, 1], dtype=float)
    kernel /= kernel.sum()
    ext = np.concatenate([hist[-2:], hist, hist[:2]])
    smooth = np.convolve(ext, kernel, mode="same")[2:-2]

    # candidate peaks: top-k local maxima, scored by row-structure quality.
    # Walls also produce angle peaks, but only the true row direction yields
    # tight, high-occupancy bands when the cells are projected on the
    # cross axis. Score each candidate and take the best.
    order = np.argsort(smooth)[::-1]
    cands: list[float] = []
    for idx in order:
        a = (edges[idx] + edges[idx + 1]) / 2
        if all(_ang_dist(a, c) > math.radians(8) for c in cands):
            cands.append(float(a))
        if len(cands) >= 3:
            break
    if not cands:
        return _global_pca_yaw(cells)

    best_yaw, best_score = 0.0, -1.0
    for a in cands:
        for yaw_c in (a, a - math.pi / 2):  # both Manhattan directions
            score = _row_band_score(cells, yaw_c)
            if score > best_score:
                best_score, best_yaw = score, yaw_c
    # map to [-pi/4, pi/4) minimal rotation
    yaw = math.remainder(best_yaw, math.pi / 2)
    if yaw >= math.pi / 4:
        yaw -= math.pi / 2
    elif yaw < -math.pi / 4:
        yaw += math.pi / 2
    return float(yaw)


def _ang_dist(a: float, b: float) -> float:
    """Distance in the folded [0, pi/2) angle space."""
    d = abs(a - b) % (math.pi / 2)
    return min(d, math.pi / 2 - d)


def _row_band_score(cells: np.ndarray, yaw: float) -> float:
    """How well does this yaw explain device *rows*?

    Project occupied cells onto the cross axis; real rows give a histogram
    with several narrow, dense bands separated by empty aisles. Walls give
    one thin band at the border (low total mass). Score = sum over bands of
    (band mass) restricted to bands with plausible row width (0.2..2m).
    """
    cross = np.array([-math.sin(yaw), math.cos(yaw)])
    v = cells @ cross
    bin_w = 0.15
    lo, hi = v.min(), v.max()
    nb = max(int((hi - lo) / bin_w), 4)
    hist, edges = np.histogram(v, bins=nb, range=(lo, hi))
    thr = max(2, 0.2 * hist.max())
    dense = hist >= thr
    score = 0.0
    i = 0
    while i < len(dense):
        if dense[i]:
            j = i
            while j + 1 < len(dense) and dense[j + 1]:
                j += 1
            width = (j - i + 1) * bin_w
            mass = float(hist[i:j + 1].sum())
            if 0.1 <= width <= 2.0:      # plausible single-row band
                score += mass
            elif width > 2.0:            # blob: wrong direction merges rows
                score += mass * 0.2
            i = j + 1
        else:
            i += 1
    return score


def _global_pca_yaw(cells: np.ndarray) -> float:
    """Fallback: principal direction of the occupied cells."""
    c = cells - cells.mean(axis=0)
    cov = c.T @ c / max(len(c) - 1, 1)
    evals, evecs = np.linalg.eigh(cov)
    v = evecs[:, int(np.argmax(evals))]
    yaw = math.atan2(v[1], v[0])
    yaw = math.remainder(yaw, math.pi / 2)
    if yaw >= math.pi / 4:
        yaw -= math.pi / 2
    elif yaw < -math.pi / 4:
        yaw += math.pi / 2
    return float(yaw)
