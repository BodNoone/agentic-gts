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
    return float(estimate_yaw_detailed(points, z_range, voxel)["yaw"])


def estimate_yaw_detailed(points: np.ndarray, z_range: tuple[float, float] = (0.4, 2.5),
                          voxel: float = 0.25) -> dict:
    """Same as estimate_yaw but returns intermediate results for diagnosis.

    Returns dict with:
      yaw        float          chosen yaw in [-pi/4, pi/4)
      candidates list           [(deg, score)] scored Manhattan candidates
      cells      ndarray | None occupancy cells after boundary removal
      device_pts ndarray        2D points in the device height band
    """
    z = points[:, 2]
    m = (z > z_range[0]) & (z < z_range[1])
    pts = points[m][:, :2]
    print(f"[diag][yaw] points in z({z_range[0]},{z_range[1]}): {int(m.sum())}/{len(points)}")
    if len(pts) < 100:
        print("[diag][yaw] too few device-height points -> fallback yaw=0")
        return {"yaw": 0.0, "candidates": [], "cells": None, "device_pts": pts}

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
    print(f"[diag][yaw] voxel cells: {len(cells)}")
    cells = _remove_boundary_cells(cells)
    if len(cells) < 12:
        print("[diag][yaw] too few cells after boundary removal -> fallback yaw=0")
        return {"yaw": 0.0, "candidates": [], "cells": None, "device_pts": pts}

    # local direction per cell: PCA over neighboring cells within radius
    from scipy.spatial import cKDTree
    tree = cKDTree(cells)
    pairs = tree.query_pairs(r=voxel * 2.2, output_type="ndarray")
    print(f"[diag][yaw] neighbor pairs: {len(pairs)}")
    if len(pairs) < 10:
        print("[diag][yaw] too few pairs -> fallback global PCA")
        return {"yaw": _global_pca_yaw(cells), "candidates": [], "cells": cells,
                "device_pts": pts}
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
        if len(cands) >= 5:
            break
    print(f"[diag][yaw] candidate angles (deg): {[round(math.degrees(a), 1) for a in cands]}")
    if not cands:
        print("[diag][yaw] no candidates -> fallback global PCA")
        return {"yaw": _global_pca_yaw(cells), "candidates": [], "cells": cells,
                "device_pts": pts}

    best_yaw, best_score = 0.0, -1.0
    cand_scores: list[tuple[float, float]] = []
    for a in cands:
        for yaw_c in (a, a - math.pi / 2):  # both Manhattan directions
            score = _row_band_score(cells, yaw_c)
            cand_scores.append((math.degrees(yaw_c), score))
            if score > best_score:
                best_score, best_yaw = score, yaw_c
    print(f"[diag][yaw] candidate scores: "
          f"{[(round(d, 1), round(s)) for d, s in cand_scores]}")
    # fine refinement: PCA-fit each detected row band. Candidate selection is
    # coarse (2-deg histogram bins) and mass scores tie within several
    # degrees (tilt smears bands without losing mass), but each row band is
    # an elongated rectangle whose PCA major axis gives its direction to
    # ~1 deg.
    refined = _refine_yaw_by_rows(cells, best_yaw)
    print(f"[diag][yaw] chosen yaw = {math.degrees(refined):.1f} deg "
          f"(candidate={math.degrees(best_yaw):.1f} mass_score={best_score:.0f})")
    best_yaw = refined
    if best_score <= 0:
        print("[diag][yaw] WARNING: band score 0 -> no row-like structure at this yaw")
    # map to [-pi/4, pi/4) minimal rotation
    yaw = math.remainder(best_yaw, math.pi / 2)
    if yaw >= math.pi / 4:
        yaw -= math.pi / 2
    elif yaw < -math.pi / 4:
        yaw += math.pi / 2
    return {"yaw": float(yaw), "candidates": cand_scores, "cells": cells,
            "device_pts": pts}


def _ang_dist(a: float, b: float) -> float:
    """Distance in the folded [0, pi/2) angle space."""
    d = abs(a - b) % (math.pi / 2)
    return min(d, math.pi / 2 - d)


def _remove_boundary_cells(cells: np.ndarray, dist: float = 0.35) -> np.ndarray:
    """Drop cells near the convex-hull boundary of the occupied area.

    Walls lie on the room boundary; device rows are interior. In real 3DGS
    clouds walls are denser than device surfaces and their direction would
    otherwise dominate the yaw histogram (walls axis-aligned, devices
    rotated). Removing a boundary strip suppresses wall bands while leaving
    the row bands intact.
    """
    if len(cells) < 12:
        return cells
    from scipy.spatial import ConvexHull
    try:
        hull = ConvexHull(cells)
    except Exception:
        return cells
    verts = cells[hull.vertices]
    keep = np.ones(len(cells), dtype=bool)
    for i in range(len(verts)):
        a = verts[i]
        b = verts[(i + 1) % len(verts)]
        ab = b - a
        L = float(np.dot(ab, ab))
        if L < 1e-12:
            continue
        t = np.clip((cells - a) @ ab / L, 0.0, 1.0)
        proj = a + t[:, None] * ab
        d = np.linalg.norm(cells - proj, axis=1)
        keep &= d > dist
    kept = cells[keep]
    print(f"[diag][yaw] boundary (wall) cell removal: {len(cells)} -> {len(kept)}")
    return kept


def _row_band_score(cells: np.ndarray, yaw: float, bin_w: float = 0.15) -> float:
    """How well does this yaw explain device *rows*?

    Project occupied cells onto the cross axis; real rows give a histogram
    with several narrow, dense bands separated by empty aisles. Walls are
    assumed removed beforehand (boundary-cell stripping). Score = sum over
    bands of (band mass) restricted to bands with plausible row width
    (0.2..2m). The perpendicular direction only yields thin side-face
    spikes with little total mass, so mass discriminates directions well.
    """
    cross = np.array([-math.sin(yaw), math.cos(yaw)])
    v = cells @ cross
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


def _refine_yaw_by_rows(cells: np.ndarray, yaw: float, bin_w: float = 0.15) -> float:
    """Fine-tune yaw by PCA-fitting each detected row band.

    Bands are extracted from the cross-axis histogram at the candidate yaw
    (assumed within ~10 deg of the true direction). Each plausible band is
    an elongated rectangle of racks; the PCA major axis of its cells gives
    the row direction to ~1 deg. Returns the weighted circular mean of the
    row directions, or the input yaw unchanged if no usable rows exist.
    """
    cross = np.array([-math.sin(yaw), math.cos(yaw)])
    v = cells @ cross
    lo, hi = v.min(), v.max()
    nb = max(int((hi - lo) / bin_w), 4)
    hist, edges = np.histogram(v, bins=nb, range=(lo, hi))
    thr = max(2, 0.2 * hist.max())
    dense = hist >= thr
    votes: list[tuple[complex, float]] = []   # (unit direction, weight)
    i = 0
    while i < len(dense):
        if dense[i]:
            j = i
            while j + 1 < len(dense) and dense[j + 1]:
                j += 1
            width = (j - i + 1) * bin_w
            if 0.3 <= width <= 3.0:      # plausible row(s) band
                mask = (v >= edges[i]) & (v < edges[j + 1])
                rc = cells[mask]
                if len(rc) >= 12:
                    rc = rc - rc.mean(axis=0)
                    cov = rc.T @ rc
                    evals, evecs = np.linalg.eigh(cov)
                    elong = math.sqrt(evals[-1] / max(evals[0], 1e-9))
                    if elong > 2.0:      # clearly row-like, not a square blob
                        theta = math.atan2(evecs[1, -1], evecs[0, -1])
                        # fold to the direction nearest to the candidate yaw
                        d = (theta - yaw + math.pi / 2) % math.pi - math.pi / 2
                        votes.append((np.exp(1j * (yaw + d)), float(len(rc))))
            i = j + 1
        else:
            i += 1
    if not votes:
        print(f"[diag][yaw] row refinement: no usable rows, keeping candidate")
        return yaw
    total = sum(w for _, w in votes)
    zsum = sum(c * w for c, w in votes) / total
    return math.atan2(zsum.imag, zsum.real)


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
