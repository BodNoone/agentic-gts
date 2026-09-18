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


def estimate_residual_yaw(points: np.ndarray, yaw: float) -> float:
    """Self-consistency residual of a yaw estimate (radians, folded to
    [-pi/4, pi/4)).

    Rotates the cloud by -yaw and re-runs the SAME estimator on the
    result: if the yaw is right, the rotated rows are axis-aligned and
    the re-estimate returns ~0; if the first pass was hijacked (a wall
    or sloped floor pulling the histogram peak), the rotated rows sit
    at the ERROR angle and the re-estimate returns it as the residual.
    A closed-loop check the raw score cannot fake -- consistency, not
    confidence. Curved walls do NOT trip it (their energy spreads
    evenly over the angle histogram, adding a flat floor rather than
    a competing peak); a large residual that PERSISTS after one
    correction means a genuinely multi-directional layout.
    """
    c, s = math.cos(yaw), math.sin(yaw)
    rot = np.array(points, dtype=np.float64, copy=True)
    x, y = rot[:, 0].copy(), rot[:, 1].copy()
    rot[:, 0] = c * x + s * y          # rotation by -yaw about +z
    rot[:, 1] = -s * x + c * y
    return float(estimate_yaw_detailed(rot)["yaw"])


def top_yaw_candidates(info: dict, k: int = 3,
                       bin_deg: float = 2.0) -> list[tuple[float, float]]:
    """Distinct mod-90 layout directions from an estimate's candidate
    scores, ranked by best mass score: [(yaw_rad, score), ...].

    The candidate list carries BOTH Manhattan orientations of each
    histogram peak (a and a - pi/2, the row-side and the cross-side
    band) with different scores -- they are the SAME layout direction
    once folded to [-pi/4, pi/4). Fold, keep the best score per
    direction, return the top-k.

    Why: on knife-edged scenes (several near-equal structures, the
    argmax flipping with tiny upstream perturbations) the TRUE row
    direction sits at #2-3 by score, never winning outright -- user
    logs: truth -17.5 deg scored 468/430 while the WRONG winners took
    504/440. Downstream arbitration by grounding yield needs this
    short list that is guaranteed to CONTAIN the truth.
    """
    best: dict[int, tuple[float, float]] = {}
    for deg, score in (info.get("candidates") or []):
        w = math.remainder(math.radians(float(deg)), math.pi / 2)
        key = int(round(math.degrees(w) / bin_deg))
        if key not in best or score > best[key][1]:
            best[key] = (w, float(score))
    ranked = sorted(best.values(), key=lambda t: -t[1])
    return ranked[:k]


def pick_yaw_trial(trials: list[dict],
                   agree: float = math.radians(3.0)) -> dict | None:
    """Best yaw trial from arbitration: {"yaw", "n", "delta"}, ... .

    Preference: trials whose fitted boxes' OWN directions AGREE with
    the render yaw (|delta| <= agree) first -- agreement is unique to
    the true direction, because at every wrong yaw the boxes' interior
    PCA votes the ERROR angle (the rows are still physically wherever
    they are; only the RENDER was tilted) -- then the most boxes (a
    straight view detects more structures than a skewed one). None
    when no trials.
    """
    if not trials:
        return None

    def _key(t: dict):
        d = t.get("delta")
        ok = d is not None and abs(d) <= agree
        return (1 if ok else 0, t.get("n", 0),
                -(abs(d) if d is not None else 9.0))

    return max(trials, key=_key)


def seed_axis_delta(boxes, points: np.ndarray, cur_yaw: float,
                    min_len: float = 2.0, min_pts: int = 150,
                    top_cut: float | None = None):
    """Yaw offset (radians, folded to [-pi/4, pi/4)) of the grounded
    seeds' OWN directions against the render yaw, or None with too
    few votes.

    The stageG feedback signal. The seeds fit in the row frame (their
    box.yaw is 0 / pi/2 BY CONSTRUCTION -- it carries no direction
    information), so each seed's direction is MEASURED here, in the
    EXACT context the original local-PCA seed fit used (dd21246, the
    version that selected the yaw correctly on real scenes): PCA on
    the MIDDLE z-slice of the structure -- [0.35, 0.75] x height
    above the box's floor, the band that cuts floor creep AND tray /
    ceiling remnants alike -- drawn from a pool cut at z_top + 0.10.
    A whole-device-band PCA (the first reinstatement) let the haze at
    both ends of the band pull the covariance and the votes came out
    imperfect (user report). This is deliberately NOT the older pool
    feedback either, which re-ran the GLOBAL histogram estimator on
    the union of the boxes' points -- a pool CARVED by box geometry
    cut along the ASSUMED yaw, so slanted rows re-confirmed the
    assumed yaw. Per-box middle-slice PCA keeps each vote LOCAL to
    one structure: no histogram to hijack, one stray wall-ish fit
    cannot drag the aggregate.

    Votes: only boxes long enough for a trustworthy axis (a stubby AC
    unit's PCA direction is noise), THICK enough to be a device (a
    WALL box is long and -- in a mesh -- dense, exactly the heaviest
    possible voter at the WRONG angle: the cluster recall net proposes
    wall blobs, the VLM sometimes calls them rack rows, and the vote
    then drags the median and corrupts a CORRECT yaw, run-to-run
    randomly with the VLM's own nondeterminism -- user report), and
    with enough point support; folded mod-90 (perpendicular rows
    agree), weighted by point count, taken as the weighted MEDIAN.
    """
    hi = float(top_cut) if top_cut else 2.5
    band = points[(points[:, 2] > 0.30) & (points[:, 2] < hi)]
    if len(band) < 500:
        return None
    votes = []
    for b in boxes:
        if b.size[0] < min_len or b.size[1] < 0.35:
            continue
        inside = band[b.contains(band)]
        if len(inside) < min_pts:
            continue
        # the seed fit's measurement context (dd21246): the middle
        # z-slice of the structure, relative to the box's OWN floor
        # (the fit set the bottom at the local floor) -- NOT the
        # whole device band, whose haze at both ends drags the PCA
        bot = float(b.center[2]) - 0.5 * float(b.size[2])
        h = float(b.size[2])
        zc0 = bot + max(0.30, 0.35 * h)
        zc1 = max(zc0 + 0.10, bot + 0.75 * h)
        core = inside[(inside[:, 2] >= zc0) & (inside[:, 2] <= zc1)]
        if len(core) < 30:
            core = inside               # thin structure: whole band
        d = core[:, :2] - core[:, :2].mean(axis=0)
        cov = d.T @ d / len(d)
        _, V = np.linalg.eigh(cov)       # ascending eigenvalues
        v_row = V[:, 1]                  # the structure's long axis
        votes.append((math.remainder(
            math.atan2(v_row[1], v_row[0]) - cur_yaw, math.pi / 2),
            float(len(inside))))
    if len(votes) < 2:
        return None
    votes.sort()
    w_tot = sum(w for _, w in votes)
    if w_tot <= 0:
        return None
    acc = 0.0
    for d, w in votes:
        acc += w
        if acc >= 0.5 * w_tot:
            return float(d)
    return float(votes[-1][0])


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
    band = points[m]
    pts = band[:, :2]
    print(f"[diag][yaw] points in z({z_range[0]},{z_range[1]}): {int(m.sum())}/{len(points)}")
    if len(pts) < 100:
        print("[diag][yaw] too few device-height points -> fallback yaw=0")
        return {"yaw": 0.0, "candidates": [], "cells": None, "device_pts": pts}

    # subsample for speed
    if len(band) > 150_000:
        sel = np.random.default_rng(0).choice(len(band), 150_000, replace=False)
        band = band[sel]

    # keep only VERTICAL surfaces: rack side faces carry the row direction,
    # while horizontal planes (rack-top fields, ceilings, floors, pipe runs)
    # pollute the direction histogram and can hijack the estimate. Normals
    # from local PCA (kNN=20); falls back to all points if too few survive.
    if len(band) > 2_000:
        try:
            import open3d as o3d
            pc = o3d.geometry.PointCloud()
            pc.points = o3d.utility.Vector3dVector(band)
            pc.estimate_normals(o3d.geometry.KDTreeSearchParamKNN(knn=20))
            nz = np.abs(np.asarray(pc.normals)[:, 2])
            vert = band[nz < 0.5]
            print(f"[diag][yaw] vertical-surface filter: {len(vert)}/{len(band)}")
            if len(vert) > 1_000:
                band = vert
        except Exception as e:
            print(f"[diag][yaw] normal estimation skipped ({type(e).__name__})")
    pts = band[:, :2]

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
    # Hint-free bootstrap byproducts (user request: run the grounding
    # WITHOUT initial boxes): the cells that SURVIVE boundary removal
    # are the device layout -- walls sit on the room boundary (dropped
    # here), floors/ceilings/cable trays are horizontal (dropped by the
    # vertical-surface filter) or outside the height band. Export
    #   device_footprint  world-frame xy bounds of the surviving cells
    #   z_top             device top height (anchored estimate of the
    #                     band points inside those cells) -- the ceiling-
    #                     cut and framing reference ground_stage needs
    #                     (there is no box input to take them from).
    boot = {"z_top": None, "device_footprint": None, "device_cells": None}
    keep = boundary_keep_mask(cells)
    print(f"[diag][yaw] boundary (wall) cell removal: {len(cells)} -> {int(keep.sum())}")
    if keep.any():
        kc = cells[keep]
        # the CELLS themselves are exported for framing: a world-frame
        # AABB of a ROTATED layout is inflated (a 20x1m row at 45 deg
        # bounds to ~14x14m), and AABB-then-rotate inflates again in
        # _render_topdown -- the camera rose and the nadir view came
        # back mostly empty. Rotating the cells by the ACTUAL yaw and
        # AABBing once keeps the framing tight (and stays correct when
        # the caller pins a different yaw).
        boot["device_cells"] = kc
        boot["device_footprint"] = (
            float(kc[:, 0].min()), float(kc[:, 1].min()),
            float(kc[:, 0].max()), float(kc[:, 1].max()))
        # band points whose voxel survived: the pk encoding (kx * M + ky)
        # is collision-free for any realistic grid (|k| < M/2 voxels)
        pk = key[:, 0].astype(np.int64) * 10_000_000 + key[:, 1]
        kept_pk = (uniq[keep][:, 0].astype(np.int64) * 10_000_000
                   + uniq[keep][:, 1])
        memb = np.isin(pk, kept_pk)
        if int(memb.sum()) > 500:
            # ANCHORED column top, not P99.5: a percentile lets ANY
            # >0.5% overhead tail (cable-tray supports, hanging bundles
            # standing in device cells) drag the reference up to the
            # clutter height -- and this one reference feeds the
            # grounding fit cap AND the per-piece height columns, so
            # every box inherits the inflated top (user report:
            # hint-free runs came out with unadjusted, too-tall
            # heights). A device body is density-CONNECTED from the
            # ground; floating layers sit above a near-empty gap the
            # anchored walk stops at.
            from agentic_gts.agent.mask_refine import _anchored_top
            z = band[memb][:, 2]
            top = _anchored_top(z)
            boot["z_top"] = (float(top) if top is not None
                             else float(np.percentile(z, 99.5)))
            print(f"[diag][yaw] bootstrap: z_top={boot['z_top']:.2f} "
                  f"footprint={tuple(round(v, 2) for v in boot['device_footprint'])}")
    cells = cells[keep]
    if len(cells) < 12:
        print("[diag][yaw] too few cells after boundary removal -> fallback yaw=0")
        return {"yaw": 0.0, "candidates": [], "cells": None, "device_pts": pts,
                **boot}

    # local direction per cell: PCA over neighboring cells within radius
    from scipy.spatial import cKDTree
    tree = cKDTree(cells)
    pairs = tree.query_pairs(r=voxel * 2.2, output_type="ndarray")
    print(f"[diag][yaw] neighbor pairs: {len(pairs)}")
    if len(pairs) < 10:
        print("[diag][yaw] too few pairs -> fallback global PCA")
        return {"yaw": _global_pca_yaw(cells), "candidates": [], "cells": cells,
                "device_pts": pts, **boot}
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
                "device_pts": pts, **boot}

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
            "device_pts": pts, **boot}


def _ang_dist(a: float, b: float) -> float:
    """Distance in the folded [0, pi/2) angle space."""
    d = abs(a - b) % (math.pi / 2)
    return min(d, math.pi / 2 - d)


def boundary_keep_mask(cells: np.ndarray, dist: float = 0.35) -> np.ndarray:
    """Boolean mask of cells farther than `dist` from the convex-hull boundary.

    Walls lie on the room boundary; device rows are interior. Used by yaw
    estimation / layout bootstrap to suppress wall structure.
    """
    if len(cells) < 12:
        return np.ones(len(cells), dtype=bool)
    from scipy.spatial import ConvexHull
    try:
        hull = ConvexHull(cells)
    except Exception:
        return np.ones(len(cells), dtype=bool)
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
        keep &= np.linalg.norm(cells - proj, axis=1) > dist
    return keep


def _remove_boundary_cells(cells: np.ndarray, dist: float = 0.35) -> np.ndarray:
    """Drop cells near the convex-hull boundary of the occupied area.

    In real 3DGS clouds walls are denser than device surfaces and their
    direction would otherwise dominate the yaw histogram (walls axis-aligned,
    devices rotated). Removing a boundary strip suppresses wall bands while
    leaving the row bands intact.
    """
    kept = cells[boundary_keep_mask(cells, dist)]
    print(f"[diag][yaw] boundary (wall) cell removal: {len(cells)} -> {len(kept)}")
    return kept


def _row_band_score(cells: np.ndarray, yaw: float, bin_w: float = 0.15) -> float:
    """How well does this yaw explain device *rows*?

    Project occupied cells onto the cross axis; real rows give a histogram
    with several narrow, dense bands separated by empty aisles. Walls are
    assumed removed beforehand (boundary-cell stripping). Score = sum over
    bands of (band mass) restricted to bands with plausible row width
    (0.5..2m). The perpendicular direction only yields thin side-face
    spikes with little total mass, so mass discriminates directions well.

    MESH hijack guard (user report: yaw far off with --mesh-cloud): a
    mesh renders walls as PERFECT dense planes, and when the
    reconstruction extends past the machine room (captured corridor /
    neighboring space) the room walls are INTERIOR to the hull -- the
    boundary strip cannot remove them, and their single-line bands
    carry full mass: wall mass ~ row-face mass and the score flips on
    noise. A wall band and a rack FACE band are geometrically
    identical thin vertical planes -- the discriminator is PAIRING: a
    rack face always has its front/back sibling one rack-depth away
    (0.5..2.4m, incl. back-to-back doubles), a wall stands alone.
    Thin bands score full mass only when PAIRED; solitary thin lines
    (walls, starved single faces) score 10%.
    """
    cross = np.array([-math.sin(yaw), math.cos(yaw)])
    v = cells @ cross
    lo, hi = v.min(), v.max()
    nb = max(int((hi - lo) / bin_w), 4)
    hist, edges = np.histogram(v, bins=nb, range=(lo, hi))
    thr = max(2, 0.2 * hist.max())
    dense = hist >= thr
    score = 0.0
    thins: list[tuple[float, float]] = []   # (center, mass)
    i = 0
    while i < len(dense):
        if dense[i]:
            j = i
            while j + 1 < len(dense) and dense[j + 1]:
                j += 1
            width = (j - i + 1) * bin_w
            mass = float(hist[i:j + 1].sum())
            if 0.5 <= width <= 2.0:      # solid single-row band
                score += mass
            elif width > 2.0:            # blob: wrong direction merges rows
                score += mass * 0.2
            else:                        # thin: face sheet OR wall line
                thins.append((0.5 * (edges[i] + edges[j + 1]), mass))
            i = j + 1
        else:
            i += 1
    # thin bands: a rack FACE always has its front/back sibling one
    # rack-depth away (network racks 0.45m to back-to-back doubles
    # 2.2m); a wall stands alone or in pairs metres apart. Sibling
    # support is NON-EXCLUSIVE (greedy exclusive pairing mis-couples
    # an interior artifact band with one face and starves the other):
    # any thin band with another thin band within 0.40..2.4m scores
    # full mass; solitary thin lines (walls, lone starved faces)
    # score 10%.
    tc = [c for c, _ in thins]
    for k, (c, m) in enumerate(thins):
        sib = any(0.40 <= abs(c - c2) <= 2.4
                  for k2, c2 in enumerate(tc) if k2 != k)
        score += m if sib else 0.1 * m
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
