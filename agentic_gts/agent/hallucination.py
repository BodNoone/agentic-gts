"""Structured-coordinate hallucination filter for Qwen3-VL grounding
output (user direction; replaces the earlier tower-only guard).

The failure mode: the autoregressive grounding sometimes slips into a
repetitive coordinate-generation mode -- a flood of bboxes over empty
floor or background that is NOT random noise but highly structured:
x1/x2 (or y1/y2) near-identical, width/height near-constant, marching
along one axis at a fixed small pitch, neighbours head-to-tail, and
coordinates running past the image boundary (e.g. [700,8,857,471] /
[700,47,857,104] / [708,104,857,160] ...). No model weights are
touched and no second VLM is consulted -- the verdict is geometric
statistics on the PARSED rects, inserted after judge.ground_regions
(the bbox parser + 0-1000-relative -> view-pixel conversion lives in
agent/judge.py) and before any 2D->3D lifting (_shift_rect crop
offset -> _frame_rect camera unprojection -> _fit_region_boxes
point-support fit, all in agent/ground.py).

Coordinate system (confirmed on the current pipeline): rects arrive in
the VIEW's pixel space (post-crop W x H), already CLIPPED to the frame
by the parser -- raw out-of-bounds coordinates therefore surface here
as EDGE-PINNED rects (an edge at exactly 0 / W / H), which is the OOB
proxy this filter scores. Every godview render puts the ROW frame
axis-aligned in the image (image x = row axis, image y = across-row),
for nadir, tiles and recall tilts alike.

What is deliberately NOT treated as hallucination: real machine rooms
are regular. Rows repeat at a fixed aisle pitch and an over-split row
is a TIGHT chain of same-size cabinets along the row axis -- both
perfectly legitimate. The safety line is physical: adjacent real rows
always have an aisle (>= 0.6 m ~= 55% of a 1.1 m row's image depth,
>= 27% even for a 2.2 m back-to-back double), so chain links only
form on gaps <= 25% of the box's along-axis size, and the row-axis
(X) score threshold is far stricter than the cross-row (Y) threshold:
a tight X chain is the genuine over-split signature, a tight Y stack
is physically impossible. Count alone is never a verdict (row-heavy
rooms legitimately ground 30+ regions; the count component is per
CHAIN, not per view).

Per view: rects chain (union-find, transitive -- perspective-scaled
towers whose ends differ far beyond any pairwise tolerance still link
neighbour-to-neighbour) on each axis; every chain of >= min_chain
members is scored on count anomaly, size consistency, edge alignment,
head-to-tail adjacency, gap periodicity, out-of-bounds/edge pinning
and cross-axis grid membership. Chains at or above the axis threshold
drop WHOLE (their members never reach the fits nor the grounded.png
audit); everything else survives untouched. All thresholds and
weights live in HallucinationConfig; every scored chain is reported
(score, verdict, per-indicator stats) in the diagnostics JSON
(hallucination_diag.json), and a debug PNG marks dropped (red) vs
kept (green) rects per view.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass
class HallucinationConfig:
    # ---- chain links (fractions of box size; scale/zoom invariant) ----
    cross_size_tol: float = 0.20    # Y-chain: pair widths within 20%
    cross_overlap: float = 0.55     # cross-axis interval overlap >= 55%
    along_size_tol: float = 0.60    # along sizes within 60% (tilt perspective)
    gap_frac: float = 0.25          # max neighbour gap: 25% of along size
                                    # (real aisles: >= 27% even for doubles)
    overlap_frac: float = 0.35      # max neighbour overlap (sloppy tiling)
    # ---- scoring (each component in [0,1]; score = weighted mean) ----
    min_chain: int = 3              # a back-to-back double is 2, never 3+
    n_ref: int = 5                  # count component saturates here
    tight_gap: float = 0.12         # head-to-tail: gap <= 12% of along size
    size_norm: float = 0.30         # cross-size deviation fully inconsistent
    align_norm: float = 0.40        # edge spread fully misaligned
    y_score_thr: float = 0.42       # cross-row tight stack: impossible
    x_score_thr: float = 0.75       # row-axis march: real over-split lives
                                    # around 0.63-0.69; OOB / grids push past
    # ---- weights ----
    w_count: float = 0.20
    w_size: float = 0.20
    w_align: float = 0.15
    w_adjacency: float = 0.25
    w_periodic: float = 0.10
    w_oob: float = 0.25
    w_grid: float = 0.15


HALLUCINATION_CFG = HallucinationConfig()


def _find_chains(idx, rects, axis: str, cfg: HallucinationConfig):
    """Maximal tight chains along `axis` ("y" = vertical stack,
    "x" = horizontal march) over the rect indices `idx`."""
    parent = {i: i for i in idx}

    def find(i):
        while parent[i] != i:
            parent[i] = parent[parent[i]]
            i = parent[i]
        return i

    for a in range(len(idx)):
        i = idx[a]
        ri = rects[i]
        for b in range(a + 1, len(idx)):
            j = idx[b]
            rj = rects[j]
            if axis == "y":          # cross = x, along = y
                ci0, ci1, ai0, ai1 = ri[0], ri[2], ri[1], ri[3]
                cj0, cj1, aj0, aj1 = rj[0], rj[2], rj[1], rj[3]
            else:                    # cross = y, along = x
                ci0, ci1, ai0, ai1 = ri[1], ri[3], ri[0], ri[2]
                cj0, cj1, aj0, aj1 = rj[1], rj[3], rj[0], rj[2]
            cw_i, cw_j = ci1 - ci0, cj1 - cj0        # cross sizes
            ah_i, ah_j = ai1 - ai0, aj1 - aj0        # along sizes
            if cw_i <= 0 or cw_j <= 0 or ah_i <= 0 or ah_j <= 0:
                continue
            # tiling edge: roughly equal cross sizes
            if abs(cw_i - cw_j) > cfg.cross_size_tol * min(cw_i, cw_j):
                continue
            # same column/row: substantial cross-axis overlap (a
            # leaning chain is still a chain)
            if (min(ci1, cj1) - max(ci0, cj0)
                    < cfg.cross_overlap * min(cw_i, cw_j)):
                continue
            # comparable along sizes (perspective scales gradually)
            if abs(ah_i - ah_j) > cfg.along_size_tol * max(ah_i, ah_j):
                continue
            # tight along-axis separation: a gap above gap_frac is an
            # aisle (real rows), a deeper overlap than overlap_frac is
            # another detection crossing the chain, not a tile
            g = max(aj0 - ai1, ai0 - aj1)
            mh = min(ah_i, ah_j)
            if g > cfg.gap_frac * mh or g < -cfg.overlap_frac * mh:
                continue
            parent[find(i)] = find(j)
    groups: dict[int, list[int]] = {}
    for i in idx:
        groups.setdefault(find(i), []).append(i)
    return [g for g in groups.values() if len(g) >= cfg.min_chain]


def _score_chain(members, rects, axis: str, W: int, H: int,
                 other_members: set, cfg: HallucinationConfig):
    """(score, stats) for one chain. `other_members`: rect indices
    already belonging to a chain on the OTHER axis (grid membership)."""
    n = len(members)
    if axis == "y":
        cross = [(float(r[0]), float(r[2])) for r in
                 (rects[i] for i in members)]
        along = [(float(r[1]), float(r[3])) for r in
                 (rects[i] for i in members)]
    else:
        cross = [(float(r[1]), float(r[3])) for r in
                 (rects[i] for i in members)]
        along = [(float(r[0]), float(r[2])) for r in
                 (rects[i] for i in members)]
    cw = [hi - lo for lo, hi in cross]
    ah = [hi - lo for lo, hi in along]
    med_cw = float(np.median(cw))
    med_ah = float(np.median(ah))

    # count anomaly (per CHAIN -- never a per-view verdict)
    count_c = min(n / cfg.n_ref, 1.0)
    # consistency of the tiling edge (cross sizes)
    dev = max(abs(s - med_cw) for s in cw) / med_cw
    size_c = float(np.clip(1.0 - dev / cfg.size_norm, 0.0, 1.0))
    # alignment: spread of BOTH cross edges relative to the size
    lo_sp = (max(lo for lo, _ in cross) - min(lo for lo, _ in cross)) / med_cw
    hi_sp = (max(hi for _, hi in cross) - min(hi for _, hi in cross)) / med_cw
    align_c = float(np.clip(1.0 - max(lo_sp, hi_sp) / cfg.align_norm,
                            0.0, 1.0))
    # neighbour gaps along the axis (ordered), head-to-tail adjacency
    order = sorted(range(n), key=lambda k: along[k][0] + along[k][1])
    gaps = [along[order[k + 1]][0] - along[order[k]][1]
            for k in range(n - 1)]
    tight = sum(1 for g in gaps if g <= cfg.tight_gap * med_ah)
    adj_c = tight / max(len(gaps), 1)
    # gap periodicity: a fixed pitch (gaps may be small but nonzero)
    gs = [abs(g) for g in gaps]
    if not gs or max(gs) < 1e-6:
        per_c = 1.0
    else:
        med_g = float(np.median(gs))
        per_c = float(np.clip(
            1.0 - (max(gs) - min(gs)) / max(0.30 * med_g, 1e-6), 0.0, 1.0))
    # out-of-bounds proxy: the parser clips, so raw OOB coordinates
    # surface as rects PINNED at the frame edge
    pinned = sum(1 for i in members
                 if (float(rects[i][0]) <= 0.5
                     or float(rects[i][1]) <= 0.5
                     or float(rects[i][2]) >= W - 0.5
                     or float(rects[i][3]) >= H - 0.5))
    oob_c = min(pinned / 2.0, 1.0)
    # grid membership: also chained on the other axis
    grid_c = len([i for i in members if i in other_members]) / n

    comps = {"count": round(count_c, 3), "size": round(size_c, 3),
             "align": round(align_c, 3), "adjacency": round(adj_c, 3),
             "periodic": round(per_c, 3), "oob": round(oob_c, 3),
             "grid": round(grid_c, 3)}
    weights = {"count": cfg.w_count, "size": cfg.w_size,
               "align": cfg.w_align, "adjacency": cfg.w_adjacency,
               "periodic": cfg.w_periodic, "oob": cfg.w_oob,
               "grid": cfg.w_grid}
    score = sum(weights[k] * comps[k] for k in comps) / sum(weights.values())
    stats = {"n": n, "n_pinned": pinned,
             "median_cross_size": round(med_cw, 1),
             "median_along_size": round(med_ah, 1),
             "gaps": [round(g, 1) for g in gaps], **comps}
    return float(score), stats


def filter_hallucination_rects(rects, W: int, H: int, view: str = "view",
                               cfg: HallucinationConfig | None = None):
    """Drop structured-coordinate hallucination chains from ONE
    grounding view.

    rects: [(x0, y0, x1, y1, label)] in the VIEW's pixel space (as
    parsed by judge._parse_ground_regions -- clipped, so OOB coords
    appear as edge-pinned rects). Returns (kept_rects, diag); diag
    lists every scored chain with score, verdict and per-indicator
    stats (empty chains list when nothing scored)."""
    cfg = cfg or HALLUCINATION_CFG
    diag = {"view": view, "n_in": len(rects), "n_kept": len(rects),
            "n_dropped": 0, "chains": []}
    if len(rects) < cfg.min_chain:
        return list(rects), diag
    idx = list(range(len(rects)))
    chains_y = _find_chains(idx, rects, "y", cfg)
    chains_x = _find_chains(idx, rects, "x", cfg)
    y_members = {i for ch in chains_y for i in ch}
    x_members = {i for ch in chains_x for i in ch}
    drop = set()
    for axis, chains, other, thr in (
            ("y", chains_y, x_members, cfg.y_score_thr),
            ("x", chains_x, y_members, cfg.x_score_thr)):
        for members in chains:
            score, stats = _score_chain(members, rects, axis, W, H,
                                        other, cfg)
            dropped = score >= thr
            if dropped:
                drop.update(members)
            diag["chains"].append({
                "axis": axis, "n": len(members),
                "score": round(score, 3), "threshold": thr,
                "dropped": dropped,
                "reason": ("cross-row tight stack: real rows always "
                           "have an aisle" if axis == "y" else
                           "row-axis repetitive march (over-split "
                           "needs a much higher bar)"),
                "stats": stats,
                "rects": [[round(float(rects[i][k]), 1) for k in range(4)]
                          for i in members]})
    kept = [r for i, r in enumerate(rects) if i not in drop]
    diag["n_kept"] = len(kept)
    diag["n_dropped"] = len(rects) - len(kept)
    diag["drop_indices"] = sorted(drop)
    return kept, diag


def save_debug_png(img, rects, diag, path: str):
    """Debug render of one filtered view: RED = dropped hallucination
    chains (with axis + score), GREEN = kept rects (best-effort)."""
    try:
        from PIL import Image, ImageDraw
        u8 = (np.clip(img, 0, 1) * 255).astype(np.uint8)[..., :3].copy()
        pil = Image.fromarray(u8)
        dr = ImageDraw.Draw(pil)
        drop = set(diag.get("drop_indices", ()))
        for i, r in enumerate(rects):
            color = (255, 50, 50) if i in drop else (60, 220, 60)
            dr.rectangle([float(r[0]), float(r[1]),
                          float(r[2]), float(r[3])], outline=color, width=3)
        for ch in diag["chains"]:
            if not ch["dropped"]:
                continue
            for r in ch["rects"]:
                dr.text((float(r[0]) + 3, float(r[1]) + 3),
                        f"{ch['axis']} {ch['score']:.2f}",
                        fill=(255, 255, 255))
        pil.save(path)
        return path
    except Exception as e:
        print(f"[halluc][debug] save failed ({type(e).__name__}: {e})")
        return None
