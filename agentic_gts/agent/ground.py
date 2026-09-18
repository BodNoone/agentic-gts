"""VLM global 2D grounding -- the human surveyor's FIRST pass.

The VLM looks at a top-down view of the whole room and outlines EVERY
device structure (a joined cabinet ROW counts as one region), geometry
turns each region into a full-depth 3D box, and a per-row split pass
(front+back face renders) resolves how many cabinets each row box
contains. There are no input boxes: the stage0 bootstrap byproducts
(device_footprint + z_top) drive the framing and the ceiling cut.

Why this kills the thin-fragment problem at its root: a region covers
the whole row footprint, so BOTH observed faces -- and the hollow
interior between them -- are inside one box from the start.

Division of labour (the project's standing contract): the VLM answers
WHERE things are in the image; geometry MEASURES the 3D boxes (edges
snap to point support, cuts snap to density-profile gaps).
"""
from __future__ import annotations

import math

import numpy as np

from agentic_gts.core.models import DeviceType, OrientedBox


# ---------- row-frame rotation ----------

def _rot_xy(pts: np.ndarray, yaw: float) -> np.ndarray:
    """Rotate XY by +yaw about the origin (z kept). Returns a copy."""
    c, s = math.cos(yaw), math.sin(yaw)
    out = np.array(pts, dtype=np.float64, copy=True)
    x, y = out[:, 0].copy(), out[:, 1].copy()
    out[:, 0] = c * x - s * y
    out[:, 1] = s * x + c * y
    return out


# ---------- grounding evidence render ----------

def _render_cut(top: float | None, mesh_mode: bool = False) -> float:
    """Nadir RENDER cut height, relative to the device top.

    min(top - 0.10, max(0.70 * top, 1.0)):
      * tall structures (top 2.1m) cut at 1.47m -- the VLM still sees
        the full rack footprint and most of the body, while anything
        the top reference dragged upward (trays, ceiling remnants)
        stays far above the cut;
      * LOW structures are protected by the max(., 1.0) floor and the
        top - 0.10 cap: a 0.9m-high device bank cuts at 0.8m, keeping
        nearly everything -- only tall structures get the relative trim;
      * no reference at all -> inf (render the full height band).

    mesh_mode: a mesh sampling makes overhead cable trays REAL dense
    gapless geometry that connects to the rack tops -- the anchored
    density walk runs right up them, so z_top itself is dragged to the
    tray top and the 0.70 trim no longer clears them (user report:
    trays visible in the groundview, confusing the VLM). Cut at HALF
    height instead: a nadir view only needs WHERE the tall devices
    are, and no device category is invisible at half its height.
    """
    if not top:
        return float("inf")
    top = float(top)
    frac = 0.50 if mesh_mode else 0.70
    return min(top - 0.10, max(frac * top, 1.0))


def _floor_map(points: np.ndarray, grid: float = 1.5, band: float = 1.0,
               min_pts: int = 30, mesh_mode: bool = False):
    """Per-tile LOCAL floor z for stepped rooms (small level changes).

    align_to_ground levels the DOMINANT floor to z=0; a raised (or
    sunken) section keeps its offset, and every absolute-z band cut
    downstream (render band, fit pool) then slices the wrong heights
    over that section: the raised slab passes the 0.30 device-band
    floor and renders as a bright sheet, and the ceiling cut sits a
    step too high over the section (user report: ceiling remnants in
    part of the groundview). The heightmap restores a per-SECTION
    zero: h = z - floor(x, y) makes every downstream threshold
    height-relative again -- exactly as if each section had been
    aligned independently.

    Per grid tile: P2 of the points in the near-ground band (P2 of
    the cloud + 1.0m). Devices STAND on the floor, so the bottom of
    any tile's z-range is that tile's floor -- and a raised slab's
    top coincides with the device bottoms standing on it, making the
    estimate robust to how much of the slab was reconstructed. Tiles
    without enough support fall back to the global dominant level
    (P2 of the whole cloud, ~= 0 after alignment).

    mesh_mode (geometry from a discretized MESH): no haze / floaters
    / under-floor diffusion exist, so a tile's floor is simply its
    MINIMUM z -- no near-ground band, no percentile, no support
    threshold (user simplification).

    Returns a callable f(x, y) -> floor z (scalar or array input).
    """
    P = np.asarray(points, dtype=np.float64)
    if len(P) < 200:
        base = float(np.percentile(P[:, 2], 2)) if len(P) else 0.0
        return lambda x, y: base
    base = float(np.percentile(P[:, 2], 2))
    # MESH geometry (user simplification): a mesh sampling carries no
    # under-floor diffusion / floaters, so a tile's floor is simply its
    # minimum z -- no near-ground band, no P2, no fallbacks. The GS
    # branch below keeps all of that: gaussian means smear below and
    # around the slab, and a bare min would chase haze.
    near = P if mesh_mode else P[P[:, 2] < base + band]
    min_pts_eff = 3 if mesh_mode else min_pts
    stat = (lambda z: float(np.min(z))) if mesh_mode \
        else (lambda z: float(np.percentile(z, 2)))
    if len(near) < (10 if mesh_mode else 100):
        return lambda x, y: base
    ix = np.floor(near[:, 0] / grid).astype(np.int64)
    iy = np.floor(near[:, 1] / grid).astype(np.int64)
    keys, inv = np.unique(np.column_stack([ix, iy]), axis=0,
                          return_inverse=True)
    order = np.lexsort((near[:, 2], inv))
    inv_s, z_s = inv[order], near[order][:, 2]
    starts = np.searchsorted(inv_s, np.arange(len(keys)))
    ends = np.searchsorted(inv_s, np.arange(len(keys)), side="right")
    fz = np.array([stat(z_s[s:e]) if e - s >= min_pts_eff
                   else base for s, e in zip(starts, ends)])
    i0, j0 = keys[:, 0].min(), keys[:, 1].min()
    G = np.full((keys[:, 0].max() - i0 + 1, keys[:, 1].max() - j0 + 1),
                base, dtype=np.float64)
    G[keys[:, 0] - i0, keys[:, 1] - j0] = fz

    def _fl(x, y):
        i = np.clip(np.floor(np.asarray(x, dtype=np.float64) / grid
                             ).astype(np.int64) - i0, 0, G.shape[0] - 1)
        j = np.clip(np.floor(np.asarray(y, dtype=np.float64) / grid
                             ).astype(np.int64) - j0, 0, G.shape[1] - 1)
        return G[i, j]
    return _fl


def _layout_frame(scene, yaw: float):
    """Rotated-frame AABB of the bootstrap layout (device cells /
    footprint), or None. The kept CELLS are rotated by the ACTUAL yaw
    and AABB'd ONCE: rotating the world-frame device_footprint AABB
    instead double-inflates for rotated layouts (AABB of a 45-deg
    row, then AABB of rotating that box)."""
    cells = scene.meta.get("device_cells")
    if cells is not None and len(cells):
        cr = _rot_xy(np.column_stack([cells,
                                      np.zeros(len(cells))]), -yaw)
        return cr.min(axis=0), cr.max(axis=0)
    fp = scene.meta.get("device_footprint")
    if fp:
        corners_w = np.array([[fp[0], fp[1]], [fp[2], fp[1]],
                              [fp[2], fp[3]], [fp[0], fp[3]]])
        cr = _rot_xy(np.column_stack([corners_w,
                                      np.zeros(4)]), -yaw)
        return cr.min(axis=0), cr.max(axis=0)
    return None


# a single nadir view's usable ground span: past this the camera
# climbs so high that a 0.6m cabinet renders a dozen pixels wide and
# grounding degrades into whole-room bands (user report: walls boxed
# as long rows). Below it, one view one VLM call, exactly as before.
_MAX_SINGLE_SPAN = 25.0
_TILE_OVERLAP = 2.5


def _tile_frames(layout):
    """Overlapping tile frames covering the rotated-frame layout AABB,
    or None when a SINGLE nadir view suffices.

    Tiling trades VLM calls for ground resolution (user directive:
    small rooms must NOT be tiled). Tiles overlap by _TILE_OVERLAP so
    every structure on a boundary appears WHOLE in at least one tile;
    the cross-tile duplicates and seams heal downstream in the shared
    world frame -- overlapping rects of one structure die in the
    IoU/containment dedup, tile-cut row pieces rejoin in the
    density-bridged adjacency merge.
    """
    if layout is None:
        return None
    lo, hi = layout

    def _axis(a0, a1):
        span = float(a1 - a0)
        if span <= _MAX_SINGLE_SPAN:
            return [(float(a0), float(a1))]
        n = int(np.ceil((span - _TILE_OVERLAP)
                        / (_MAX_SINGLE_SPAN - _TILE_OVERLAP)))
        w = (span + (n - 1) * _TILE_OVERLAP) / n   # exact cover
        step = w - _TILE_OVERLAP
        return [(a0 + i * step, a0 + i * step + w) for i in range(n)]

    xs, ys = _axis(lo[0], hi[0]), _axis(lo[1], hi[1])
    if len(xs) == 1 and len(ys) == 1:
        return None
    return [(x[0], y[0], x[1], y[1]) for x in xs for y in ys]


def _render_keep_mask(hg: np.ndarray, op: np.ndarray,
                      cut: float) -> np.ndarray:
    """Opacity-aware dual-band floor cut for the groundview GS render.

    SOLID gaussians (opacity >= 0.5) keep the fit-pool band from
    0.30m -- a low AC unit shows its FULL body -- while low-opacity
    ones (3DGS haze: diffuse floor floaters, the regional leak) stay
    under the 1.00m haze trim. A blanket 1.00m trim would cut sub-1m
    devices entirely; a blanket 0.30 would wash the view in floor
    haze (user reports: both, in sequence). The well-reconstructed
    floor slab is solid but sits at h ~ 0 (< 0.30) under the local
    floor map, so it stays out regardless.
    """
    solid = op >= 0.50
    keep = (solid & (hg > 0.30)) | (~solid & (hg > 1.00))
    if np.isfinite(cut):
        keep &= hg < cut
    return keep


def _render_topdown(scene, yaw: float, W: int = 1280, H: int = 1024,
                    frame=None):
    """Base top-down render, no overlays. Camera fitted over the
    yaw-rotated cloud (rows parallel to the image axes), then rotated
    back into world so the GS render and the pixel back-projection
    share one consistent camera. Shared by the grounding input view
    and the grounded-result audit view.

    True nadir: rows axis-aligned in the image (an axis-aligned image
    rectangle captures a row exactly, vertical rays carry no
    perspective dilation).

    Framing and ceiling cut both come from the stage0 bootstrap
    byproducts (scene.meta z_top + device_footprint): there is no
    hint-box input anymore. `frame` (rotated-frame AABB) overrides the
    framing for TILED views over a big layout (ground resolution).

    Returns (img_float, cam, W, H).
    """
    from agentic_gts.output.gs_render import (Cam, make_godview_cam,
                                              render_gs_view)
    points = np.asarray(scene.points, dtype=np.float64)
    # ceiling cut for the RENDER only (user decision: no need to be
    # conservative -- devices have real height and the nadir view only
    # needs each device's BASIC features for the VLM to outline it, not
    # a complete structure). A RELATIVE cut (70% of the device top)
    # also buys a large margin against top over-estimation: overhead
    # trays or a dense ceiling mesh dragging the reference top upward
    # still land above the cut. The FIT pool is cut independently
    # (top + 0.10 in ground_stage), so fitted box heights keep the true
    # rack top no matter how deep this renders.
    top = float(scene.meta["z_top"]) if scene.meta.get("z_top") else None
    cut = _render_cut(top, mesh_mode=bool(scene.meta.get("geometry_is_mesh")))
    # stepped-floor support: the heightmap restores a per-SECTION zero
    # (align_to_ground levels only the dominant floor). h = z - local
    # floor turns both cuts into HEIGHTS, valid over raised / sunken
    # sections alike: the raised slab drops out of the band (it is
    # that section's FLOOR, not a device) and the ceiling cut no
    # longer sits a step too high over it (user report: ceiling
    # remnants across part of the groundview in stepped rooms).
    fl = _floor_map(points,
                    mesh_mode=bool(scene.meta.get("geometry_is_mesh")))
    h = points[:, 2] - fl(points[:, 0], points[:, 1])
    # FLOOR cut for the RENDER: high on purpose (user directive --
    # devices are tall and the nadir view only needs WHERE they are,
    # not their full bodies). 3DGS floor gaussians are diffuse, their
    # means float tens of cm above the slab, and the old 0.30m cut let
    # them smear the whole view as a bright wash that buried the rows
    # (user report: the floor rendering into the groundview). 1.00m
    # (raised from 0.80): floor haze can float up to ~1m in badly
    # reconstructed regions, and the per-tile P2 floor estimate dips
    # BELOW the true slab where under-floor smear exceeds the
    # percentile -- the height-relative cut then leaks the taller
    # floor haze REGIONALLY (user report: part of the floor back in
    # the groundview). The FIT pool is cut independently in
    # ground_stage (0.30m, keeps the whole device body), so fitting
    # is unaffected.
    if np.isfinite(cut):
        band = points[(h > 1.00) & (h < cut)]
    else:
        band = points[h > 1.00]
    if len(band) < 100:
        # thin band (very low structures: AC banks etc.): relax toward
        # the old floor cut -- falling back to the RAW cloud would pull
        # the ceiling back into the view, which is exactly what the cut
        # exists to remove
        band = points[h > 0.30]
    pts_rot = _rot_xy(band, -yaw)
    # frame over the BOOTSTRAP layout, not the raw cloud bbox (user
    # directive: the cloud-framed version raised the camera to fit
    # walls too, and the racks rendered small). Walls were already
    # dropped as boundary cells, so the framing hugs the layout.
    # `frame` (tiled views over a big layout) overrides it.
    boxes_rot = []
    lo = hi = None
    if frame is not None:
        lo, hi = np.asarray(frame[:2], dtype=float), \
            np.asarray(frame[2:], dtype=float)
    else:
        lh = _layout_frame(scene, yaw)
        if lh is not None:
            lo, hi = lh
    if lo is not None:
        c = (lo + hi) / 2.0
        boxes_rot.append(OrientedBox(
            center=(float(c[0]), float(c[1]), 1.0),
            size=(float(hi[0] - lo[0]), float(hi[1] - lo[1]), 2.0),
            yaw=0.0))
    cam_r = make_godview_cam(pts_rot, boxes_rot, nadir=True, W=W, H=H)
    # rotate the camera back into world (rotation about z: the nadir
    # axis rotates with it)
    if abs(yaw) > 1e-9:
        c_, s_ = math.cos(yaw), math.sin(yaw)
        rz = lambda v: np.array([c_ * v[0] - s_ * v[1],
                                 s_ * v[0] + c_ * v[1], v[2]])
        cam = Cam(eye=rz(cam_r.eye), target=rz(cam_r.target),
                  up=rz(cam_r.up), fovy_deg=cam_r.fovy_deg, W=W, H=H)
    else:
        cam = cam_r
    # render: true 3DGS preferred, cam-consistent projected scatter else
    img = None
    gs_ply = scene.meta.get("gs_ply")
    if gs_ply:
        try:
            from agentic_gts.tools.gs_io import read_gaussian_ply
            gs = read_gaussian_ply(gs_ply)
            # OPACITY-AWARE dual-band floor cut (user report: part of
            # the floor back in the groundview, yet sub-1m devices --
            # AC banks, low cabinets -- must not be cut by a blanket
            # 1.00m trim). The rasterizer's SCALAR cut_z / cut_z_low
            # cannot express any of this -- keep_mask only.
            gm = np.asarray(gs.means, dtype=np.float64)
            hg = gm[:, 2] - fl(gm[:, 0], gm[:, 1])
            op = 1.0 / (1.0 + np.exp(
                -np.asarray(gs.raw_opacity, dtype=np.float64)))
            keep = _render_keep_mask(hg, op, cut)
            if keep.sum() < 100:       # very low structures: relax
                keep = hg > 0.30
                if np.isfinite(cut):
                    keep &= hg < cut
            img = render_gs_view(gs, (), cam,
                                 cut_z=float("inf"),
                                 cut_z_low=float("-inf"),
                                 keep_mask=keep)
        except Exception as e:
            print(f"[ground] GS render failed ({type(e).__name__}: {e}) "
                  f"-> scatter")
    if img is None:
        img = _projected_scatter(band, cam, W, H)
    return img, cam, W, H


def _projected_scatter(points: np.ndarray, cam, W: int, H: int) -> np.ndarray:
    """Cam-consistent top-down scatter (works without a CUDA rasterizer).

    Points are PROJECTED with the same camera the grounding uses, so a
    pixel rectangle back-projects identically for the GS and scatter
    paths -- the VLM's answer means the same thing either way."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    fig = plt.figure(figsize=(W / 100.0, H / 100.0), dpi=100)
    ax = fig.add_axes([0, 0, 1, 1])
    ax.set_xlim(0, W)
    ax.set_ylim(H, 0)          # pixel coords, y down
    ax.axis("off")
    if len(points):
        uv = cam.project_cv(points)
        zmax = max(float(points[:, 2].max()), 1.0)
        ax.scatter(uv[:, 0], uv[:, 1], s=0.15, c=points[:, 2],
                   cmap="viridis", vmin=0.0, vmax=zmax)
    fig.canvas.draw()
    arr = np.asarray(fig.canvas.buffer_rgba())[:, :, :3]
    plt.close(fig)
    return np.clip(arr.astype(np.float32) / 255.0, 0.0, 1.0)


# ---------- official-style audit plot (cookbook plot_bounding_boxes) ----------

# the cookbook's per-box color cycle (plot_bounding_boxes), minus
# near-black colors that vanish on a dark data-center render
_AUDIT_COLORS = [
    (220, 20, 60), (34, 139, 34), (0, 0, 255), (255, 215, 0),
    (255, 140, 0), (255, 105, 180), (138, 43, 226), (165, 42, 42),
    (128, 128, 128), (0, 206, 209), (0, 255, 255), (255, 0, 255),
    (0, 255, 0), (25, 25, 112), (0, 128, 128), (240, 128, 128),
]


def _draw_raw_regions(img: np.ndarray, raw_rects: list) -> np.ndarray:
    """Draw the VLM's RAW pixel rects the way the official cookbook's
    plot_bounding_boxes does: one DISTINCT color per region, 3-px
    rectangle outline, and the label text at the box's top-left.
    raw_rects: [(x0, y0, x1, y1, label)] in THIS image's pixel space."""
    from PIL import Image, ImageDraw, ImageFont
    u8 = (np.clip(img, 0, 1) * 255).astype(np.uint8)[..., :3].copy()
    pil = Image.fromarray(u8)
    dr = ImageDraw.Draw(pil)
    try:
        font = ImageFont.truetype("arial.ttf", 16)
    except OSError:
        font = ImageFont.load_default()
    for i, (x0, y0, x1, y1, label) in enumerate(raw_rects):
        color = _AUDIT_COLORS[i % len(_AUDIT_COLORS)]
        dr.rectangle(((int(x0), int(y0)), (int(x1), int(y1))),
                     outline=color, width=3)
        dr.text((int(x0) + 4, max(int(y0) - 18, 0)), label,
                fill=color, font=font)
    return np.asarray(pil, dtype=np.float32) / 255.0


def _draw_result_boxes(img: np.ndarray, cam, boxes) -> np.ndarray:
    """Solid red outlines for the grounded result boxes (result-only
    audit: the VLM answered on the clean base, the fit is shown apart)."""
    from PIL import Image, ImageDraw
    u8 = (np.clip(img, 0, 1) * 255).astype(np.uint8)[..., :3].copy()
    pil = Image.fromarray(u8)
    dr = ImageDraw.Draw(pil)
    red = (255, 60, 60)
    for b in boxes:
        z = b.center[2] + b.size[2] / 2.0
        cs = b.corners_2d()
        uv = cam.project_cv(np.column_stack([cs, np.full(len(cs), z)]))
        pts = [(int(round(p[0])), int(round(p[1]))) for p in uv]
        pts.append(pts[0])
        for a, c in zip(pts, pts[1:]):
            dr.line((a, c), fill=red, width=2)
    return np.asarray(pil, dtype=np.float32) / 255.0


def _save_grounded_png(base_img, cam, boxes, raw_rects, out_dir,
                       fname: str = "grounded.png") -> None:
    """The grounding audit image, drawn the way the official 2d_grounding
    cookbook plots its answers.

    Two layers over the clean view base the VLM answered on:
      - COLORED 3-px rectangles with labels = the VLM's RAW regions
        for this view (one distinct color per region, official
        plot_bounding_boxes style)
      - RED 2-px wireframes = the geometry-fitted final row boxes
    This separates WHAT the VLM said from what the point-support fit
    made of it -- when the result is wrong, the audit shows whether
    the VLM mis-boxed or the fit mangled it. One image per view
    (grounded.png / grounded_az90.png / grounded_az270.png).
    """
    import os
    try:
        from agentic_gts.output.gs_render import png_bytes
        img = _draw_result_boxes(_draw_raw_regions(base_img, raw_rects),
                                 cam, boxes)
        path = os.path.join(out_dir, fname)
        with open(path, "wb") as f:
            f.write(png_bytes(img))
        print(f"[ground] result render -> {path}")
    except Exception as e:
        print(f"[ground] result render failed ({type(e).__name__}: {e})")


def _save_grounded_fail_png(base_img, out_dir: str, why: str) -> None:
    """Failure is an audit result too: a loud banner instead of silence.

    grounded.png used to be written ONLY on success, so a failed
    grounding left nothing but the raw groundview_*.png renders --
    indistinguishable from 'grounding never ran' (user report: 'all I
    see are the raw renders'). Now the same filename always carries the
    outcome: regions + fitted wireframes when it worked, a red banner
    with the reason when it did not.
    """
    import os
    try:
        from PIL import Image, ImageDraw, ImageFont
        from agentic_gts.output.gs_render import png_bytes
        u8 = (np.clip(base_img, 0, 1) * 255).astype(np.uint8)[..., :3].copy()
        pil = Image.fromarray(u8)
        dr = ImageDraw.Draw(pil)
        try:
            font = ImageFont.truetype("arialbd.ttf", 28)
        except OSError:
            font = ImageFont.load_default()
        dr.rectangle(((0, 0), (pil.width, 46)), fill=(180, 0, 0))
        dr.text((10, 9), f"GROUNDING FAILED - {why}"[:110],
                fill=(255, 255, 255), font=font)
        path = os.path.join(out_dir, "grounded.png")
        with open(path, "wb") as f:
            f.write(png_bytes(np.asarray(pil, dtype=np.float32) / 255.0))
        print(f"[ground] FAILURE audit render -> {path} ({why})")
    except Exception as e:
        print(f"[ground] failure render failed ({type(e).__name__}: {e})")


def _region_axis_span(v: np.ndarray, cell: float = 0.05):
    """Axis extent for region fitting: peak-peeling strong clusters
    with a percentile sanity floor.

    Why not plain _robust_span here (user report: rects over real
    devices lost their boxes): the strong-bin threshold is relative
    to the GLOBAL peak, and 3DGS reconstructs the two rack faces at
    very different densities -- the wall-facing / occluded side is
    starved, its bins fall below the front-face peak, the depth span
    collapses to ONE face (~5cm) and the sliver guard rejects the
    whole region. Peeling fixes it: accept the strongest bin's
    cluster (connected bins >= 20% of that pass's peak), zero it
    out, repeat while the next pass's peak >= 6% of the first peak
    -- starved faces and thin real sections survive, while haze
    bins (~1% of the face peak) never do. The connect/keep cuts are
    deliberately GENEROUS (user report: fitted red boxes came out
    far SMALLER than the VLM rects -- moderate-density real extent
    was being trimmed): the stageG box is a SEED, the local refine
    tightens it later; a seed that under-covers the device has no
    recovery path.

    Safety floor: when the peeled span still covers < 65% of the
    P0.5-P99.5 extent, the structure is more heterogeneous than the
    bins can see (or the slice was too thin) -- return the percentile
    extent instead of letting the fit collapse: a slightly loose box
    is correctable by the local refine, a rejected region is lost
    recall. Returns None only when there is nothing to bin."""
    v = np.asarray(v, dtype=float)
    if len(v) < 30:
        return None
    p_lo, p_hi = (float(x) for x in np.percentile(v, [0.5, 99.5]))
    lo, hi = float(v.min()), float(v.max())
    nb = int(np.floor((hi - lo) / cell)) + 2
    edges = lo + cell * np.arange(nb + 1)
    hist, _ = np.histogram(v, bins=edges)
    first_peak = float(hist.max())
    if first_peak < 3.0:
        return p_lo, p_hi
    keep_thr = 0.06 * first_peak
    remaining = hist.astype(float).copy()
    kept = np.zeros(len(hist), dtype=bool)
    while True:
        peak = float(remaining.max())
        if peak < max(keep_thr, 3.0):
            break
        thr = 0.20 * peak
        i0 = int(np.argmax(remaining))
        kept[i0] = True
        remaining[i0] = 0.0
        i = i0 - 1
        while i >= 0 and remaining[i] >= thr:
            kept[i] = True
            remaining[i] = 0.0
            i -= 1
        j = i0 + 1
        while j < len(remaining) and remaining[j] >= thr:
            kept[j] = True
            remaining[j] = 0.0
            j += 1
    idx = np.where(kept)[0]
    if not len(idx):
        return p_lo, p_hi
    s_lo, s_hi = float(edges[idx[0]]), float(edges[idx[-1] + 1])
    if s_hi - s_lo < 0.65 * (p_hi - p_lo):
        return p_lo, p_hi
    return s_lo, s_hi


def _fit_region_box(points: np.ndarray, rect, min_pts: int = 60,
                    floor_z: float = 0.0):
    """Fit a full-depth OBB (yaw=0; points already in the row-aligned
    frame) to the points inside a grounded 2D rect.

    The rect is the VLM's coarse outline; the DEVICE-BAND point support
    snaps the edges. Horizontal extent is fitted on a MIDDLE z-slice of
    the structure (user insight: face sheets are vertical, so any
    knee-height band cuts the exact same footprint as the whole cloud):
    the slice dodges floor creep and top floaters entirely, and the
    peak-peeling estimator (_region_axis_span) drops aisle haze -- a
    low plateau that percentile trimming cannot cut (haze is often >
    the 0.5% a P0.5-P99.5 removes) while keeping starved occluded
    faces that a global-peak threshold would cut. z comes from the
    device band: devices stand ON the ground, so the box BOTTOM is
    the local floor (floor_z; 0 = the dominant level, the raised-slab
    height over a stepped section) and the top is the anchored
    density-connected run's top; the box HEIGHT is the difference --
    without floor_z a stepped-section box would run a step too deep
    and a step too tall.

    Guards reject hallucinated regions (no support) and floor patches
    (no height): a VLM box drawn over empty floor never becomes a real
    device box.
    """
    def _reject(reason: str, n_dev: int = 0) -> None:
        # every drop is visible in the console -- a rect that silently
        # vanishes between grounded.png and stageG_ground.png is
        # undebuggable otherwise (user report: "way fewer boxes than
        # raw regions, many wrongly deleted")
        print(f"[ground] region rejected ({reason}): "
              f"rect=({x0:.2f},{y0:.2f})-({x1:.2f},{y1:.2f}) "
              f"n_pts={len(pts)} n_dev={n_dev}")

    x0, y0, x1, y1 = rect
    m = ((points[:, 0] >= x0) & (points[:, 0] <= x1) &
         (points[:, 1] >= y0) & (points[:, 1] <= y1))
    pts = points[m]
    if len(pts) < min_pts:
        _reject(f"no point support (<{min_pts})")
        return None
    # device band, floor-relative: exclude that section's floor texture
    dev = pts[pts[:, 2] > floor_z + 0.30]
    if len(dev) < max(30, min_pts // 2):
        _reject("floor patch, no structure above 0.30m", n_dev=len(dev))
        return None
    # anchored column top, P99.5 fallback: the rect's points form one
    # union column (mixed-height cabinets, all standing on the ground),
    # so the density-connected run's top is the row's true tallest --
    # a percentile lets floating overhead clutter inside the rect drag
    # it higher (same failure the hint-free bootstrap z_top had)
    from agentic_gts.agent.mask_refine import _anchored_top
    _at = _anchored_top(dev[:, 2])
    z_top = float(_at) if _at is not None \
        else float(np.percentile(dev[:, 2], 99.5))
    height = z_top - floor_z
    if height < 0.50:
        _reject(f"too short for a device (height={height:.2f}m "
                f"over floor {floor_z:.2f})")
        return None
    # middle z-slice: [0.35, 0.75] x height above the LOCAL floor --
    # cuts every vertical face of a tall rack, stays above floor
    # texture, below trays/floaters
    zc0 = floor_z + max(0.30, 0.35 * height)
    zc1 = max(zc0 + 0.10, floor_z + 0.75 * height)
    core = dev[(dev[:, 2] >= zc0) & (dev[:, 2] <= zc1)]
    if len(core) < 30:
        core = dev                   # thin structure: whole band
    sx = _region_axis_span(core[:, 0])
    sy = _region_axis_span(core[:, 1])
    if sx is not None and sy is not None:
        (x_lo, x_hi), (y_lo, y_hi) = sx, sy
    else:                            # too sparse to bin: percentile fit
        x_lo, y_lo = np.percentile(dev[:, :2], 0.5, axis=0)
        x_hi, y_hi = np.percentile(dev[:, :2], 99.5, axis=0)
    dx, dy = float(x_hi - x_lo), float(y_hi - y_lo)
    if dx < 0.30 or dy < 0.20:
        _reject(f"sliver (span {dx:.2f} x {dy:.2f}m; "
                f"core={len(core)} pts, slice z "
                f"[{zc0:.2f},{zc1:.2f}])")
        return None                  # sliver, not a structure
    c = np.array([(x_lo + x_hi) / 2.0, (y_lo + y_hi) / 2.0])
    # Ride the LONG side on the yaw axis (size[0]): a row that runs
    # along the rotated-y axis still fits here as (dx, dy) with
    # yaw=0 -- but then the box's yaw axis is its THICKNESS, and
    # refine_box (which projects along-row spans on the seed's yaw
    # axis) splits the row ACROSS its depth (user report: a joined
    # row split into 3 pieces along the thickness, not the row).
    if dy > dx:
        return OrientedBox(
            center=(float(c[0]), float(c[1]), floor_z + height / 2.0),
            size=(dy, dx, height), yaw=math.pi / 2.0,
            device_type=DeviceType.RACK,
            meta={"n_pts": len(dev)})
    return OrientedBox(center=(float(c[0]), float(c[1]),
                               floor_z + height / 2.0),
                       size=(dx, dy, height), yaw=0.0,
                       device_type=DeviceType.RACK,
                       meta={"n_pts": len(dev)})


# a rect the VLM drew around TWO opposing rows (front + back, aisle
# between) fits as one box with the UNION depth; no later stage can
# split across the thickness (stageC spans project on the ROW axis)
_MAX_DEVICE_DEPTH = 1.8
_MIN_DEVICE_DEPTH = 0.40
_SPLIT_MIN_GAP = 0.30


def _cross_gap_split(v: np.ndarray, peak_frac: float = 0.25) -> float | None:
    """Split coordinate of the most BALANCED interior weak run in a 1-D
    cross-axis density profile, or None.

    A gap qualifies when it is >= _SPLIT_MIN_GAP wide, interior (not
    touching the profile edges) and BOTH sides keep a >=
    _MIN_DEVICE_DEPTH strong span: a run with a single face-sheet on
    one side is a rack INTERIOR (hollow row), not an aisle -- splitting
    there shreds one row into its two faces. Among qualifying gaps the
    most balanced split wins (minimises the wider side's strong span);
    the caller recurses on any side still deeper than
    _MAX_DEVICE_DEPTH, which handles 3+ rows in one rect.
    """
    if len(v) < 60:
        return None
    cell = 0.05
    lo, hi = float(v.min()), float(v.max())
    n = int(np.ceil((hi - lo) / cell)) + 1
    if n < 5:
        return None
    edges = lo + cell * np.arange(n + 1)
    hist, _ = np.histogram(v, bins=edges)
    peak = int(hist.max())
    if peak == 0:
        return None
    strong = hist >= peak_frac * peak
    si = np.nonzero(strong)[0]
    if len(si) == 0:
        return None
    first, last = int(si[0]), int(si[-1])
    best = None                     # (wider_side_span, gap_lo, gap_hi)
    i = first
    while i <= last:
        if strong[i]:
            i += 1
            continue
        j = i
        while j <= last and not strong[j]:
            j += 1
        # weak run bins [i, j-1] with strong bins at i-1 and j
        g_lo, g_hi = float(edges[i]), float(edges[j])
        if g_hi - g_lo >= _SPLIT_MIN_GAP:
            l_span = g_lo - float(edges[first])
            r_span = float(edges[last + 1]) - g_hi
            if l_span >= _MIN_DEVICE_DEPTH and r_span >= _MIN_DEVICE_DEPTH:
                wider = max(l_span, r_span)
                if best is None or wider < best[0]:
                    best = (wider, g_lo, g_hi)
        i = j
    if best is None:
        return None
    return 0.5 * (best[1] + best[2])


def _fit_region_boxes(points: np.ndarray, rect, min_pts: int = 60,
                      floor_z: float = 0.0) -> list:
    """Fit one rect, then split DEEP fits: a rect the VLM drew around
    TWO opposing rows (front + back, an aisle between) fits as ONE box
    with the union depth, and nothing downstream can split across the
    thickness -- stageC's spans project on the ROW axis, so the two
    rows stay glued forever (user report). No device category
    (rack / cabinet / AC) is deeper than _MAX_DEVICE_DEPTH, so a deeper
    fit is by construction multiple structures: split at the cross
    profile's most balanced weak run (the aisle; hollow-rack interiors
    fail the min-side-depth rule in _cross_gap_split) and refit each
    side, recursively. Sides whose refit fails the guards (a wall
    strip, a sliver) are dropped by _fit_region_box itself; if NO side
    survives the original whole box is kept (recall first)."""
    bb = _fit_region_box(points, rect, min_pts, floor_z)
    if bb is None:
        return []
    axis = 1 if abs(float(bb.yaw)) < 1e-6 else 0   # cross axis of the fit
    if bb.size[1] <= _MAX_DEVICE_DEPTH:
        return [bb]
    x0, y0, x1, y1 = rect
    m = ((points[:, 0] >= x0) & (points[:, 0] <= x1) &
         (points[:, 1] >= y0) & (points[:, 1] <= y1))
    dev = points[m]
    dev = dev[dev[:, 2] > floor_z + 0.30]
    s = _cross_gap_split(dev[:, axis])
    if s is None:
        print(f"[ground] deep fit (depth {bb.size[1]:.2f}m) with no "
              f"splittable aisle gap -> kept whole (back-to-back rows "
              f"have no gap; the local refine still splits along-row)")
        return [bb]
    print(f"[ground] deep fit (depth {bb.size[1]:.2f}m) -> split at "
          f"{'y' if axis else 'x'}={s:.2f} (two rows in one rect)")
    subs = ((x0, y0, x1, s), (x0, s, x1, y1)) if axis == 1 \
        else ((x0, y0, s, y1), (s, y0, x1, y1))
    out = []
    for sub in subs:
        out.extend(_fit_region_boxes(points, sub, min_pts, floor_z))
    return out or [bb]


def _merge_adjacent_boxes(boxes: list, pts_fit: np.ndarray, yaw: float,
                          bridge_tol: float = 0.50,
                          min_gap_pts: int = 15,
                          density_ratio: float = 0.30,
                          floor_at=None) -> list:
    """Merge tightly-ADJACENT grounded boxes; splitting is stageC's job.

    The VLM sometimes over-splits ONE physical structure into several
    tight rects (a regular layout reads as several bands); each
    rect fits its own box and the seam never heals later -- stageC only
    SPLITS, never merges. Candidate pairs (same orientation bucket,
    real band overlap on the perpendicular axis, gap <= bridge_tol)
    merge when the device band DENSELY fills the junction.

    Density, not bare counts (user report: two rows a clear aisle
    apart got merged): the fitted AABBs are percentile-snug to their
    own rect's points, and 3DGS aisle haze inflates both facing
    edges, so two SEPARATE rows can arrive TOUCHING or overlapping in
    the fitted frame. A probe slab is placed strictly between the two
    boxes' facing surfaces -- inside the gap when apart, inside the
    overlap band when the fits cross, around the junction when they
    kiss -- and must carry >= min_gap_pts points at a density >=
    density_ratio x the sparser box's own device-band density. A real
    over-split seam IS device interior (same density as the boxes);
    haze is orders of magnitude sparser and fails the ratio even when
    it outnumbers the count threshold. Each union is REFITTED to point
    support (never boundary-united: noise would inflate the edges);
    the local refine then does the true splitting.
    """
    n = len(boxes)
    if n < 2:
        return boxes
    # row-frame AABB + orientation bucket (long side on x or on y)
    rects, buckets = [], []
    for b in boxes:
        cs = _rot_xy(np.column_stack([b.corners_2d(),
                                      np.zeros(4)]), -yaw)
        rects.append((float(cs[:, 0].min()), float(cs[:, 1].min()),
                      float(cs[:, 0].max()), float(cs[:, 1].max())))
        buckets.append(int(round((b.yaw - yaw) / (math.pi / 2.0))) % 2)
    # per-box device-band density from the SAME pool the probe uses
    # (surfaces are dense, an inflated fit barely dilutes it)
    dens = []
    for r in rects:
        m = ((pts_fit[:, 0] >= r[0]) & (pts_fit[:, 0] <= r[2]) &
             (pts_fit[:, 1] >= r[1]) & (pts_fit[:, 1] <= r[3]))
        area = max((r[2] - r[0]) * (r[3] - r[1]), 1e-6)
        dens.append(float(m.sum()) / area)
    parent = list(range(n))

    def _find(i: int) -> int:
        while parent[i] != i:
            parent[i] = parent[parent[i]]
            i = parent[i]
        return i

    for i in range(n):
        for j in range(i + 1, n):
            if buckets[i] != buckets[j]:
                continue          # an L-junction is two structures
            a, b = rects[i], rects[j]
            for axis in (0, 1):
                o = 1 - axis
                gap = max(a[axis] - b[axis + 2], b[axis] - a[axis + 2])
                if gap > bridge_tol:
                    continue      # not adjacent on this axis
                p_lo = max(a[o], b[o])
                p_hi = min(a[o + 2], b[o + 2])
                if p_hi - p_lo < 0.20:
                    continue      # corner kiss, no shared band
                # probe slab strictly BETWEEN the two facing surfaces:
                # inside the gap when apart, inside the overlap band
                # when the (haze-inflated) fits cross, around the
                # junction when they kiss. A real over-split seam holds
                # device interior there; a haze-inflated "seam" holds
                # only the haze that inflated the fits in the first
                # place -- the density test tells them apart.
                c0 = max(a[axis], b[axis])          # right-most left edge
                c1 = min(a[axis + 2], b[axis + 2])  # left-most right edge
                if c1 > c0 + 0.10:                  # fits overlap
                    s_lo, s_hi = c0 + 0.05, c1 - 0.05
                elif c1 >= c0 - 0.10:
                    # kiss / tiny overlap / tiny gap (percentile-trimmed
                    # fits of rects that TOUCH leave a ~cm seam): band
                    # around the junction -- a real seam holds the
                    # continuous device sheets through it
                    mid = 0.5 * (c0 + c1)
                    s_lo, s_hi = mid - 0.10, mid + 0.10
                else:                               # real gap
                    s_lo, s_hi = c1 + 0.05, c0 - 0.05
                if s_hi - s_lo < 0.05:
                    continue          # degenerate probe, no evidence
                m = ((pts_fit[:, axis] >= s_lo) &
                     (pts_fit[:, axis] <= s_hi) &
                     (pts_fit[:, o] >= p_lo) & (pts_fit[:, o] <= p_hi))
                n_br = int(m.sum())
                if n_br < min_gap_pts:
                    continue          # nothing bridging at all
                d_br = n_br / ((s_hi - s_lo) * (p_hi - p_lo))
                if d_br >= density_ratio * min(dens[i], dens[j]):
                    parent[_find(i)] = _find(j)
                    break
    comps: dict[int, list[int]] = {}
    for i in range(n):
        comps.setdefault(_find(i), []).append(i)
    out, n_union = [], 0
    for members in comps.values():
        if len(members) == 1:
            out.append(boxes[members[0]])
            continue
        rs = [rects[i] for i in members]
        u = (min(r[0] for r in rs), min(r[1] for r in rs),
             max(r[2] for r in rs), max(r[3] for r in rs))
        uw = _rot_xy(np.array([[(u[0] + u[2]) / 2.0,
                                (u[1] + u[3]) / 2.0, 0.0]]), yaw)[0]
        fz = float(floor_at(uw[0], uw[1])) if floor_at is not None else 0.0
        # SINGULAR fit on purpose: these members were bridged by a
        # dense device band -- one verified continuous structure -- so
        # the union refit must NOT re-split it across that bridge (the
        # deep-split belongs to the per-RECT path, where the rect
        # itself is the only evidence)
        bb = _fit_region_box(pts_fit, u, floor_z=fz)
        if bb is None:
            out.extend(boxes[i] for i in members)   # keep the pieces
            continue
        c = _rot_xy(np.array([[bb.center[0], bb.center[1], 0.0]]), yaw)[0]
        out.append(OrientedBox(
            center=(float(c[0]), float(c[1]), bb.center[2]),
            size=bb.size, yaw=yaw + float(bb.yaw),
            device_type=DeviceType.RACK,
            meta={"grounded": True, "n_pts": bb.meta.get("n_pts", 0),
                  "merged_from": len(members)}))
        n_union += 1
    if n_union:
        print(f"[ground] adjacency merge: {n} -> {len(out)} boxes "
              f"({n_union} union(s) refitted to point support; "
              f"splitting is the local refine's job)")
    return out


# ---------- grounding stage ----------


def ground_stage(scene, judge, out_dir: str | None = None) -> bool:
    """Replace scene.boxes with VLM-grounded per-region boxes.

    ONE global NADIR view (rows axis-aligned, exact footprint capture,
    vertical rays carry no perspective dilation). Regions are
    back-projected and point-support fitted -- each region its OWN
    box, unmerged. The per-box local refinement (front + side
    renders -> VLM SAM boxes -> mask -> back-projected points ->
    split-corrected seed) runs downstream.

    The stage0 bootstrap byproducts (scene.meta z_top +
    device_footprint -- device vertical surfaces, walls/ceiling
    excluded) drive the nadir framing and the ceiling cut; there is
    no box input of any kind.

    False = grounding unavailable (mock backend / VLM failure / no
    region survived the point-support guards): the scene stays empty.
    """
    import os
    from agentic_gts.output.gs_render import png_bytes, unproject_ground
    if not (scene.meta.get("device_footprint")
            and scene.meta.get("z_top")):
        print("[ground] no stage0 bootstrap footprint / z_top "
              "-> nothing to ground")
        return False
    yaw = float(scene.meta.get("yaw", 0.0) or 0.0)
    # ---- views: one nadir view, or TILES over a big layout ----
    # A single view must fit the whole layout; past ~25m of span the
    # camera climbs so high that cabinets render a dozen pixels wide
    # and grounding degrades into whole-room bands (user report:
    # walls boxed as long rows). Tiles overlap so boundary structures
    # appear whole in at least one; duplicates/seams heal in the
    # shared world frame (dedup + density-bridged merge). Every tile
    # is one extra VLM call, so a layout that fits stays single-view
    # (user directive: small rooms must not be tiled).
    try:
        tiles = _tile_frames(_layout_frame(scene, yaw))
    except Exception:
        tiles = None
    views = []                       # (img, cam, W, H, fname, rects)
    view_specs = [("groundview.png", None)] if tiles is None else \
        [(f"groundview_t{i}.png", fr) for i, fr in enumerate(tiles)]
    if tiles is not None:
        print(f"[ground] layout exceeds a single nadir view "
              f"(>{_MAX_SINGLE_SPAN:.0f}m span) -> {len(tiles)} tiled "
              f"views ({len(tiles)} VLM calls)")
    for fname, fr in view_specs:
        try:
            img, cam, W, H = _render_topdown(scene, yaw, frame=fr)
            png = png_bytes(img)     # CLEAN view: no overlays
        except Exception as e:
            print(f"[ground] nadir render failed ({type(e).__name__}: {e})")
            continue
        png_path = None
        if out_dir:
            png_path = os.path.join(out_dir, fname)
            try:
                with open(png_path, "wb") as f:
                    f.write(png)
            except Exception as e:
                print(f"[ground] png save failed ({type(e).__name__})")
                png_path = None
        rects = judge.ground_regions(png, W, H, png_path=png_path)
        print(f"[ground] view {fname}: {len(rects)} regions")
        views.append((img, cam, W, H, fname, rects))
    if not views:
        return False                 # every render failed (logged above)
    if not any(v[5] for v in views):
        print("[ground] VLM returned no usable regions")
        if out_dir:
            _save_grounded_fail_png(views[0][0], out_dir,
                                    "VLM returned no usable regions")
        return False
    # FIT points: the device band only. The render band cuts lower
    # (relative to the device top), but the FIT must keep the rack
    # top, so cut at z_top + 0.1: everything above (ceiling / cable
    # trays -- the raw cloud still carries them) is excluded. Ceiling
    # points span the WHOLE room in XY, so even a correct rect whose
    # fit included them produced a tray-height box hugging the loose
    # rect edges (user report: red boxes all too large and wrong while
    # the raw colored rects were right). Both cuts are HEIGHT-relative
    # to the local floor (stepped rooms: a raised section's slab is
    # that section's floor, and its racks are NOT a step taller).
    P = np.asarray(scene.points, dtype=np.float64)
    fl = _floor_map(P, mesh_mode=bool(scene.meta.get("geometry_is_mesh")))
    h_fit = P[:, 2] - fl(P[:, 0], P[:, 1])
    fit_top = float(scene.meta.get("z_top", 2.5) or 2.5)
    pts_fit = _rot_xy(P[(h_fit > 0.30) & (h_fit <= fit_top + 0.10)], -yaw)
    if len(pts_fit) < 100:
        pts_fit = _rot_xy(P[h_fit > 0.30], -yaw)

    def _frame_rect(cam, r, z_plane):
        uv = np.array([[r[0], r[1]], [r[2], r[1]], [r[2], r[3]], [r[0], r[3]]],
                      dtype=float)
        corners_w = unproject_ground(cam, uv, z_plane)[:, :2]
        # image rects are axis-aligned in the ROW frame: rotate the world
        # corners back into it and take the AABB
        corners_r = _rot_xy(np.column_stack([corners_w,
                                             np.zeros(len(corners_w))]),
                            -yaw)[:, :2]
        return (float(corners_r[:, 0].min()), float(corners_r[:, 1].min()),
                float(corners_r[:, 0].max()), float(corners_r[:, 1].max()))

    # Each grounded rect is fitted as its OWN box; tightly-ADJACENT
    # over-split pieces of one structure merge right after the dedup
    # (below) -- the local refine then splits the merged unions, so
    # structure membership is decided on refinement evidence, never
    # pre-merged beyond touching pieces. Rects from ALL views (single
    # or tiled) fit through their OWN camera -- pixel coords only mean
    # something relative to the view they were drawn on.
    boxes = []
    for cam_v, r in [(v[1], r) for v in views for r in v[5]]:
        rect_r = _frame_rect(cam_v, r, 1.0)
        # the rect's own LOCAL floor (stepped rooms): the section's
        # slab height, looked up at the rect's world centre
        cw = _rot_xy(np.array([[(rect_r[0] + rect_r[2]) / 2.0,
                                (rect_r[1] + rect_r[3]) / 2.0, 0.0]]),
                     yaw)[0]
        # _fit_region_boxes (plural): a deep fit -- the VLM drew ONE
        # rect around two opposing rows -- splits at the aisle here,
        # before the box enters the pipeline (stageC can only split
        # along the row axis)
        for bb in _fit_region_boxes(pts_fit, rect_r,
                                    floor_z=float(fl(cw[0], cw[1]))):
            c = _rot_xy(np.array([[bb.center[0], bb.center[1], 0.0]]),
                        yaw)[0]
            # bb.yaw is 0 (row along the rotated-x axis) or pi/2 (row
            # along rotated-y): both rotate into the world by ADDING
            # the frame yaw
            box = OrientedBox(center=(float(c[0]), float(c[1]),
                                      bb.center[2]),
                              size=bb.size, yaw=yaw + float(bb.yaw),
                              device_type=DeviceType.RACK,
                              meta={"grounded": True,
                                    "n_pts": bb.meta.get("n_pts", 0)})
            boxes.append(box)
    if not boxes:
        print("[ground] no region survived the point-support guards")
        if out_dir:
            _save_grounded_fail_png(views[0][0], out_dir,
                                    "no region survived point-support guards")
        return False
    n_rects_total = sum(len(v[5]) for v in views)
    print(f"[ground] {n_rects_total} VLM regions "
          f"({len(views)} view(s)) -> {len(boxes)} fitted boxes")
    # DEDUPLICATE: the VLM often outlines the SAME device more than
    # once (overlapping rects in one reply). Each rect fits its own
    # near-identical box with a DIFFERENT box_id, and the per-box local
    # refinement then renders mask_prompt_<id>_front.png per box --
    # one device, several duplicate renders (user report). Drop a box
    # when it overlaps a better-supported kept fit (IoU >= 0.5) OR is
    # >= 85% CONTAINED in one: a small box nested inside a big row box
    # has IoU = area ratio (< 0.5) but containment ~1.0 -- pure IoU let
    # the nesting through (user report: big box with small boxes
    # inside on the audit render). Containment also catches the
    # cross-yaw nesting the adjacency merge cannot (different
    # orientation buckets never enter it).
    dedup = []
    for b in sorted(boxes, key=lambda x: -int(x.meta.get("n_pts", 0))):
        if any(b.iou_2d(d) >= 0.5 or b.containment_2d(d) >= 0.85
               for d in dedup):
            continue
        dedup.append(b)
    if len(dedup) < len(boxes):
        print(f"[ground] dropped {len(boxes) - len(dedup)} duplicate/contained "
              f"box(es) (IoU >= 0.5 or >= 85% contained in a "
              f"better-supported fit)")
    boxes = dedup
    # MERGE tightly-adjacent over-split pieces (user request): the VLM
    # sometimes outlines one physical structure as several tight rects;
    # each fits its own box and the seam never heals (stageC only
    # splits, never merges). Touching / point-bridged boxes merge into
    # a point-support-refitted union; the true splitting is the local
    # refine's job.
    boxes = _merge_adjacent_boxes(boxes, pts_fit, yaw, floor_at=fl)
    scene.boxes = boxes
    # result audit: one image per view -- the view's own raw VLM rects
    # (colored) plus the final fitted boxes (red) projected through the
    # same camera. Tiled views draw ALL boxes (cross-tile ones project
    # outside the frame), so each tile's audit stays self-contained.
    if out_dir:
        for img_v, cam_v, _, _, fname_v, rects_v in views:
            _save_grounded_png(
                img_v, cam_v, boxes, rects_v, out_dir,
                fname=("grounded.png" if tiles is None else
                       fname_v.replace("groundview", "grounded")))
    return True
