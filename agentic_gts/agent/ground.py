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


# MESH floor-map tile size: FINE, so a tile straddling a step does not
# take the lower floor for the whole (coarse) cell. ~0.3m localises the
# step to a sub-grid strip; a mesh is dense enough for the >=3-pt guard.
_MESH_FLOOR_GRID = 0.3

# a MESH tile whose floor reads this far ABOVE the global floor has no
# floor points -- only overhead structure (trays / ceiling / a device in
# a floor-less cell) -- and must fall back, not report the structure's z
# as the floor (user report: floor map max 4.6m, boxes raised).
_MESH_FLOOR_BAND = 1.5


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
    threshold (user simplification). Mesh tiles are FINE
    (_MESH_FLOOR_GRID): a COARSE tile straddling a step takes the
    LOWER floor, so the raised section inside it reads h = step and
    renders as structure (user report: floor gaussians back in the
    groundview on a stepped mesh). A fine tile localises the step to a
    sub-grid strip; tiles are dense so the >=3-point guard still holds.

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

    def _build(g):
        """Tile map at resolution g; NaN where a tile is under-supported."""
        ix = np.floor(near[:, 0] / g).astype(np.int64)
        iy = np.floor(near[:, 1] / g).astype(np.int64)
        keys, inv = np.unique(np.column_stack([ix, iy]), axis=0,
                              return_inverse=True)
        order = np.lexsort((near[:, 2], inv))
        inv_s, z_s = inv[order], near[order][:, 2]
        starts = np.searchsorted(inv_s, np.arange(len(keys)))
        ends = np.searchsorted(inv_s, np.arange(len(keys)), side="right")
        i0, j0 = int(keys[:, 0].min()), int(keys[:, 1].min())
        G = np.full((int(keys[:, 0].max()) - i0 + 1,
                     int(keys[:, 1].max()) - j0 + 1), np.nan,
                    dtype=np.float64)
        for k, (s, e) in enumerate(zip(starts, ends)):
            if e - s < min_pts_eff:
                continue
            v = stat(z_s[s:e])
            # a MESH tile with NO floor points (only overhead structure)
            # would read that structure's z as the floor; reject a tile
            # whose floor is implausibly far above the global floor.
            if mesh_mode and v > base + _MESH_FLOOR_BAND:
                continue
            G[keys[k, 0] - i0, keys[k, 1] - j0] = v
        return G, i0, j0

    def _lookup(G, i0, j0, g, x, y):
        i = np.clip(np.floor(np.asarray(x, dtype=np.float64) / g
                             ).astype(np.int64) - i0, 0, G.shape[0] - 1)
        j = np.clip(np.floor(np.asarray(y, dtype=np.float64) / g
                             ).astype(np.int64) - j0, 0, G.shape[1] - 1)
        return G[i, j]

    if mesh_mode:
        # FINE tiles resolve a step (a COARSE tile straddling it takes
        # the lower floor and the raised section reads h=step); the
        # COARSE map fills fine tiles too sparse to support, so a
        # sparsely sampled slab still reads its OWN floor instead of
        # falling to the global base.
        Gc, i0c, j0c = _build(grid)
        Gf, i0f, j0f = _build(_MESH_FLOOR_GRID)
        Gc = np.where(np.isnan(Gc), base, Gc)

        def _fl(x, y):
            v = _lookup(Gf, i0f, j0f, _MESH_FLOOR_GRID, x, y)
            if np.ndim(v) == 0:
                if not np.isnan(v):
                    return float(v)
                return float(_lookup(Gc, i0c, j0c, grid, x, y))
            v = np.asarray(v, dtype=np.float64)
            cv = _lookup(Gc, i0c, j0c, grid, x, y)
            return np.where(np.isnan(v), cv, v)
        return _fl

    G, i0, j0 = _build(grid)
    G = np.where(np.isnan(G), base, G)

    def _fl(x, y):
        return _lookup(G, i0, j0, grid, x, y)
    return _fl


def _floor_at_local(fl, yaw: float):
    """Wrap a WORLD-frame floor callable so it accepts ROTATED-frame
    (row-frame) xy.

    The fit functions receive points rotated by -yaw into the row frame,
    while _floor_map is a function of world xy. Without the inverse
    rotation the lookup hits the WRONG location -- a flat floor hid the
    bug (every lookup returned ~0) while a stepped room looked up the
    wrong step and lifted the box onto it (bottom off the ground, top
    off too). Rotating local -> world first restores the correct step.
    """
    c, s = math.cos(yaw), math.sin(yaw)

    def _f(lx, ly):
        lx = np.asarray(lx, dtype=float)
        ly = np.asarray(ly, dtype=float)
        return fl(c * lx - s * ly, s * lx + c * ly)

    return _f


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

# groundview framing: expand the device-layout AABB by this margin so edge
# devices stay inside while far background does not inflate the camera.
_FRAME_MARGIN = 0.5

# recall tilt views (user direction 1): the extra L/R cameras deviate
# from vertical by this much -- small enough to keep the nadir's
# layout fidelity (rows stay near-axis-aligned, rects back-project
# close to their footprint), large enough to expose device FACES.
_RECALL_TILT_DEG = 20.0


def _grounding_frame(scene, yaw: float):
    """Rotated-frame AABB the grounding views are framed on.

    Frame the DEVICE LAYOUT (bootstrap device_cells / footprint), for a
    MESH too -- not the raw cloud bbox. Long-tail mesh noise (background
    captured OUTSIDE the room) inflates a whole-cloud AABB: the camera
    climbs to fit it and the room shrinks to a corner with lots of empty
    space and non-room structure in frame (user report). A small margin
    keeps edge devices in. Falls back to a ROBUST (percentile) cloud AABB
    when the bootstrap layout is unavailable, so a long tail is still
    trimmed.
    """
    lh = _layout_frame(scene, yaw)
    if lh is not None:
        lo = np.asarray(lh[0], dtype=np.float64) - _FRAME_MARGIN
        hi = np.asarray(lh[1], dtype=np.float64) + _FRAME_MARGIN
        return lo, hi
    all_rot = _rot_xy(np.asarray(scene.points, dtype=np.float64), -yaw)
    return (np.percentile(all_rot[:, :2], 1.0, axis=0),
            np.percentile(all_rot[:, :2], 99.0, axis=0))


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
                    frame=None, tilt_deg: float = 0.0, tilt_dir: int = 0):
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

    tilt_deg + tilt_dir (user direction 1, recall views): tilt the
    nadir camera SLIGHTLY toward +/- row-frame y (across the rows, the
    device faces). The nadir frame only shows roof-plates; a slight
    tilt exposes the FACES (doors, panels, AC grilles) and lets the
    VLM ground structures the flat view renders featureless. The tilt
    is small on purpose (default 20 deg): the view keeps most of the
    nadir's layout fidelity, and each rect still back-projects through
    ITS OWN camera (unproject_ground handles non-nadir rays).

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
        lh = _grounding_frame(scene, yaw)
        if lh is not None:
            lo, hi = lh
    if lo is not None:
        c = (lo + hi) / 2.0
        boxes_rot.append(OrientedBox(
            center=(float(c[0]), float(c[1]), 1.0),
            size=(float(hi[0] - lo[0]), float(hi[1] - lo[1]), 2.0),
            yaw=0.0))
    # crop the render to the framing region + a margin (row frame):
    # background beyond the layout -- long-tail mesh noise captured
    # outside the room -- must not render or inflate the camera (user
    # report: lots of empty space and non-room structure in the view).
    crop = None
    if lo is not None:
        pad = _TILE_OVERLAP
        crop = (np.asarray(lo, dtype=float) - pad,
                np.asarray(hi, dtype=float) + pad)
        m = ((pts_rot[:, 0] >= crop[0][0]) & (pts_rot[:, 0] <= crop[1][0]) &
             (pts_rot[:, 1] >= crop[0][1]) & (pts_rot[:, 1] <= crop[1][1]))
        band = band[m]
        pts_rot = pts_rot[m]
    cam_r = make_godview_cam(pts_rot, boxes_rot, nadir=True, W=W, H=H)
    if tilt_deg and tilt_dir:
        # recall tilt (user direction 1): shift the eye along +/-y in
        # the ROW frame (the cam below rotates it back to world), so
        # the tilt direction is across-the-rows regardless of yaw.
        # The shift is set so the sight line through the LAYOUT
        # CENTRE deviates from vertical by tilt_deg; the eye also
        # climbs ~25% of the tilt factor because the perspective
        # spreads the near edge outward and the frame must still
        # cover the whole layout. up: the old +y hint projected onto
        # the (now tilted) image plane -- the view stays row-aligned.
        t = math.tan(math.radians(float(tilt_deg)))
        tz = float(cam_r.target[2])
        ez2 = float(cam_r.eye[2]) * (1.0 + 0.25 * t)
        off = t * max(ez2 - tz, 1.0)
        eye2 = np.array([float(cam_r.eye[0]),
                         float(cam_r.eye[1]) + tilt_dir * off,
                         ez2])
        fwd = np.asarray(cam_r.target, dtype=float) - eye2
        up_h = np.array([0.0, 1.0, 0.0])
        up_h = up_h - float(up_h @ fwd / (fwd @ fwd)) * fwd
        up_h = up_h / np.linalg.norm(up_h)
        cam_r = Cam(eye=eye2, target=np.asarray(cam_r.target, dtype=float),
                    up=up_h, fovy_deg=cam_r.fovy_deg, W=W, H=H)
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
            from agentic_gts.tools.gs_io import read_scene_gaussian_ply
            gs = read_scene_gaussian_ply(scene)
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
            if crop is not None:
                # drop background gaussians outside the layout (row frame)
                gr = _rot_xy(gm, -yaw)
                keep &= ((gr[:, 0] >= crop[0][0]) & (gr[:, 0] <= crop[1][0])
                         & (gr[:, 1] >= crop[0][1]) & (gr[:, 1] <= crop[1][1]))
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


def _draw_cluster_candidates(img: np.ndarray, cam, cands, yaw: float,
                              W: int, H: int) -> np.ndarray:
    """Yellow numbered boxes for the missed cluster candidates, drawn
    on the nadir view: the row-frame rect corners rotate back to world
    at structure height and project through this view's own camera."""
    from PIL import Image, ImageDraw, ImageFont
    u8 = (np.clip(img, 0, 1) * 255).astype(np.uint8)[..., :3].copy()
    pil = Image.fromarray(u8)
    dr = ImageDraw.Draw(pil)
    try:
        font = ImageFont.truetype("arialbd.ttf", 30)
    except OSError:
        font = ImageFont.load_default()
    yellow = (255, 220, 0)
    for cid, (rect, _n) in enumerate(cands, start=1):
        x0, y0, x1, y1 = rect
        corners = np.array([[x0, y0, 0.0], [x1, y0, 0.0],
                             [x1, y1, 0.0], [x0, y1, 0.0]])
        cw = _rot_xy(corners, yaw)[:, :2]
        uv = cam.project_cv(np.column_stack([cw, np.full(4, 1.2)]))
        pts = [(int(round(p[0])), int(round(p[1]))) for p in uv]
        pts.append(pts[0])
        for a, b in zip(pts, pts[1:]):
            dr.line((a, b), fill=yellow, width=4)
        cpx = int(np.clip(np.mean([p[0] for p in pts]), 10, W - 60))
        cpy = int(np.clip(np.mean([p[1] for p in pts]), 30, H - 10))
        dr.text((cpx, cpy), str(cid), fill=yellow, font=font)
    return np.asarray(pil, dtype=np.float32) / 255.0


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
    """Solid outlines for the grounded result boxes, COLOR-CODED by
    provenance (result-only audit: the VLM answered on the clean base,
    the fit is shown apart). Red = nadir-grounded; ORANGE = tilt-view
    fit; CYAN = cluster recall net. When a result box looks wrong, the
    color says WHICH stage produced it -- a merged box that is orange
    is tilt-perspective inflation, cyan is the recall net's cluster
    spanning devices, red is the nadir fit itself (user debugging)."""
    from PIL import Image, ImageDraw
    u8 = (np.clip(img, 0, 1) * 255).astype(np.uint8)[..., :3].copy()
    pil = Image.fromarray(u8)
    dr = ImageDraw.Draw(pil)
    colors = {"nadir": (255, 60, 60), "tilt": (255, 170, 40),
              "cluster": (40, 200, 230)}
    for b in boxes:
        z = b.center[2] + b.size[2] / 2.0
        cs = b.corners_2d()
        uv = cam.project_cv(np.column_stack([cs, np.full(len(cs), z)]))
        pts = [(int(round(p[0])), int(round(p[1]))) for p in uv]
        pts.append(pts[0])
        col = colors.get(str(b.meta.get("view", "nadir")), (255, 60, 60))
        for a, c in zip(pts, pts[1:]):
            dr.line((a, c), fill=col, width=2)
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


def _region_axis_span(v: np.ndarray, cell: float = 0.05,
                     mesh_mode: bool = False):
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

    mesh_mode (user directive): NO denoising at all on a mesh input
    -- there is no haze, every anti-haze cut only ever bit REAL
    sparser sections (thin dividers, starved face bands, sparse tops
    -- user report: fitted boxes SMALLER than the devices). The span
    is simply the raw min/max.

    Safety floor: when the peeled span still covers < 65% of the
    P0.5-P99.5 extent, the structure is more heterogeneous than the
    bins can see (or the slice was too thin) -- return the percentile
    extent instead of letting the fit collapse: a slightly loose box
    is correctable by the local refine, a rejected region is lost
    recall. Returns None only when there is nothing to bin."""
    v = np.asarray(v, dtype=float)
    if len(v) < 30:
        return None
    if mesh_mode:
        return float(v.min()), float(v.max())
    p_lo, p_hi = (float(x) for x in np.percentile(v, [0.5, 99.5]))
    lo, hi = float(v.min()), float(v.max())
    nb = int(np.floor((hi - lo) / cell)) + 2
    edges = lo + cell * np.arange(nb + 1)
    hist, _ = np.histogram(v, bins=edges)
    first_peak = float(hist.max())
    if first_peak < 3.0:
        return p_lo, p_hi
    keep_thr = (0.02 if mesh_mode else 0.06) * first_peak
    connect = 0.10 if mesh_mode else 0.20
    remaining = hist.astype(float).copy()
    kept = np.zeros(len(hist), dtype=bool)
    while True:
        peak = float(remaining.max())
        if peak < max(keep_thr, 3.0):
            break
        thr = connect * peak
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
                    floor_z: float = 0.0, mesh_mode: bool = False,
                    seed_top: float | None = None, floor_at=None):
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
    device band: devices stand ON the ground, so the box BOTTOM is the
    local floor and the top is the anchored density-connected run's
    top.

    floor_at (the _floor_map callable): the local floor is the MEDIAN
    of the per-point local floor over the rect's OWN points -- a
    scalar sampled at the rect CENTRE lifted the box onto the wrong
    step when the centre sat a level above the device (user report:
    raised-floor boxes came out shifted). Consistent with the
    per-point render cut / fit-pool bands. Without it the scalar
    `floor_z` is used (direct unit calls). seed_top (scene-level
    bootstrap) is a HEIGHT above the DOMINANT floor, so the local top
    is floor_z + seed_top and the local HEIGHT is seed_top -- not
    seed_top - floor_z, which sank the top a step on a raised section.

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
    # LOCAL floor from the rect's OWN points (user report: a scalar
    # sampled at the rect CENTRE lifted the box onto the wrong step when
    # the centre sat a level above the device). Median of the per-point
    # local floor -- consistent with the per-point render cut / fit-pool
    # bands (h = z - floor(x, y)). Falls back to the scalar `floor_z`.
    if floor_at is not None:
        local_fl = np.asarray(floor_at(pts[:, 0], pts[:, 1]), dtype=float)
        floor_z = float(np.median(local_fl))
        h_local = pts[:, 2] - local_fl
    else:
        h_local = pts[:, 2] - floor_z
    # device band, floor-relative: exclude that section's floor texture
    dev = pts[h_local > 0.30]
    if len(dev) < max(30, min_pts // 2):
        _reject("floor patch, no structure above 0.30m", n_dev=len(dev))
        return None
    # anchored column top, P99.5 fallback: the rect's points form one
    # union column (mixed-height cabinets, all standing on the ground),
    # so the density-connected run's top is the row's true tallest --
    # a percentile lets floating overhead clutter inside the rect drag
    # it higher (same failure the hint-free bootstrap z_top had)
    # HEIGHT: with seed_top (the standing contract -- user directive)
    # the global fit provides XY ONLY; the height is the scene-level
    # z_top seed and the local refine (SAM mask back-projection)
    # re-measures it per box. Without seed_top (direct unit calls /
    # tests) the top comes from the band itself.
    if seed_top is not None:
        # seed_top is a HEIGHT above the DOMINANT floor (scene-level
        # bootstrap): the local top is floor_z + seed_top, so the local
        # HEIGHT is seed_top -- NOT seed_top - floor_z, which sank the
        # top a whole step on a raised section (user report: raised-
        # floor boxes came out shifted).
        z_top = float(seed_top)
        height = z_top
    else:
        if mesh_mode:
            # MESH (user directive: no denoising): the top is the raw MAX
            # of the device band -- the anchored walk exists to stop at
            # 3DGS haze tails and only ever bit real sparse tops on a
            # mesh.
            z_top = float(dev[:, 2].max())
        else:
            from agentic_gts.agent.mask_refine import _anchored_top
            _at = _anchored_top(dev[:, 2])
            z_top = float(_at) if _at is not None \
                else float(np.percentile(dev[:, 2], 99.5))
        height = z_top - floor_z          # absolute top - local floor
    if height < 0.50:
        _reject(f"too short for a device (height={height:.2f}m "
                f"over floor {floor_z:.2f})")
        return None
    # middle z-slice: [0.35, 0.75] x height above the LOCAL floor --
    # cuts every vertical face of a tall rack, stays above floor
    # texture, below trays/floaters. With seed_top the pool IS a low
    # band already (user directive: 0.30-1.00m, XY only) -- no slicing.
    if seed_top is not None:
        core = dev
    else:
        zc0 = floor_z + max(0.30, 0.35 * height)
        zc1 = max(zc0 + 0.10, floor_z + 0.75 * height)
        core = dev[(dev[:, 2] >= zc0) & (dev[:, 2] <= zc1)]
        if len(core) < 30:
            core = dev               # thin structure: whole band
    sx = _region_axis_span(core[:, 0], mesh_mode=mesh_mode)
    sy = _region_axis_span(core[:, 1], mesh_mode=mesh_mode)
    if sx is not None and sy is not None:
        (x_lo, x_hi), (y_lo, y_hi) = sx, sy
    elif mesh_mode:                  # sparse but clean: raw extent
        x_lo, y_lo = dev[:, :2].min(axis=0)
        x_hi, y_hi = dev[:, :2].max(axis=0)
    else:                            # too sparse to bin: percentile fit
        x_lo, y_lo = np.percentile(dev[:, :2], 0.5, axis=0)
        x_hi, y_hi = np.percentile(dev[:, :2], 99.5, axis=0)
    dx, dy = float(x_hi - x_lo), float(y_hi - y_lo)
    if dx < 0.30 or dy < 0.20:
        _reject(f"sliver (span {dx:.2f} x {dy:.2f}m; "
                f"core={len(core)} pts, seed_top="
                f"{seed_top if seed_top is not None else 'fit'})")
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

# a box whose SHORTER horizontal axis exceeds this is a "fat blob" (user
# report: a huge box covering aisles and junk), not a device row. No
# device row is that deep; a joined row is long but thin and passes.
# Sits above the back-to-back double-row depth (~2.2m) so those survive.
_MAX_DEVICE_SPAN = 2.5


def _oversized_blob(b, max_span: float = _MAX_DEVICE_SPAN) -> bool:
    """True when a fitted box is far too 'fat' to be a device row: the
    SHORTER horizontal axis exceeds any plausible device depth. Long
    thin rows are unaffected (the long axis is unbounded); a big blob
    spanning aisles/junk in BOTH axes is caught. Cheap geometry, runs
    BEFORE the expensive local refine."""
    return min(float(b.size[0]), float(b.size[1])) > max_span



def _cross_gap_split(v: np.ndarray, peak_frac: float = 0.25,
                     min_side: float = _MIN_DEVICE_DEPTH) -> float | None:
    """Split coordinate of the most BALANCED interior weak run in a 1-D
    cross-axis density profile, or None.

    A gap qualifies when it is >= _SPLIT_MIN_GAP wide, interior (not
    touching the profile edges) and BOTH sides keep a >= min_side
    strong span: with the default _MIN_DEVICE_DEPTH a run with a
    single face-sheet on one side is a rack INTERIOR (hollow row), not
    an aisle -- splitting there shreds one row into its two faces.
    Among qualifying gaps the most balanced split wins (minimises the
    wider side's strong span); the caller recurses on any side still
    deeper than _MAX_DEVICE_DEPTH, which handles 3+ rows in one rect.

    min_side: the CLUSTER net lowers it (0.15) -- a wall-adjacent
    device's blob holds a THIN wall sheet on one side (0.2m < the 0.40
    device rule), and the strict default would reject the wall/device
    gap as if it were a hollow-row interior. With the relaxed side the
    gap qualifies, and each side's own refit does the real filtering:
    the wall strip dies in _fit_region_box's sliver guard, the device
    keeps its clean box."""
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
            if l_span >= min_side and r_span >= min_side:
                wider = max(l_span, r_span)
                if best is None or wider < best[0]:
                    best = (wider, g_lo, g_hi)
        i = j
    if best is None:
        return None
    return 0.5 * (best[1] + best[2])


def _fit_region_boxes(points: np.ndarray, rect, min_pts: int = 60,
                      floor_z: float = 0.0,
                      max_depth: float = _MAX_DEVICE_DEPTH,
                      min_side: float = _MIN_DEVICE_DEPTH,
                      mesh_mode: bool = False,
                      seed_top: float | None = None,
                      floor_at=None) -> list:
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
    survives the original whole box is kept (recall first).

    max_depth: the CLUSTER net lowers it to 1.35 -- a device standing
    against a wall merges with it into one blob of depth 0.2 (wall) +
    gap + 1.1 (device) ~= 1.6m, under the 1.8 default yet NOT a single
    device; the split's min-side rule (_MIN_DEVICE_DEPTH 0.40) then
    discards the wall side and keeps the clean device box."""
    bb = _fit_region_box(points, rect, min_pts, floor_z,
                         mesh_mode=mesh_mode, seed_top=seed_top,
                         floor_at=floor_at)
    if bb is None:
        return []

    def _dev_of(rect_):
        x0, y0, x1, y1 = rect_
        m = ((points[:, 0] >= x0) & (points[:, 0] <= x1) &
             (points[:, 1] >= y0) & (points[:, 1] <= y1))
        d = points[m]
        if floor_at is not None:
            d = d[(d[:, 2] - np.asarray(floor_at(d[:, 0], d[:, 1]))) > 0.30]
        else:
            d = d[d[:, 2] > floor_z + 0.30]
        return d
    axis = 1 if abs(float(bb.yaw)) < 1e-6 else 0   # cross axis of the fit
    if bb.size[1] <= max_depth:
        # LOW-BAND contract (seed_top set): a rect over a row AND a
        # separate LOW clump (an AC bank under the old middle slice's
        # 0.35 x height line) now fits as ONE long box -- the 0.30-1.00
        # band carries the clump too. Joined cabinets TOUCH along the
        # row, so an interior >= 0.3m empty run with strong spans on
        # both sides is two structures: split there (stageC would also
        # split along-row, but the grounding output should map to
        # structures, and the recall net's coverage test reads the
        # fitted footprints).
        if seed_top is None:
            return [bb]
        rax = 1 - axis                # the ROW (long) axis
        x0, y0, x1, y1 = rect
        dev = _dev_of(rect)
        s = _cross_gap_split(dev[:, rax], min_side=0.30)
        if s is None:
            return [bb]
        print(f"[ground] long fit (span {bb.size[0]:.2f}m) -> split at "
              f"{'y' if rax else 'x'}={s:.2f} (row + separate clump)")
        subs = ((x0, y0, x1, s), (x0, s, x1, y1)) if rax == 1 \
            else ((x0, y0, s, y1), (s, y0, x1, y1))
        out = []
        for sub in subs:
            out.extend(_fit_region_boxes(points, sub, min_pts, floor_z,
                                         max_depth=max_depth,
                                         min_side=min_side,
                                         mesh_mode=mesh_mode,
                                         seed_top=seed_top,
                                         floor_at=floor_at))
        return out or [bb]
    x0, y0, x1, y1 = rect
    dev = _dev_of(rect)
    s = _cross_gap_split(dev[:, axis], min_side=min_side)
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
        out.extend(_fit_region_boxes(points, sub, min_pts, floor_z,
                                     max_depth=max_depth,
                                     min_side=min_side,
                                     mesh_mode=mesh_mode,
                                     seed_top=seed_top,
                                     floor_at=floor_at))
    return out or [bb]


def _cluster_candidates(points: np.ndarray, cell: float = 0.30,
                        min_cell_pts: int = 6, min_cluster_pts: int = 60,
                        margin: float = 0.15) -> list:
    """Geometry-first recall net: connected DENSITY clusters over the
    fit pool (ROW frame -- same coordinates the rects fit in).

    User insight: after the ground cut and the top cut the pool holds
    nothing but walls, devices and junk -- on the groundview they all
    read as PIXEL CLUMPS. Free-form image detection (the VLM) misses
    structures on a clean nadir view; connected components over the
    density grid cannot -- a device is a dense blob of cells however
    axis-aligned and featureless the view. The VLM's job shrinks from
    DETECTION (find everything) to CLASSIFICATION (judge pre-marked
    candidates), which it does far more reliably.

    Occupancy: a 30cm cell counts when >= min_cell_pts points fall in
    it (haze cells stay under); components are 8-connected (diagonal
    row continuities) over a ONE-CELL DILATED grid -- a closed rack
    row is a HOLLOW shell (two face bands, empty interior cells), and
    without the dilation the two faces land two cells apart and split
    into separate clusters. The dilation is 0.3m: real aisles (>=0.6m)
    still separate. Clusters under min_cluster_pts are noise.
    Returns [(rect, n_pts)] in row-frame coordinates.
    """
    from collections import deque
    P = np.asarray(points, dtype=np.float64)
    if len(P) < 200:
        return []
    ix = np.floor(P[:, 0] / cell).astype(np.int64)
    iy = np.floor(P[:, 1] / cell).astype(np.int64)
    keys, inv = np.unique(np.column_stack([ix, iy]), axis=0,
                          return_inverse=True)
    counts = np.bincount(inv, minlength=len(keys))
    occ = counts >= min_cell_pts
    if not occ.any():
        return []
    i0, j0 = int(keys[:, 0].min()), int(keys[:, 1].min())
    G = np.zeros((int(keys[:, 0].max()) - i0 + 1,
                  int(keys[:, 1].max()) - j0 + 1), dtype=bool)
    G[keys[occ, 0] - i0, keys[occ, 1] - j0] = True
    # dilate one cell (8-neighbour): stitch hollow-shell interiors
    Pd = np.pad(G, 1)
    Gd = np.zeros_like(G)
    for dx in (0, 1, 2):
        for dy in (0, 1, 2):
            Gd |= Pd[dx:dx + G.shape[0], dy:dy + G.shape[1]]
    lab = np.zeros(G.shape, dtype=np.int32)
    cur = 0
    for a in range(G.shape[0]):
        for b in range(G.shape[1]):
            if not Gd[a, b] or lab[a, b]:
                continue
            cur += 1
            lab[a, b] = cur
            q = deque([(a, b)])
            while q:
                x, y = q.popleft()
                for dx in (-1, 0, 1):
                    for dy in (-1, 0, 1):
                        nx, ny = x + dx, y + dy
                        if (0 <= nx < G.shape[0] and 0 <= ny < G.shape[1]
                                and Gd[nx, ny] and not lab[nx, ny]):
                            lab[nx, ny] = cur
                            q.append((nx, ny))
    kl = lab[keys[:, 0] - i0, keys[:, 1] - j0]   # 0 = haze-only cell
    plab = kl[inv]
    out = []
    for c in range(1, cur + 1):
        m = plab == c
        if int(m.sum()) < min_cluster_pts:
            continue
        q = P[m][:, :2]
        lo, hi = q.min(axis=0) - margin, q.max(axis=0) + margin
        out.append(((float(lo[0]), float(lo[1]),
                     float(hi[0]), float(hi[1])), int(m.sum())))
    return out


def _rect_covered(a, b, iou_thr: float = 0.10,
                  contain_thr: float = 0.60,
                  rev_contain_thr: float = 0.75) -> bool:
    """Is row-frame rect `a` already accounted for by VLM rect `b`?
    Loose on purpose: a cluster overlapping any real part of a VLM
    rect is NOT a missed device -- only fully-uncovered clumps go to
    adjudication (recall of the NET, precision of the VLM).

    The REVERSE containment matters for devices standing AGAINST a
    wall: the wall merges the device's cluster into a wall+device
    BLOB whose area dwarfs the VLM rect (that outlined only the
    device), so the forward containment fails -- yet re-proposing
    that blob fits a wall-inflated box that out-supports (wall
    points!) and EATS the correct VLM box in the dedup (user
    question: wall-adjacent devices). A VLM rect sitting ~WHOLLY
    inside the cluster means the VLM already handled the structure
    that rect covers: covered."""
    ax0, ay0, ax1, ay1 = a
    bx0, by0, bx1, by1 = b
    ix = max(0.0, min(ax1, bx1) - max(ax0, bx0))
    iy = max(0.0, min(ay1, by1) - max(ay0, by0))
    inter = ix * iy
    if inter <= 0:
        return False
    area_a = max((ax1 - ax0) * (ay1 - ay0), 1e-9)
    area_b = max((bx1 - bx0) * (by1 - by0), 1e-9)
    return inter / area_a >= contain_thr or \
        inter / area_b >= rev_contain_thr or \
        inter / max(area_a + area_b - inter, 1e-9) >= iou_thr


def _rect_inside(outer, inner, thr: float = 0.75) -> bool:
    """>=thr of INNER's area lies inside OUTER (reverse containment)."""
    ax0, ay0, ax1, ay1 = outer
    bx0, by0, bx1, by1 = inner
    ix = max(0.0, min(ax1, bx1) - max(ax0, bx0))
    iy = max(0.0, min(ay1, by1) - max(ay0, by0))
    inter = ix * iy
    if inter <= 0:
        return False
    area_b = max((bx1 - bx0) * (by1 - by0), 1e-9)
    return inter / area_b >= thr


def _cluster_pts_covered(rect, pts: np.ndarray, fitted: list,
                         thresh: float = 0.65, pad: float = 0.10,
                         min_judge: int = 30) -> bool:
    """Is this cluster's point mass already inside the FITTED boxes?

    The recall net's real coverage test (replaces rect-overlap
    heuristics): a cluster counts as covered when >= `thresh` of its
    OWN points fall inside the union of the fitted box footprints
    (row-frame AABBs, padded). Rect-AREA overlap was too loose -- a
    VLM rect that merely grazes a neighbour, or one whose fit snapped
    to a DIFFERENT peak run of the same pool (the peak-peeling span
    estimator drops the weaker structure), leaves the clump's points
    entirely undetected while every rect-overlap rule calls it
    'covered' (user report: obvious rectangular clumps left unboxed).
    Point mass cannot lie about what the pipeline actually detected."""
    x0, y0, x1, y1 = rect
    m = ((pts[:, 0] >= x0) & (pts[:, 0] <= x1) &
         (pts[:, 1] >= y0) & (pts[:, 1] <= y1))
    q = pts[m][:, :2]
    if len(q) < min_judge:
        return True                # noise-level clump: nothing to add
    inside = np.zeros(len(q), dtype=bool)
    for (ax0, ay0, ax1, ay1) in fitted:
        inside |= ((q[:, 0] >= ax0 - pad) & (q[:, 0] <= ax1 + pad) &
                   (q[:, 1] >= ay0 - pad) & (q[:, 1] <= ay1 + pad))
    return float(inside.mean()) >= thresh


def _merge_adjacent_boxes(boxes: list, pts_fit: np.ndarray, yaw: float,
                          bridge_tol: float = 0.50,
                          min_gap_pts: int = 15,
                          density_ratio: float = 0.30,
                          floor_at=None,
                          probe_pool: np.ndarray | None = None,
                          seed_top: float | None = None) -> list:
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

    probe_pool: the pool the DENSITY EVIDENCE is measured on. With a
    mesh the default pts_fit (0.30 .. z_top + 0.10) carries the CABLE
    TRAYS -- dense, gapless, physically bridging adjacent device tops
    -- and a tray strip in the junction passes the density ratio like
    a real seam, merging devices that have NO overlap on the rendered
    groundview (user report: the render cut hides the trays, the
    merge probe does not see them). The caller passes the render-cut
    pool (trays removed, device bodies kept) so only the devices'
    own band can testify. The final union refit still uses pts_fit
    (rack tops belong in the box height)."""
    pp = pts_fit if probe_pool is None else probe_pool
    n = len(boxes)
    if n < 2:
        return boxes
    # row-frame AABB + orientation bucket (long side on x or on y)
    rects, buckets, srcs = [], [], []
    for b in boxes:
        cs = _rot_xy(np.column_stack([b.corners_2d(),
                                      np.zeros(4)]), -yaw)
        rects.append((float(cs[:, 0].min()), float(cs[:, 1].min()),
                      float(cs[:, 0].max()), float(cs[:, 1].max())))
        buckets.append(int(round((b.yaw - yaw) / (math.pi / 2.0))) % 2)
        srcs.append(str(b.meta.get("view", "nadir")))
    # per-box device-band density from the SAME pool the probe uses
    # (surfaces are dense, an inflated fit barely dilutes it)
    dens = []
    for r in rects:
        m = ((pp[:, 0] >= r[0]) & (pp[:, 0] <= r[2]) &
             (pp[:, 1] >= r[1]) & (pp[:, 1] <= r[3]))
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
            if srcs[i] != srcs[j]:
                continue          # a tilt box never chains onto a
                # nadir one: the merge exists to heal ONE view's
                # over-split of one structure; cross-view pairs are
                # by construction different devices (user report:
                # merged red result boxes)
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
                m = ((pp[:, axis] >= s_lo) &
                     (pp[:, axis] <= s_hi) &
                     (pp[:, o] >= p_lo) & (pp[:, o] <= p_hi))
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
        # SINGULAR fit on purpose: these members were bridged by a
        # dense device band -- one verified continuous structure -- so
        # the union refit must NOT re-split it across that bridge (the
        # deep-split belongs to the per-RECT path, where the rect
        # itself is the only evidence)
        bb = _fit_region_box(pts_fit, u, seed_top=seed_top,
                             floor_at=floor_at)
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


# a VLM rect covering more than this fraction of the view is a "hedge"
# box (whole room / large empty floor / background), not a device row:
# an 8B grounding model falls back to one big box when the frame is
# ambiguous (user report). Rejected BEFORE the fit/dedup so it cannot
# out-support and eat the correct boxes. A legit row band covers less.
_MAX_RECT_FRAC = 0.8


def _huge_rect(r, W: int, H: int, max_frac: float = _MAX_RECT_FRAC) -> bool:
    """True when a pixel rect covers more than `max_frac` of the view."""
    w = max(0.0, float(r[2]) - float(r[0]))
    h = max(0.0, float(r[3]) - float(r[1]))
    return (w * h) > max_frac * float(W) * float(H)


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
        tiles = _tile_frames(_grounding_frame(scene, yaw))
    except Exception:
        tiles = None
    views = []                       # (img, cam, W, H, fname, rects)
    view_specs = [("groundview.png", None)] if tiles is None else \
        [(f"groundview_t{i}.png", fr) for i, fr in enumerate(tiles)]
    # recall tilts run ONLY for a single view: a tiled layout already
    # renders edge devices obliquely (perspective nadir + tile overlap),
    # so the per-tile tilts are redundant -- see the note in the loop.
    want_tilts = tiles is None
    if tiles is not None:
        print(f"[ground] layout exceeds a single nadir view "
              f"(>{_MAX_SINGLE_SPAN:.0f}m span) -> {len(tiles)} tiled "
              f"views ({len(tiles)} VLM call(s), recall tilts "
              f"{'on' if want_tilts else 'skipped'})")
    if not want_tilts:
        print("[ground] recall tilts skipped: tiling already renders "
              "edge devices obliquely (perspective nadir + overlap)")
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
        # ---- recall tilt views (user direction 1) ----
        # Two extra cameras slightly tilted toward +/- across-row, on
        # THIS view's frame: the flat nadir frame renders every device
        # as a roof-plate -- featureless boxes are where image
        # grounding misses -- while a slight tilt exposes the faces.
        # Recall only ADDS views: the dedup below folds the duplicate
        # boxes, and each tilted rect back-projects through its own
        # camera (unproject_ground), so no geometry is shared with the
        # nadir frame by mistake.
        # SKIPPED for tiled layouts (user insight): tiling already gives
        # the oblique edge views these exist to add.
        if not want_tilts:
            continue
        stem = fname[:-4] if fname.endswith(".png") else fname
        for tag, d in (("L", -1), ("R", +1)):
            try:
                img_t, cam_t, W_t, H_t = _render_topdown(
                    scene, yaw, frame=fr, tilt_deg=_RECALL_TILT_DEG,
                    tilt_dir=d)
                png_t = png_bytes(img_t)
            except Exception as e:
                print(f"[ground] tilt-{tag} render failed "
                      f"({type(e).__name__}: {e})")
                continue
            pngp_t = None
            if out_dir:
                pngp_t = os.path.join(out_dir, f"{stem}_{tag}.png")
                try:
                    with open(pngp_t, "wb") as f:
                        f.write(png_t)
                except Exception as e:
                    print(f"[ground] png save failed "
                          f"({type(e).__name__})")
                    pngp_t = None
            rects_t = judge.ground_regions(png_t, W_t, H_t,
                                           png_path=pngp_t)
            print(f"[ground] view {stem}_{tag}: {len(rects_t)} regions")
            views.append((img_t, cam_t, W_t, H_t, f"{stem}_{tag}.png",
                          rects_t))
    if not views:
        return False                 # every render failed (logged above)
    if not any(v[5] for v in views):
        print("[ground] VLM returned no usable regions")
        if out_dir:
            _save_grounded_fail_png(views[0][0], out_dir,
                                    "VLM returned no usable regions")
        return False
    # FIT points: the LOW device band only -- 0.30 to 1.00m above the
    # local floor (user directive: the global fit provides XY ONLY;
    # heights are the scene z_top seed and the local refine
    # re-measures them). The old z_top+0.10 cut existed to keep rack
    # tops in the HEIGHT fit -- no longer needed; and ceiling / tray
    # points span the WHOLE room in XY, so any cut that lets them in
    # produces boxes hugging the loose rect edges (user report: red
    # boxes too large while the colored rects were right). Height cuts
    # are relative to the local floor (stepped rooms: a raised
    # section's slab is that section's floor).
    P = np.asarray(scene.points, dtype=np.float64)
    is_mesh = bool(scene.meta.get("geometry_is_mesh"))
    fl = _floor_map(P, mesh_mode=is_mesh)
    # floor_at for the FIT functions: they receive points in the ROTATED
    # (row) frame, while _floor_map is a function of WORLD xy -- wrap it
    # so the lookup rotates local -> world first (see _floor_at_local).
    # The render cut / fit pool below keep the world `fl`.
    fl_local = _floor_at_local(fl, yaw)
    h_fit = P[:, 2] - fl(P[:, 0], P[:, 1])
    fit_top = float(scene.meta.get("z_top", 2.5) or 2.5)
    pts_fit = _rot_xy(P[(h_fit > 0.30) & (h_fit <= 1.00)], -yaw)
    if len(pts_fit) < 100:
        pts_fit = _rot_xy(P[h_fit > 0.30], -yaw)

    # RENDER-CUT pool (trays removed): every "what the groundview
    # shows" judgement -- the cluster recall net, the adjacency-merge
    # probe -- runs on THIS pool. The render cut (GS 0.70, mesh 0.50
    # of the top) clears the CABLE TRAYS: dense, gapless, spanning
    # every aisle -- a tray strip in a junction would pass the merge
    # probe's density ratio like a real seam (user report: the render
    # cut hides the trays, the probe does not see them). Devices and
    # walls remain, and an inter-device gap is EMPTY cells the
    # evidence cannot cite.
    pts_clu = pts_fit
    try:
        cc = _render_cut(
            fit_top, mesh_mode=bool(scene.meta.get("geometry_is_mesh")))
        pts_clu = _rot_xy(
            P[(h_fit > 0.30) & (h_fit <= (cc if np.isfinite(cc)
                                          else fit_top + 0.10))], -yaw)
    except Exception as e:
        print(f"[ground] render-cut pool failed ({type(e).__name__}: {e})"
              f" -> using the fit pool")

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
    row_rects = []                   # row-frame AABBs, for the recall net
    fitted_rects = []                # row-frame footprints of FITTED boxes

    def _fit_ground_rect(rect_r, source: str = "nadir") -> int:
        """Fit one row-frame rect; append its boxes. Returns the number
        of boxes appended. `source` tags the box's view ("nadir" /
        "tilt"): the dedup and the adjacency merge both rank and pair
        on it -- a tilt box may never eat or chain onto a nadir one
        (user report: red result boxes merging devices the colored
        rects showed apart)."""
        # the rect's own LOCAL floor (stepped rooms) comes from ITS OWN
        # points via floor_at (per-point median) -- not a scalar at the
        # rect centre, which lifted the box onto the wrong step
        bbs = _fit_region_boxes(pts_fit, rect_r, floor_at=fl_local,
                                mesh_mode=is_mesh, seed_top=fit_top)
        # _fit_region_boxes (plural): a deep fit -- the VLM drew ONE
        # rect around two opposing rows -- splits at the aisle here,
        # before the box enters the pipeline (stageC can only split
        # along the row axis)
        n_appended = 0
        for bb in bbs:
            c = _rot_xy(np.array([[bb.center[0], bb.center[1], 0.0]]),
                        yaw)[0]
            # bb.yaw is 0 (row along the rotated-x axis) or pi/2 (row
            # along rotated-y): both rotate into the world by ADDING
            # the frame yaw
            box = OrientedBox(center=(float(c[0]), float(c[1]),
                                      bb.center[2]),
                              size=bb.size, yaw=yaw + float(bb.yaw),
                              device_type=DeviceType.RACK,
                              meta={"grounded": True, "view": source,
                                    "n_pts": bb.meta.get("n_pts", 0)})
            boxes.append(box)
            # the recall net judges coverage on what was ACTUALLY
            # detected: the fitted footprint (row-frame AABB; a pi/2
            # box swaps its extents), not the raw VLM rect
            sx, sy = float(bb.size[0]), float(bb.size[1])
            if abs(float(bb.yaw)) > 1e-6:
                sx, sy = sy, sx
            fitted_rects.append((float(bb.center[0]) - 0.5 * sx,
                                 float(bb.center[1]) - 0.5 * sy,
                                 float(bb.center[0]) + 0.5 * sx,
                                 float(bb.center[1]) + 0.5 * sy))
            n_appended += 1
        return n_appended

    def _is_tilt_view(fname_v: str) -> bool:
        return fname_v.endswith("_L.png") or fname_v.endswith("_R.png")

    # PASS 1 -- the NADIR views are the layout AUTHORITIES: vertical
    # rays carry no perspective dilation, their rects ARE the foot-
    # prints, and every later coverage question is answered against
    # what they fitted.
    for v in views:
        if _is_tilt_view(v[4]):
            continue
        cam_v, W_v, H_v, rects_v = v[1], v[2], v[3], v[5]
        for r in rects_v:
            if _huge_rect(r, W_v, H_v):
                print(f"[ground] huge VLM rect dropped "
                      f"({(r[2] - r[0]) * (r[3] - r[1]) / (W_v * H_v):.0%} "
                      f"of {v[4]}): hedge box, not a device row")
                continue
            rect_r = _frame_rect(cam_v, r, 1.0)
            row_rects.append(rect_r)
            _fit_ground_rect(rect_r)
    # PASS 2 -- the tilt views are RECALL-ONLY (user report: red
    # result boxes merging devices the colored rects showed apart).
    # A tilted camera's back-projection is perspective-INFLATED: the
    # image rect is the device's visible hull, whose rays cut any
    # single z-plane in a footprint WIDER than the device -- a tilt
    # rect SPANNING an already-grounded device plus a missed one
    # passes the old point-mass coverage gate (~50% covered) and
    # fits a box across BOTH. Guards, in order:
    #   * the back-projection is TIGHTENED by intersecting the slices
    #     at two device-band heights (the oblique-view lesson: the
    #     bottom slice inflates away from the camera, the top slice
    #     toward it, the intersection trims both);
    #   * a tilt rect touching already-grounded AREA at all (>= 25%
    #     of its area inside the nadir fits / raw rects union) is
    #     skipped WHOLE -- the cluster recall net downstream recovers
    #     any genuinely missed device without spanning risk.
    def _rect_covered_frac(rect, others) -> float:
        """Fraction of rect's area inside the UNION of the AABBs."""
        x0, y0, x1, y1 = rect
        w, h = x1 - x0, y1 - y0
        if w <= 0 or h <= 0 or not others:
            return 0.0
        step = max(w, h) / 200.0
        xs = x0 + step * (np.arange(int(w / step)) + 0.5)
        ys = y0 + step * (np.arange(int(h / step)) + 0.5)
        gx, gy = np.meshgrid(xs, ys)
        px, py = gx.ravel(), gy.ravel()
        cov = np.zeros(len(px), dtype=bool)
        for o in others:
            cov |= ((px >= o[0]) & (px <= o[2]) &
                    (py >= o[1]) & (py <= o[3]))
        return float(cov.mean())

    tilt_added = tilt_skipped = 0
    for v in views:
        if not _is_tilt_view(v[4]):
            continue
        cam_v, W_v, H_v, rects_v = v[1], v[2], v[3], v[5]
        for r in rects_v:
            if _huge_rect(r, W_v, H_v):
                continue
            lo_r = _frame_rect(cam_v, r, 0.30)
            hi_r = _frame_rect(cam_v, r, 1.00)
            rect_r = (max(lo_r[0], hi_r[0]), max(lo_r[1], hi_r[1]),
                      min(lo_r[2], hi_r[2]), min(lo_r[3], hi_r[3]))
            if rect_r[0] >= rect_r[2] or rect_r[1] >= rect_r[3]:
                rect_r = _frame_rect(cam_v, r, 1.0)   # disjoint slices
            if _rect_covered_frac(rect_r, fitted_rects + row_rects) >= 0.25:
                tilt_skipped += 1
                continue
            tilt_added += _fit_ground_rect(rect_r, source="tilt")
    if tilt_added or tilt_skipped:
        print(f"[ground] tilt recall views: {tilt_added} box(es) added, "
              f"{tilt_skipped} rect(s) skipped (touch nadir-grounded area)")
    if not boxes:
        print("[ground] no region survived the point-support guards")
        if out_dir:
            _save_grounded_fail_png(views[0][0], out_dir,
                                    "no region survived point-support guards")
        return False
    n_rects_total = sum(len(v[5]) for v in views)
    print(f"[ground] {n_rects_total} VLM regions "
          f"({len(views)} view(s)) -> {len(boxes)} fitted boxes")
    # --- cluster recall net (user insight, RECALL-FIRST) ---
    # After the ground and top cuts the fit pool holds nothing but
    # walls, devices and junk -- every structure reads as a DENSITY
    # CLUMP in row-frame space, and free-form image detection MISSES
    # some on a clean nadir view (user report: obvious rectangular
    # clumps left unboxed -> missed devices). The net: connected
    # components over the density grid PROPOSE, and (user directive)
    # every uncovered cluster becomes a box UNCONDITIONALLY -- a
    # wrong proposal is CHEAP, a missed device is LOST: stageC's
    # per-box local views type-confirm each box and non-devices die
    # there (type_suspect -> LOW -> final filter). The earlier VLM
    # classification gate re-imported the very instability the net
    # exists to absorb (user report: unstable grounding, clusters
    # rejected or the call failing -> nothing added). Clusters already
    # covered by a VLM rect are still skipped: the net must only ADD
    # recall, never question the rects.
    # cluster recall net: the recall safety net over the render-cut
    # pool (trays removed -- the fit pool's trays bridge every aisle
    # and would seam the whole room into one cluster).
    try:
        cands = _cluster_candidates(pts_clu)
    except Exception as e:
        print(f"[ground] clustering failed ({type(e).__name__}: {e})")
        cands = []
    # Coverage judged on what was ACTUALLY DETECTED (user report: the
    # net never fired despite obvious unboxed clumps). The old rect-
    # overlap rules called a cluster 'covered' on a 10% AREA overlap
    # with ANY VLM rect -- including rects whose fit snapped to a
    # different structure and rects rejected by the point guards --
    # so real devices adjacent to detected ones stayed missed. Now:
    # covered = >=65% of the cluster's OWN points inside the fitted
    # footprints, OR a VLM rect ~wholly inside the cluster (the
    # wall-adjacent protection: re-proposing a wall+device blob fits
    # a wall-inflated box that out-supports and EATS the correct VLM
    # box in the dedup).
    missed = [(r, n) for (r, n) in cands
              if not _cluster_pts_covered(r, pts_clu, fitted_rects)
              and not any(_rect_inside(r, vr) for vr in row_rects)]
    if missed:
        print(f"[ground] cluster recall net: {len(missed)} uncovered "
              f"candidate(s) of {len(cands)} -> proposing ALL "
              f"(stageC local views cull non-devices)")
        n_added = 0
        try:
            W0, H0 = views[0][2], views[0][3]
            png_c = png_bytes(_draw_cluster_candidates(
                views[0][0], views[0][1], missed, yaw, W0, H0))
            if out_dir:
                try:
                    with open(os.path.join(out_dir,
                                           "cluster_check.png"), "wb") as f:
                        f.write(png_c)
                except Exception as e:
                    print(f"[ground] cluster png save failed "
                          f"({type(e).__name__})")
            for cid, (rect, _npts) in enumerate(missed, start=1):
                for bb in _fit_region_boxes(
                        pts_fit, rect, floor_at=fl_local,
                        max_depth=1.35, min_side=0.15,
                        mesh_mode=is_mesh, seed_top=fit_top):
                    # wall-thin fits never enter the pipeline: a wall
                    # blob OUT-SUPPORTS real device boxes on sheer
                    # point count and would eat them in the dedup (no
                    # device category is thinner than 0.35m; walls
                    # are 0.1-0.3m). Everything thicker is proposed
                    # and left for stageC to confirm or cull.
                    if bb.size[1] < 0.35:
                        continue
                    c = _rot_xy(np.array(
                        [[bb.center[0], bb.center[1], 0.0]]), yaw)[0]
                    boxes.append(OrientedBox(
                        center=(float(c[0]), float(c[1]), bb.center[2]),
                        size=bb.size, yaw=yaw + float(bb.yaw),
                        device_type=DeviceType.RACK,
                        meta={"grounded": True, "view": "cluster",
                              "cluster": cid,
                              "n_pts": bb.meta.get("n_pts", 0)}))
                    n_added += 1
            print(f"[ground] cluster recall net: {n_added} box(es) added")
        except Exception as e:
            print(f"[ground] cluster recall net failed "
                  f"({type(e).__name__}: {e}) -> skipped")
    # DEDUPLICATE, TWO TIERS (user report: colored rects fine, red
    # boxes merged). The old single tier sorted everything by n_pts
    # and let the biggest fit win -- a perspective-inflated tilt box
    # spanning two devices carries the most points, enters FIRST and
    # eats both correct nadir boxes as 'contained duplicates'. Now:
    #   * tier 1, NADIR (and cluster-net) boxes only: drop a box that
    #     overlaps a better-supported kept fit (IoU >= 0.5) or is
    #     >= 85% contained in one (nested rects: a small box inside a
    #     big row box has IoU = area ratio but containment ~1.0);
    #   * tier 2, TILT boxes: they may only fill EMPTY space -- any
    #     overlap with a kept box (IoU >= 0.2 or containment >= 0.3)
    #     drops the tilt box, however many points it carries. A tilt
    #     box can never replace, out-support or span across a nadir
    #     grounding.
    nadir_boxes = [b for b in boxes if b.meta.get("view") != "tilt"]
    tilt_boxes = [b for b in boxes if b.meta.get("view") == "tilt"]
    dedup = []
    for b in sorted(nadir_boxes, key=lambda x: -int(x.meta.get("n_pts", 0))):
        if any(b.iou_2d(d) >= 0.5 or b.containment_2d(d) >= 0.85
               for d in dedup):
            continue
        dedup.append(b)
    for b in sorted(tilt_boxes, key=lambda x: -int(x.meta.get("n_pts", 0))):
        if any(b.iou_2d(d) >= 0.2 or b.containment_2d(d) >= 0.3
               for d in dedup):
            continue
        dedup.append(b)
    if len(dedup) < len(boxes):
        print(f"[ground] dropped {len(boxes) - len(dedup)} duplicate/contained/"
              f"overlapping box(es) (nadir tiers authoritative, tilt "
              f"fill-only)")
    boxes = dedup
    # MERGE tightly-adjacent over-split pieces (user request): the VLM
    # sometimes outlines one physical structure as several tight rects;
    # each fits its own box and the seam never heals (stageC only
    # splits, never merges). Touching / point-bridged boxes merge into
    # a point-support-refitted union; the true splitting is the local
    # refine's job. The density EVIDENCE is measured on the render-cut
    # pool -- with a mesh the fit pool carries the cable trays, whose
    # gapless strips bridge adjacent device tops and pass the density
    # ratio like a real seam, merging devices that look fully separate
    # on the groundview (user report).
    boxes = _merge_adjacent_boxes(boxes, pts_fit, yaw, floor_at=fl_local,
                                  probe_pool=pts_clu, seed_top=fit_top)
    # drop "fat blob" boxes (user report: a huge box covering aisles and
    # junk): cheap geometric guard BEFORE the expensive local refine --
    # no device row is deeper than _MAX_DEVICE_SPAN on its SHORTER axis.
    # Long joined rows are long but thin, so they pass untouched. Each
    # drop is printed (visible, not silent).
    blobs = [b for b in boxes if _oversized_blob(b)]
    if blobs:
        for b in blobs:
            print(f"[ground] oversized blob dropped: short axis "
                  f"{min(b.size[0], b.size[1]):.2f}m > {_MAX_DEVICE_SPAN}m "
                  f"(size {b.size[0]:.2f} x {b.size[1]:.2f} m)")
        boxes = [b for b in boxes if not _oversized_blob(b)]
    scene.boxes = boxes
    # DIAG (temporary): the local-floor map's range + each box's bottom /
    # top vs the floor map at its centre -- pinpoints a uniform "raised by
    # the step" offset (floor-map level vs the cloud's own floor).
    try:
        _fv = np.asarray(fl(P[:, 0], P[:, 1]), dtype=float)
        print(f"[diag][floor] map over cloud: min={np.min(_fv):.2f} "
              f"med={np.median(_fv):.2f} max={np.max(_fv):.2f} | "
              f"cloud z p2={np.percentile(P[:, 2], 2):.2f} "
              f"mesh={is_mesh}")
        for b in boxes[:12]:
            c = np.asarray(b.center, dtype=float)
            fvb = float(fl(c[0], c[1]))
            print(f"[diag][floor] box {b.box_id[:6]} "
                  f"bottom={c[2] - b.size[2] / 2:.2f} "
                  f"top={c[2] + b.size[2] / 2:.2f} fl(centre)={fvb:.2f} "
                  f"xy=({c[0]:.1f},{c[1]:.1f})")
    except Exception as e:
        print(f"[diag][floor] failed: {type(e).__name__}: {e}")
    # result audit: one image per view -- the view's own raw VLM rects
    # (colored) plus the final fitted boxes (red) projected through the
    # same camera. Tiled views draw ALL boxes (cross-tile ones project
    # outside the frame), so each tile's audit stays self-contained.
    if out_dir:
        for idx, (img_v, cam_v, _, _, fname_v, rects_v) in enumerate(views):
            # non-tiled: views[0] (the NADIR view) owns grounded.png --
            # the before/after audit must compare the colored rects and
            # the red boxes on the view whose rays ARE the footprints;
            # the tilt views get their own grounded_L / grounded_R
            # audits (the last-view-writes overwrite used to put the
            # R view's perspective into the audit instead).
            fname_out = ("grounded.png" if (tiles is None and idx == 0)
                         else fname_v.replace("groundview", "grounded"))
            _save_grounded_png(
                img_v, cam_v, boxes, rects_v, out_dir, fname=fname_out)
    return True
