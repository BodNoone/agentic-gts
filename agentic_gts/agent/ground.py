"""VLM global 2D grounding -- the human surveyor's FIRST pass.

Instead of repairing a set of rough per-device boxes, the initial boxes
are downgraded to HINTS: the VLM looks at a top-down view of the whole
room and outlines EVERY device structure (a joined cabinet ROW counts as
one region), geometry turns each region into a full-depth 3D box, and a
per-row split pass (front+back face renders) resolves how many cabinets
each row box contains.

Why this kills the thin-fragment problem at its root: a region covers
the whole row footprint, so BOTH observed faces -- and the hollow
interior between them -- are inside one box from the start. The old
per-box pipeline had to re-discover the row depth for every fragment
(merge / depth-completion) and could not recover a face that had no box
at all; here the region IS the whole device extent.

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

def _render_topdown(scene, frame_boxes, yaw: float, W: int = 1280,
                    H: int = 1024, pan_deg: float = 0.0):
    """Base top-down render, no overlays. Camera fitted over the
    yaw-rotated cloud (rows parallel to the image axes), then rotated
    back into world so the GS render and the pixel back-projection
    share one consistent camera. Shared by the grounding input views
    and the grounded-result audit view.

    pan_deg=0: true nadir (rows axis-aligned in the image; an
    axis-aligned image rectangle captures a row exactly, but the room
    centre shows only the racks' TOP faces -- which a ground-level
    3DGS training set barely observed, so they render as a blurry
    smear the VLM cannot ground).
    pan_deg!=0: the fitted nadir camera ROTATED about the framing
    centre by a small angle (positive = view swings toward +y). One
    operation gives BOTH the slight sideways pan AND the slight tilt
    the user asked for: the camera height only drops by
    1-cos(a) (1.5% at 10 deg), but the view direction is now a
    degrees off vertical, so the racks directly under the ORIGINAL
    godview centre show their camera-facing FACES -- the parallax-only
    pan was too subtle to be useful (user report). up tilts with the
    camera so the layout stays map-like. NOT the old large tilt: the
    camera distance to the framing centre is strictly unchanged (no
    pull-back, no raising).

    Returns (img_float, cam, W, H).
    """
    from agentic_gts.output.gs_render import (Cam, make_godview_cam,
                                              render_gs_view)
    points = np.asarray(scene.points, dtype=np.float64)
    # ceiling cut from the box tops (same policy as the god-view): cut
    # 0.45m into the tallest structure so trays don't bury the layout
    top = max((b.center[2] + b.size[2] / 2.0 for b in frame_boxes),
              default=2.5)
    cut = float(top) - 0.45 if frame_boxes else float("inf")
    band = points[points[:, 2] < cut] if np.isfinite(cut) else points
    band = band[band[:, 2] > 0.30]
    if len(band) < 100:
        band = points
    pts_rot = _rot_xy(band, -yaw)
    # frame over the BOX footprint, not the raw cloud bbox (user
    # directive: the cloud-framed version raised the camera to fit
    # walls too, and the racks rendered small). The hint boxes are
    # only used for FRAMING (camera placement) -- they are not drawn
    # on the VLM input, so grounding itself stays hint-free.
    boxes_rot = []
    for b in frame_boxes:
        c = _rot_xy(np.array([[b.center[0], b.center[1], 0.0]]), -yaw)[0]
        boxes_rot.append(OrientedBox(center=(float(c[0]), float(c[1]),
                                             b.center[2]),
                                     size=b.size, yaw=0.0))
    cam_r = make_godview_cam(pts_rot, boxes_rot, nadir=True, W=W, H=H)
    if abs(pan_deg) > 1e-6:
        # small ROTATION of the fitted nadir camera about the framing
        # centre (row-frame x-axis): sideways pan + slight tilt in ONE
        # operation, camera-target distance strictly unchanged (no
        # pull-back -- that was the flaw of the old large tilt). Height
        # drops only by 1-cos(a); the off-vertical view direction is
        # what reveals the faces of the racks under the original
        # godview centre. 10 deg: strong enough to show faces, gentle
        # enough to keep the framing essentially complete.
        a = math.radians(-pan_deg)   # sign: positive pan -> eye at +y
        tgt = np.asarray(cam_r.target, dtype=float)
        eye0 = np.asarray(cam_r.eye, dtype=float)
        d = eye0 - tgt                       # nadir: ~[0, 0, +h]
        rx = np.array([[1.0, 0.0, 0.0],
                       [0.0, math.cos(a), -math.sin(a)],
                       [0.0, math.sin(a), math.cos(a)]])
        cam_r = Cam(eye=tgt + rx @ d,
                    target=tgt,
                    up=rx @ np.asarray(cam_r.up, dtype=float),
                    fovy_deg=cam_r.fovy_deg, W=W, H=H)
    # rotate the camera back into world (rotation about z: the nadir
    # axis and the tilt direction rotate with it)
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
            img = render_gs_view(gs, (), cam, cut_z=cut
                                if np.isfinite(cut) else None, cut_z_low=0.30)
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
    audit: no initial-hint overlay, grounding is independent of them)."""
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
        dr.text((10, 9), f"GROUNDING FAILED - kept hints: {why}"[:110],
                fill=(255, 255, 255), font=font)
        path = os.path.join(out_dir, "grounded.png")
        with open(path, "wb") as f:
            f.write(png_bytes(np.asarray(pil, dtype=np.float32) / 255.0))
        print(f"[ground] FAILURE audit render -> {path} ({why})")
    except Exception as e:
        print(f"[ground] failure render failed ({type(e).__name__}: {e})")


def _fit_region_box(points: np.ndarray, rect, min_pts: int = 60):
    """Fit a full-depth OBB (yaw=0; points already in the row-aligned
    frame) to the points inside a grounded 2D rect.

    The rect is the VLM's coarse outline; the DEVICE-BAND point support
    snaps the edges (percentiles -- floor points sit outside the device
    footprint only where the rect is loose, walls are excluded by height
    structure downstream). z comes from the device band (floor excluded:
    devices stand ON the ground at z~0, so the box bottom is 0 and the
    top is the band's 99.5th percentile).

    Guards reject hallucinated regions (no support) and floor patches
    (no height): a VLM box drawn over empty floor never becomes a real
    device box.
    """
    x0, y0, x1, y1 = rect
    m = ((points[:, 0] >= x0) & (points[:, 0] <= x1) &
         (points[:, 1] >= y0) & (points[:, 1] <= y1))
    pts = points[m]
    if len(pts) < min_pts:
        return None
    dev = pts[pts[:, 2] > 0.30]      # device band: exclude floor texture
    if len(dev) < max(30, min_pts // 2):
        return None                  # floor patch, no structure
    z_top = float(np.percentile(dev[:, 2], 99.5))
    if z_top < 0.50:
        return None                  # too short for a device
    lo = np.percentile(dev[:, :2], 0.5, axis=0)
    hi = np.percentile(dev[:, :2], 99.5, axis=0)
    dx, dy = float(hi[0] - lo[0]), float(hi[1] - lo[1])
    if dx < 0.30 or dy < 0.20:
        return None                  # sliver, not a structure
    c = (lo + hi) / 2.0
    return OrientedBox(center=(float(c[0]), float(c[1]), z_top / 2.0),
                       size=(dx, dy, z_top), yaw=0.0,
                       device_type=DeviceType.RACK)


def _merge_rects(rects: list[tuple], iou_thr: float = 0.25,
                 along_gap: float = 0.5, cross_gap: float = 0.35,
                 cross_union: float = 1.6,
                 pts: np.ndarray | None = None) -> list[tuple]:
    """Greedy union merge of grounded rects (row frame: x = along the
    row, y = depth).

    The same row is typically outlined in SEVERAL views (nadir + the
    two obliques); each capture is coarse in its own way (nadir: blurry
    tops; oblique: perspective stretch). Union + a fresh point-support
    fit keeps the most inclusive footprint per structure. Three ways the
    same structure's rects fuse:
      1. OVERLAP: intersection > iou_thr of the smaller rect.
      2. ALONG-ROW adjacency: one long row outlined in pieces -- gap
         <= along_gap on x with substantial y alignment. Requires
         POINT SUPPORT in the gap strip when pts is given: a row
         outlined in pieces is physically continuous (device points
         between the pieces), while colinear-but-SEPARATE rows have an
         empty cross aisle in the gap -- chaining those together was
         the over-merge the user reported. The split stage divides
         rows LATER; grounding must capture them whole.
      3. DEPTH complement: front- and back-face fragments of one rack
         (small y gap, strong x overlap, combined depth <= cross_union
         -- two full parallel rows stacked in y always exceed it).
    Parallel rows never fuse: their y gap is aisle-scale, and two full
    rows stacked in y exceed cross_union.
    """
    rs = [list(map(float, r)) for r in rects]

    def _area(r):
        return max(0.0, r[2] - r[0]) * max(0.0, r[3] - r[1])

    def _gap_supported(a, b):
        """Device points present in the x-gap strip between two pieces
        (row frame, pts pre-filtered to the device band). Touching
        pieces are always supported."""
        gx0, gx1 = min(a[2], b[2]), max(a[0], b[0])
        if gx1 - gx0 <= 0.05:
            return True
        if pts is None or not len(pts):
            return True
        m = ((pts[:, 0] >= gx0) & (pts[:, 0] <= gx1) &
             (pts[:, 1] >= max(a[1], b[1])) &
             (pts[:, 1] <= min(a[3], b[3])))
        return int(m.sum()) >= 5

    changed = True
    while changed:
        changed = False
        for i in range(len(rs)):
            for j in range(i + 1, len(rs)):
                a, b = rs[i], rs[j]
                ix = min(a[2], b[2]) - max(a[0], b[0])
                iy = min(a[3], b[3]) - max(a[1], b[1])
                inter = max(0.0, ix) * max(0.0, iy)
                smaller = min(_area(a), _area(b))
                merge = smaller > 0 and inter > iou_thr * smaller
                if not merge and iy > 0 and ix >= -along_gap:
                    # along-row pieces: touching (or a hair apart) on x,
                    # aligned on y, and the structure continues through
                    # the gap
                    ha, hb = a[3] - a[1], b[3] - b[1]
                    if (min(ha, hb) > 0 and iy >= 0.5 * min(ha, hb)
                            and _gap_supported(a, b)):
                        merge = True
                if not merge and ix > 0 and iy >= -cross_gap:
                    # depth complement: front/back face fragments
                    wa, wb = a[2] - a[0], b[2] - b[0]
                    y_union = max(a[3], b[3]) - min(a[1], b[1])
                    if (min(wa, wb) > 0 and ix >= 0.8 * min(wa, wb)
                            and y_union <= cross_union):
                        merge = True
                if merge:
                    rs[i] = [min(a[0], b[0]), min(a[1], b[1]),
                             max(a[2], b[2]), max(a[3], b[3])]
                    rs.pop(j)
                    changed = True
                    break
            if changed:
                break
    return [tuple(r) for r in rs]


def _primary_merge(prim_rects: list[tuple], obl_rects: list[tuple],
                   pts: np.ndarray | None = None) -> list[tuple]:
    """PRIMARY-VIEW merge policy (user-directed).

    The nadir view owns the footprint: its rects merge among
    themselves (fragment rules) and form the BASE set. A tilted view's
    rect can only ADD: when it overlaps any accepted rect it is simply
    DROPPED -- the nadir capture is the exact footprint (vertical rays,
    no perspective dilation), and union-ing the tilted view's
    perspective slop into it was what inflated the fitted boxes. Only a
    rect covering something the nadir MISSED (the blurry-top centre
    rows) survives and is fitted as its own structure.
    """
    def _ovf(a, b):
        ix = max(0.0, min(a[2], b[2]) - max(a[0], b[0]))
        iy = max(0.0, min(a[3], b[3]) - max(a[1], b[1]))
        inter = ix * iy
        sm = min((a[2] - a[0]) * (a[3] - a[1]),
                 (b[2] - b[0]) * (b[3] - b[1]))
        return inter / sm if sm > 0 else 0.0

    accepted = list(_merge_rects(prim_rects, pts=pts))
    for r in obl_rects:
        if any(_ovf(r, m) > 0.15 for m in accepted):
            continue          # the primary view owns this structure
        accepted.append(r)
    return accepted


def _tighten_oblique(cam, r, yaw, pts_fit, frame_rect_fn):
    """Tighten a tilted view's rect via two-plane back-projection.

    A rect the VLM draws on a TILTED view covers the structure's full
    image height; back-projected onto one z plane it is the footprint
    DILATED by (H - z)*tan(tilt) on the y sides (~0.4m total at 10 deg
    for a 2.2m rack) -- enough to swallow a neighbouring row 0.2m away
    and to union-inflate the correct nadir capture (user report: red
    fitted boxes all too large while the raw rects were right).

    The footprint is contained in the back-projection at EVERY plane
    z in [0, H], and the rect's frustum cross-section shifts
    monotonically with z, so INTERSECTING the z=0 and z=z_c
    back-projections (any z_c <= H) never clips the true footprint
    while removing nearly all of the dilation. z_c is estimated as
    0.8x the structure height under the coarse rect (strictly below
    the true height, keeping the no-clip guarantee even when the
    estimate runs hot).
    """
    coarse = frame_rect_fn(cam, r, 1.0)
    m = ((pts_fit[:, 0] >= coarse[0]) & (pts_fit[:, 0] <= coarse[2]) &
         (pts_fit[:, 1] >= coarse[1]) & (pts_fit[:, 1] <= coarse[3]))
    inner = pts_fit[m]
    h_est = float(np.percentile(inner[:, 2], 99.5)) if len(inner) else 2.0
    z_c = min(max(0.8 * h_est, 0.8), 2.2)
    lo = frame_rect_fn(cam, r, 0.0)
    hi = frame_rect_fn(cam, r, z_c)
    tight = (max(lo[0], hi[0]), max(lo[1], hi[1]),
             min(lo[2], hi[2]), min(lo[3], hi[3]))
    if tight[2] <= tight[0] or tight[3] <= tight[1]:
        return coarse            # degenerate: keep the coarse rect
    return tight


# ---------- front-view height (the 2-pass fit the user asked for) ----------

def _pixel_ray_z(cam, u: float, v: float, plane_p, n) -> float:
    """World z where the pixel (u, v)'s ray crosses the vertical plane
    through plane_p with horizontal normal n. Exact inverse of
    Cam.project_cv along the ray (same algebra as unproject_ground,
    plane axis generalised)."""
    Kinv = np.linalg.inv(cam.K())
    V = cam.view_cv()
    R, t = V[:3, :3], V[:3, 3]
    centre = -R.T @ t                       # camera centre (world)
    ray = (Kinv @ np.array([u, v, 1.0])) @ R  # world ray direction
    denom = float(ray @ n)
    if abs(denom) < 1e-9:
        return float("nan")
    s = float((np.asarray(plane_p, float) - centre) @ n) / denom
    return float((centre + s * ray)[2])


def _front_view_height(scene, box, judge, hint_top,
                       out_dir: str | None = None, idx: int = 0):
    """Region height from its FRONT elevation.

    The two-pass design the user specified: the top-down 2D fit owns
    the FOOTPRINT, the front view owns the HEIGHT -- ground-level 3DGS
    training observed rack faces well (tops are the blurry part), so
    the VLM boxes the row floor-to-top on a front render and the bbox's
    vertical extent, measured through the camera-facing face plane, is
    the height in metres. The percentile z from the fit stays as the
    fallback when the VLM answers nothing sane. Returns float | None."""
    import os
    from agentic_gts.output.gs_render import (make_local_cam,
                                              render_gs_view, png_bytes)
    W, H = 1024, 768
    try:
        cam = make_local_cam([box], extent=1.5, W=W, H=H,
                             elev_deg=10.0, azim_deg=0.0)
    except Exception as e:
        print(f"[ground] front cam failed ({type(e).__name__}: {e})")
        return None
    cut = float(hint_top) + 0.10
    img = None
    gs_ply = scene.meta.get("gs_ply")
    if gs_ply:
        try:
            from agentic_gts.tools.gs_io import read_gaussian_ply
            gs = read_gaussian_ply(gs_ply)
            img = render_gs_view(gs, (), cam, cut_z=cut)
        except Exception as e:
            print(f"[ground] front GS render failed "
                  f"({type(e).__name__}: {e}) -> scatter")
    if img is None:
        pts = np.asarray(scene.points, dtype=float)
        # floor KEPT: the VLM needs the floor-to-rack boundary at the
        # bottom of the image to place the bbox's lower edge
        img = _projected_scatter(pts[pts[:, 2] < cut], cam, W, H)
    png = png_bytes(img)
    png_path = None
    if out_dir:
        png_path = os.path.join(out_dir, f"frontview_{idx:02d}.png")
        try:
            with open(png_path, "wb") as f:
                f.write(png)
        except Exception as e:
            print(f"[ground] front png save failed ({type(e).__name__})")
            png_path = None
    rects = judge.ground_regions(png, W, H, png_path=png_path, front=True)
    if not rects:
        return None
    # the rect over THIS region: nearest x-centre to the box centre's
    # projection (a front view may also catch the row behind the aisle)
    ctr = np.asarray(box.center, dtype=float)
    uv = cam.project_cv(ctr[None, :])[0]
    best = min(rects, key=lambda r: abs((r[0] + r[2]) / 2.0 - uv[0]))
    u = (best[0] + best[2]) / 2.0
    yaw = float(box.yaw)
    d = np.array([-math.sin(yaw), math.cos(yaw), 0.0])
    face_p = ctr + d * (box.size[1] / 2.0)   # camera-facing face plane
    z_top = _pixel_ray_z(cam, u, best[1], face_p, d)
    z_bot = _pixel_ray_z(cam, u, best[3], face_p, d)
    if not (np.isfinite(z_top) and np.isfinite(z_bot)):
        return None
    h = z_top - z_bot
    if not (0.5 <= h <= 4.5) or z_bot > 0.6:
        print(f"[ground] front height rejected "
              f"(z_top {z_top:.2f}, z_bot {z_bot:.2f}) -> percentile")
        return None
    return float(h)


def ground_stage(scene, judge, out_dir: str | None = None) -> bool:
    """Replace scene.boxes with VLM-grounded full-depth row boxes.

    Two-view capture: the true NADIR view (rows axis-aligned, exact
    footprint) plus ONE OBLIQUE view whose well-trained rack FACES compensate the nadir's
    blind spot -- a ground-level 3DGS training set barely observed rack
    tops, so the nadir room centre renders as an ungroundable smear
    while the image edges (perspective showing faces) ground fine.
    Regions from all views are back-projected, union-merged, and
    point-support fitted into full-depth row boxes.

    False = grounding unavailable (mock backend / VLM failure / no
    region survived the point-support guards) and the caller keeps the
    original hint boxes -- grounding must never destroy the layout.
    """
    import os
    from agentic_gts.output.gs_render import png_bytes, unproject_ground
    hints = list(scene.boxes)
    if not hints:
        return False
    yaw = float(scene.meta.get("yaw", 0.0) or 0.0)
    views = (("nadir", 0.0),           # exact footprint capture
             ("oblique", 10.0))        # one face-visible view for missed rows
    cam_rects: list[tuple] = []       # (cam, pixel_rect, oblique)
    base = None                       # (img, cam) of the nadir render
    view_audit: list = []              # (name, img, cam, rects) per view
    for name, tilt in views:
        try:
            img, cam, W, H = _render_topdown(scene, hints, yaw,
                                             pan_deg=tilt)
            png = png_bytes(img)       # CLEAN view: no hint overlays
        except Exception as e:
            print(f"[ground] view {name} render failed "
                  f"({type(e).__name__}: {e}) -> view skipped")
            continue
        if name == "nadir":
            base = (img, cam)
        png_path = None
        if out_dir:
            fname = "groundview.png" if name == "nadir" \
                else f"groundview_{name}.png"
            png_path = os.path.join(out_dir, fname)
            try:
                with open(png_path, "wb") as f:
                    f.write(png)
            except Exception as e:
                print(f"[ground] png save failed ({type(e).__name__})")
                png_path = None
        rects = judge.ground_regions(png, W, H, png_path=png_path,
                                     oblique=tilt != 0.0)
        print(f"[ground] view {name}: {len(rects)} regions")
        view_audit.append((name, img, cam, rects))
        cam_rects += [(cam, r, tilt != 0.0) for r in rects]
    if not cam_rects:
        print("[ground] VLM returned no usable regions -> keep hints")
        if out_dir and base is not None:
            _save_grounded_fail_png(base[0], out_dir,
                                    "VLM returned no usable regions")
        return False
    pts_rot = _rot_xy(np.asarray(scene.points, dtype=np.float64), -yaw)
    # FIT points: the device band only. The render band cuts at
    # hint_top - 0.45, but the FIT must keep the rack top, so cut at
    # hint_top + 0.1: everything above (ceiling / cable trays -- the
    # raw cloud still carries them) is excluded. Ceiling points span
    # the WHOLE room in XY, so even a correct rect whose fit included
    # them produced a tray-height box hugging the loose rect edges
    # (user report: red boxes all too large and wrong while the raw
    # colored rects were right).
    hint_top = max((b.center[2] + b.size[2] / 2.0 for b in hints),
                   default=2.5)
    pts_fit = pts_rot[(pts_rot[:, 2] > 0.30) &
                      (pts_rot[:, 2] <= hint_top + 0.10)]
    if len(pts_fit) < 100:
        pts_fit = pts_rot[pts_rot[:, 2] > 0.30]

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

    prim_rects: list[tuple] = []   # nadir: the exact footprint capture
    obl_rects: list[tuple] = []   # tilted views: perspective-sloped captures
    for cam, r, oblique in cam_rects:
        rect = _frame_rect(cam, r, 1.0)
        if oblique:
            rect = _tighten_oblique(cam, r, yaw, pts_fit, _frame_rect)
            obl_rects.append(rect)
        else:
            prim_rects.append(rect)
    merged = _primary_merge(prim_rects, obl_rects, pts_fit)
    boxes = []
    for rect_r in merged:
        bb = _fit_region_box(pts_fit, rect_r)
        if bb is None:
            continue
        c = _rot_xy(np.array([[bb.center[0], bb.center[1], 0.0]]), yaw)[0]
        box = OrientedBox(center=(float(c[0]), float(c[1]), bb.center[2]),
                          size=bb.size, yaw=yaw,
                          device_type=DeviceType.RACK,
                          meta={"grounded": True})
        # HEIGHT from the front view (user-directed division of labour):
        # the top-down 2D fit owns the footprint; the well-trained rack
        # FACES on a front elevation own the height. The fit's
        # percentile z stays as the fallback when the VLM is unusable.
        h = _front_view_height(scene, box, judge, hint_top,
                               out_dir=out_dir, idx=len(boxes))
        if h is not None:
            box = OrientedBox(center=(float(c[0]), float(c[1]), h / 2.0),
                              size=(bb.size[0], bb.size[1], h), yaw=yaw,
                              device_type=DeviceType.RACK,
                              meta={"grounded": True})
        boxes.append(box)
    if not boxes:
        print("[ground] no region survived the point-support guards "
              "-> keep hints")
        if out_dir and base is not None:
            _save_grounded_fail_png(base[0], out_dir,
                                    "no region survived point-support guards")
        return False
    print(f"[ground] {len(cam_rects)} VLM regions in {len(views)} views "
          f"-> {len(merged)} merged -> {len(boxes)} full-depth row boxes")
    scene.boxes = boxes
    # result audit on EVERY view (user request): each grounded_*.png
    # shows that view's own raw VLM rects (colored) plus the final
    # fitted boxes (red) projected through the same camera -- the nadir
    # one alone could not show what the oblique views contributed
    if out_dir:
        for name, img, cam, vrects in view_audit:
            _save_grounded_png(img, cam, boxes, vrects, out_dir,
                               fname=("grounded.png" if name == "nadir"
                                      else f"grounded_{name}.png"))
    return True


# ---------- per-row split ----------

def _split_row(scene, box, count: int, gaps: list) -> list:
    """Execute one row split: VLM fractions snapped to density gaps."""
    from agentic_gts.tools import geometry as geo
    if count <= 1 and not gaps:
        return [box]
    L = float(box.size[0])
    half = L / 2.0
    prof = geo.profile_cuts(scene, box)
    geo_gaps = list(prof.get("gaps") or [])
    if gaps:
        cuts = []
        for g in gaps:
            pos = -half + float(g) * L      # VLM fraction -> local x
            # snap to a MEASURED gap centre within 12% of the row length
            near = [c for c in geo_gaps if abs(c - pos) <= 0.12 * L]
            cuts.append(min(near, key=lambda c: abs(c - pos)) if near else pos)
    elif geo_gaps:
        # no usable VLM fractions: the measured gaps are the truth
        cuts = geo_gaps
    else:
        # no gaps measurable ANYWHERE (flush cabinets leave no point gap
        # -- door seams only): divide at the VLM's count
        if count <= 1:
            return [box]
        return geo.split_box(scene, box, n=count)
    cuts = sorted(set(round(float(c), 3) for c in cuts))[:max(count, 1)]
    subs = geo.split_box(scene, box, n=max(count, len(cuts) + 1), cuts=cuts)
    if len(subs) <= 1:
        return [box]
    # re-fit each piece so its edges snap to its OWN point support
    out = []
    for s in subs:
        rb = geo.fit_box_to_points(scene, (s.center[0], s.center[1]),
                                   s.size, s.yaw,
                                   keep_height=True, keep_depth=True)
        out.append(rb if rb is not None else s)
    return out


def split_stage(scene, judge, out_dir: str | None = None) -> None:
    """Per row-box: VLM counts the cabinets inside (front+back face
    renders); geometry snaps the cuts and re-fits each piece."""
    out = []
    for box in list(scene.boxes):
        v = judge.adjudicate_split(scene, box)
        p = (v.params or {}) if v is not None else {}
        try:
            count = max(1, int(p.get("count", 1)))
        except (TypeError, ValueError):
            count = 1
        gaps = [g for g in (p.get("gaps") or [])
                if isinstance(g, (int, float)) and 0.0 < float(g) < 1.0]
        out.extend(_split_row(scene, box, count, gaps))
    n = len(out) - len(scene.boxes)
    print(f"[split] row boxes -> {len(out)} device boxes "
          f"({'+' if n >= 0 else ''}{n})")
    scene.boxes = out
