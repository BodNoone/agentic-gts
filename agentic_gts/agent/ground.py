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
                    H: int = 1024):
    """Base top-down render, no overlays. Camera fitted over the
    yaw-rotated cloud (rows parallel to the image axes), then rotated
    back into world so the GS render and the pixel back-projection
    share one consistent camera. Shared by the grounding input view
    and the grounded-result audit view.

    True nadir: rows axis-aligned in the image (an axis-aligned image
    rectangle captures a row exactly, vertical rays carry no
    perspective dilation).

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


# ---------- grounding stage ----------


def ground_stage(scene, judge, out_dir: str | None = None) -> bool:
    """Replace scene.boxes with VLM-grounded per-region boxes.

    ONE global NADIR view (rows axis-aligned, exact footprint capture,
    vertical rays carry no perspective dilation). Regions are
    back-projected and point-support fitted -- each region its OWN
    box, unmerged. The per-box local refinement (front + oblique
    renders -> VLM SAM points -> mask -> back-projected points ->
    precise OBB) and the row split run downstream.

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
    try:
        img, cam, W, H = _render_topdown(scene, hints, yaw)
        png = png_bytes(img)           # CLEAN view: no hint overlays
    except Exception as e:
        print(f"[ground] nadir render failed ({type(e).__name__}: {e}) "
              f"-> keep hints")
        return False
    png_path = None
    if out_dir:
        png_path = os.path.join(out_dir, "groundview.png")
        try:
            with open(png_path, "wb") as f:
                f.write(png)
        except Exception as e:
            print(f"[ground] png save failed ({type(e).__name__})")
            png_path = None
    rects = judge.ground_regions(png, W, H, png_path=png_path)
    print(f"[ground] nadir view: {len(rects)} regions")
    if not rects:
        print("[ground] VLM returned no usable regions -> keep hints")
        if out_dir:
            _save_grounded_fail_png(img, out_dir,
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

    # NO merging: each grounded rect is fitted as its OWN box (user
    # directive). The per-box local refinement (SAM) runs next and the
    # joined rows are split after it -- pre-merging decided structure
    # membership before the refinement evidence got a vote.
    boxes = []
    for r in rects:
        rect_r = _frame_rect(cam, r, 1.0)
        bb = _fit_region_box(pts_fit, rect_r)
        if bb is None:
            continue
        c = _rot_xy(np.array([[bb.center[0], bb.center[1], 0.0]]), yaw)[0]
        box = OrientedBox(center=(float(c[0]), float(c[1]), bb.center[2]),
                          size=bb.size, yaw=yaw,
                          device_type=DeviceType.RACK,
                          meta={"grounded": True})
        boxes.append(box)
    if not boxes:
        print("[ground] no region survived the point-support guards "
              "-> keep hints")
        if out_dir:
            _save_grounded_fail_png(img, out_dir,
                                    "no region survived point-support guards")
        return False
    print(f"[ground] {len(rects)} VLM regions -> {len(boxes)} fitted boxes")
    scene.boxes = boxes
    # result audit: the grounded.png shows the view's own raw VLM rects
    # (colored) plus the final fitted boxes (red) projected through the
    # same camera
    if out_dir:
        _save_grounded_png(img, cam, boxes, rects, out_dir,
                           fname="grounded.png")
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
