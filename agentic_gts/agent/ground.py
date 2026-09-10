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

def _render_ground_view(scene, hints, yaw: float, W: int = 1280, H: int = 1024):
    """Top-down evidence image for grounding, rows AXIS-ALIGNED in the image.

    The camera is fitted over the yaw-rotated cloud (rows parallel to the
    image axes, so an axis-aligned image rectangle captures a rotated row
    exactly), then rotated back into world so the GS render and the pixel
    back-projection share one consistent camera. Hints are drawn as thin
    gray dashed outlines + centre crosses -- anchors for the VLM's own
    grounding, visually distinct from the red adjudication frames.

    Returns (png_bytes, cam, W, H).
    """
    from agentic_gts.output.gs_render import (Cam, make_godview_cam,
                                              png_bytes, render_gs_view)
    points = np.asarray(scene.points, dtype=np.float64)
    # ceiling cut from the hint tops (same policy as the god-view): cut
    # 0.45m into the tallest structure so trays don't bury the layout
    top = max((b.center[2] + b.size[2] / 2.0 for b in hints), default=2.5)
    cut = float(top) - 0.45 if hints else float("inf")
    band = points[points[:, 2] < cut] if np.isfinite(cut) else points
    band = band[band[:, 2] > 0.30]
    if len(band) < 100:
        band = points
    pts_rot = _rot_xy(band, -yaw)
    # rotate the hint footprints too so the camera frames the layout
    hints_rot = []
    for b in hints:
        c = _rot_xy(np.array([[b.center[0], b.center[1], 0.0]]), -yaw)[0]
        hints_rot.append(OrientedBox(center=(float(c[0]), float(c[1]),
                                             b.center[2]),
                                     size=b.size, yaw=0.0))
    cam_r = make_godview_cam(pts_rot, hints_rot, nadir=True, W=W, H=H)
    # rotate the camera back into world (rotation about z keeps it nadir)
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
    img = _draw_hints(img, cam, hints)
    return png_bytes(img), cam, W, H


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


def _draw_hints(img: np.ndarray, cam, hints) -> np.ndarray:
    """Thin gray dashed outlines + centre crosses over the grounding view."""
    from PIL import Image, ImageDraw
    u8 = (np.clip(img, 0, 1) * 255).astype(np.uint8)[..., :3].copy()
    pil = Image.fromarray(u8)
    dr = ImageDraw.Draw(pil)
    gray = (150, 150, 150)
    for b in hints:
        z = b.center[2] + b.size[2] / 2.0
        cs = b.corners_2d()
        uv = cam.project_cv(np.column_stack([cs, np.full(len(cs), z)]))
        pts = np.round(uv).astype(np.int32)
        # dashed rectangle: alternate 6-px draw / 4-px skip per edge
        for i in range(4):
            a, c = pts[i], pts[(i + 1) % 4]
            n = max(int(np.linalg.norm(c - a)) // 5, 1)
            for k in range(0, n, 2):
                t0, t1 = k / n, min((k + 1) / n, 1.0)
                p0 = a + (c - a) * t0
                p1 = a + (c - a) * t1
                dr.line((*np.round(p0).astype(int), *np.round(p1).astype(int)),
                        fill=gray, width=1)
        # centre cross (the 'hint point')
        cc = cam.project_cv(np.array([[b.center[0], b.center[1], z]]))[0]
        ci = np.round(cc).astype(int)
        dr.line((int(ci[0] - 5), int(ci[1]), int(ci[0] + 5), int(ci[1])),
                fill=gray, width=1)
        dr.line((int(ci[0]), int(ci[1] - 5), int(ci[0]), int(ci[1] + 5)),
                fill=gray, width=1)
    return np.asarray(pil, dtype=np.float32) / 255.0


# ---------- region -> 3D box ----------

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


def ground_stage(scene, judge, out_dir: str | None = None) -> bool:
    """Replace scene.boxes with VLM-grounded full-depth row boxes.

    False = grounding unavailable (mock backend / VLM failure / no
    region survived the point-support guards) and the caller keeps the
    original hint boxes -- grounding must never destroy the layout.
    """
    from agentic_gts.output.gs_render import unproject_ground
    hints = list(scene.boxes)
    if not hints:
        return False
    yaw = float(scene.meta.get("yaw", 0.0) or 0.0)
    try:
        png, cam, W, H = _render_ground_view(scene, hints, yaw)
    except Exception as e:
        print(f"[ground] render failed ({type(e).__name__}: {e}) -> keep hints")
        return False
    png_path = None
    if out_dir:
        import os
        png_path = os.path.join(out_dir, "groundview.png")
        try:
            with open(png_path, "wb") as f:
                f.write(png)
        except Exception as e:
            print(f"[ground] png save failed ({type(e).__name__}: {e})")
            png_path = None
    rects = judge.ground_regions(png, W, H, png_path=png_path)
    if not rects:
        print("[ground] VLM returned no usable regions -> keep hints")
        return False
    pts_rot = _rot_xy(np.asarray(scene.points, dtype=np.float64), -yaw)
    z_plane = 1.0      # nadir rays are parallel: any plane height works
    boxes = []
    for r in rects:
        uv = np.array([[r[0], r[1]], [r[2], r[1]], [r[2], r[3]], [r[0], r[3]]],
                      dtype=float)
        corners_w = unproject_ground(cam, uv, z_plane)[:, :2]
        # image rects are axis-aligned in the ROW frame: rotate the world
        # corners back into it and take the AABB
        corners_r = _rot_xy(np.column_stack([corners_w,
                                             np.zeros(len(corners_w))]),
                            -yaw)[:, :2]
        rect_r = (float(corners_r[:, 0].min()), float(corners_r[:, 1].min()),
                  float(corners_r[:, 0].max()), float(corners_r[:, 1].max()))
        bb = _fit_region_box(pts_rot, rect_r)
        if bb is None:
            continue
        c = _rot_xy(np.array([[bb.center[0], bb.center[1], 0.0]]), yaw)[0]
        boxes.append(OrientedBox(center=(float(c[0]), float(c[1]),
                                         bb.center[2]),
                                 size=bb.size, yaw=yaw,
                                 device_type=DeviceType.RACK,
                                 meta={"grounded": True}))
    if not boxes:
        print("[ground] no region survived the point-support guards "
              "-> keep hints")
        return False
    print(f"[ground] {len(rects)} VLM regions -> {len(boxes)} "
          f"full-depth row boxes")
    scene.boxes = boxes
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
