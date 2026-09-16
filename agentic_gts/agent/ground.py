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

def _render_cut(top: float | None) -> float:
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
    """
    if not top:
        return float("inf")
    top = float(top)
    return min(top - 0.10, max(0.70 * top, 1.0))


def _render_topdown(scene, yaw: float, W: int = 1280, H: int = 1024):
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
    hint-box input anymore.

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
    cut = _render_cut(top)
    band = points[points[:, 2] < cut] if np.isfinite(cut) else points
    band = band[band[:, 2] > 0.30]
    if len(band) < 100:
        # thin band (very low structures): drop ONLY the top cut, keep
        # the floor cut -- falling back to the raw cloud would pull the
        # ceiling back into the view, which is exactly what the cut
        # exists to remove
        band = points[points[:, 2] > 0.30]
    pts_rot = _rot_xy(band, -yaw)
    # frame over the BOOTSTRAP layout, not the raw cloud bbox (user
    # directive: the cloud-framed version raised the camera to fit
    # walls too, and the racks rendered small). Walls were already
    # dropped as boundary cells, so the framing hugs the layout.
    # The kept CELLS are rotated by the ACTUAL yaw and AABB'd ONCE:
    # rotating the world-frame device_footprint AABB instead double-
    # inflates for rotated layouts (AABB of a 45-deg row, then AABB
    # of rotating that box) -- the camera rose and the view came back
    # mostly empty.
    boxes_rot = []
    lo = hi = None
    cells = scene.meta.get("device_cells")
    if cells is not None and len(cells):
        cr = _rot_xy(np.column_stack([cells,
                                       np.zeros(len(cells))]), -yaw)
        lo, hi = cr.min(axis=0), cr.max(axis=0)
    else:
        fp = scene.meta.get("device_footprint")
        if fp:
            corners_w = np.array([[fp[0], fp[1]], [fp[2], fp[1]],
                                  [fp[2], fp[3]], [fp[0], fp[3]]])
            cr = _rot_xy(np.column_stack([corners_w,
                                          np.zeros(4)]), -yaw)
            lo, hi = cr.min(axis=0), cr.max(axis=0)
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
    # anchored column top, P99.5 fallback: the rect's points form one
    # union column (mixed-height cabinets, all standing on the ground),
    # so the density-connected run's top is the row's true tallest --
    # a percentile lets floating overhead clutter inside the rect drag
    # it higher (same failure the hint-free bootstrap z_top had)
    from agentic_gts.agent.mask_refine import _anchored_top
    _at = _anchored_top(dev[:, 2])
    z_top = float(_at) if _at is not None \
        else float(np.percentile(dev[:, 2], 99.5))
    if z_top < 0.50:
        return None                  # too short for a device
    lo = np.percentile(dev[:, :2], 0.5, axis=0)
    hi = np.percentile(dev[:, :2], 99.5, axis=0)
    dx, dy = float(hi[0] - lo[0]), float(hi[1] - lo[1])
    if dx < 0.30 or dy < 0.20:
        return None                  # sliver, not a structure
    c = (lo + hi) / 2.0
    # Ride the LONG side on the yaw axis (size[0]): a row that runs
    # along the rotated-y axis still fits here as (dx, dy) with
    # yaw=0 -- but then the box's yaw axis is its THICKNESS, and
    # refine_box (which projects along-row spans on the seed's yaw
    # axis) splits the row ACROSS its depth (user report: a joined
    # row split into 3 pieces along the thickness, not the row).
    if dy > dx:
        return OrientedBox(center=(float(c[0]), float(c[1]), z_top / 2.0),
                           size=(dy, dx, z_top), yaw=math.pi / 2.0,
                           device_type=DeviceType.RACK,
                           meta={"n_pts": len(dev)})
    return OrientedBox(center=(float(c[0]), float(c[1]), z_top / 2.0),
                       size=(dx, dy, z_top), yaw=0.0,
                       device_type=DeviceType.RACK,
                       meta={"n_pts": len(dev)})


def _merge_adjacent_boxes(boxes: list, pts_fit: np.ndarray, yaw: float,
                          touch_tol: float = 0.10,
                          bridge_tol: float = 0.50,
                          min_gap_pts: int = 15) -> list:
    """Merge tightly-ADJACENT grounded boxes; splitting is stageC's job.

    The VLM sometimes over-splits ONE physical structure into several
    tight rects (a regular layout reads as several bright bands); each
    rect fits its own box and the seam never heals later -- stageC only
    SPLITS, never merges. Adjacent boxes (same orientation bucket, real
    band overlap on the perpendicular axis) merge when they either
    TOUCH (gap <= touch_tol) or sit across a small gap that the device
    band FILLS: bridge points strictly inside the gap mean the
    structure is continuous, an empty gap is a real cut (the old B0
    convention that kept 0.3-0.5m lateral gaps between separate rows
    unmerged -- surface points hug the box edges, so the interior is
    probed 5cm inside each side). Each union is REFITTED to point
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
                if gap <= touch_tol:
                    parent[_find(i)] = _find(j)
                    break
                # small gap: merge only when the device band fills it
                # (probe strictly inside -- 5cm off each box edge, so
                # the two FACING SURFACES never count as a bridge)
                lo_slab = min(a[axis + 2], b[axis + 2]) + 0.05
                hi_slab = max(a[axis], b[axis]) - 0.05
                if hi_slab <= lo_slab:
                    continue
                m = ((pts_fit[:, axis] >= lo_slab) &
                     (pts_fit[:, axis] <= hi_slab) &
                     (pts_fit[:, o] >= p_lo) & (pts_fit[:, o] <= p_hi))
                if int(m.sum()) >= min_gap_pts:
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
        bb = _fit_region_box(pts_fit, u)
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
    try:
        img, cam, W, H = _render_topdown(scene, yaw)
        png = png_bytes(img)           # CLEAN view: no overlays
    except Exception as e:
        print(f"[ground] nadir render failed ({type(e).__name__}: {e})")
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
        print("[ground] VLM returned no usable regions")
        if out_dir:
            _save_grounded_fail_png(img, out_dir,
                                    "VLM returned no usable regions")
        return False
    pts_rot = _rot_xy(np.asarray(scene.points, dtype=np.float64), -yaw)
    # FIT points: the device band only. The render band cuts lower
    # (relative to the device top), but the FIT must keep the rack
    # top, so cut at z_top + 0.1: everything above (ceiling / cable
    # trays -- the raw cloud still carries them) is excluded. Ceiling
    # points span the WHOLE room in XY, so even a correct rect whose
    # fit included them produced a tray-height box hugging the loose
    # rect edges (user report: red boxes all too large and wrong while
    # the raw colored rects were right).
    fit_top = float(scene.meta.get("z_top", 2.5) or 2.5)
    pts_fit = pts_rot[(pts_rot[:, 2] > 0.30) &
                      (pts_rot[:, 2] <= fit_top + 0.10)]
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

    # Each grounded rect is fitted as its OWN box; tightly-ADJACENT
    # over-split pieces of one structure merge right after the dedup
    # (below) -- the local refine then splits the merged unions, so
    # structure membership is decided on refinement evidence, never
    # pre-merged beyond touching pieces.
    boxes = []
    for r in rects:
        rect_r = _frame_rect(cam, r, 1.0)
        bb = _fit_region_box(pts_fit, rect_r)
        if bb is None:
            continue
        c = _rot_xy(np.array([[bb.center[0], bb.center[1], 0.0]]), yaw)[0]
        # bb.yaw is 0 (row along the rotated-x axis) or pi/2 (row along
        # rotated-y): both rotate into the world by ADDING the frame yaw
        box = OrientedBox(center=(float(c[0]), float(c[1]), bb.center[2]),
                          size=bb.size, yaw=yaw + float(bb.yaw),
                          device_type=DeviceType.RACK,
                          meta={"grounded": True,
                                "n_pts": bb.meta.get("n_pts", 0)})
        boxes.append(box)
    if not boxes:
        print("[ground] no region survived the point-support guards")
        if out_dir:
            _save_grounded_fail_png(img, out_dir,
                                    "no region survived point-support guards")
        return False
    print(f"[ground] {len(rects)} VLM regions -> {len(boxes)} fitted boxes")
    # DEDUPLICATE: the VLM often outlines the SAME device more than
    # once (overlapping rects in one reply). Each rect fits its own
    # near-identical box with a DIFFERENT box_id, and the per-box local
    # refinement then renders mask_prompt_<id>_front.png per box --
    # one device, several duplicate renders (user report). Keep the
    # best-point-supported fit per IoU >= 0.5 cluster.
    dedup = []
    for b in sorted(boxes, key=lambda x: -int(x.meta.get("n_pts", 0))):
        if any(b.iou_2d(d) >= 0.5 for d in dedup):
            continue
        dedup.append(b)
    if len(dedup) < len(boxes):
        print(f"[ground] dropped {len(boxes) - len(dedup)} duplicate "
              f"box(es) (IoU >= 0.5 with a better-supported fit)")
    boxes = dedup
    # MERGE tightly-adjacent over-split pieces (user request): the VLM
    # sometimes outlines one physical structure as several tight rects;
    # each fits its own box and the seam never heals (stageC only
    # splits, never merges). Touching / point-bridged boxes merge into
    # a point-support-refitted union; the true splitting is the local
    # refine's job.
    boxes = _merge_adjacent_boxes(boxes, pts_fit, yaw)
    scene.boxes = boxes
    # result audit: the grounded.png shows the view's own raw VLM rects
    # (colored) plus the final fitted boxes (red) projected through the
    # same camera
    if out_dir:
        _save_grounded_png(img, cam, boxes, rects, out_dir,
                           fname="grounded.png")
    return True
