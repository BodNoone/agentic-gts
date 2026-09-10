"""True 3DGS rendering for VLM evidence images.

Backend chain (first available wins):
  1. gsplat.rendering.rasterization          (pip install gsplat)
  2. diff_gaussian_rasterization             (the official 3DGS repo's lib)
  3. None -> caller falls back to the matplotlib scatter render

Only needs to work where the pipeline actually runs (CUDA server); on
machines without torch it degrades gracefully to the old scatter view.
Pure numpy + PIL for everything except the two rasterizer calls.
"""
from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np

from agentic_gts.tools.gs_io import GaussianData

SH_C0 = 0.28209479177387814


# ---------------------------------------------------------------- cameras
@dataclass
class Cam:
    eye: np.ndarray      # camera position (world)
    target: np.ndarray    # look-at point (world)
    up: np.ndarray        # up hint (world)
    fovy_deg: float
    W: int
    H: int

    # ---- official-3DGS-convention view matrix: cam axes x=right, y=up, z=BACKWARD
    def view_official(self) -> np.ndarray:
        zax = self.eye - self.target
        zax = zax / (np.linalg.norm(zax) + 1e-12)
        xax = np.cross(self.up, zax)
        xax = xax / (np.linalg.norm(xax) + 1e-12)
        yax = np.cross(zax, xax)
        V = np.eye(4)
        V[:3, :3] = np.vstack([xax, yax, zax])
        V[:3, 3] = V[:3, :3] @ (-self.eye)
        return V

    # ---- standard CV view matrix: x=right, y=down, z=FORWARD (gsplat's layout)
    def view_cv(self) -> np.ndarray:
        D = np.diag([1.0, -1.0, -1.0, 1.0])
        return D @ self.view_official()

    def K(self) -> np.ndarray:
        fy = (self.H / 2.0) / math.tan(math.radians(self.fovy_deg) / 2.0)
        fx = fy  # square pixels
        return np.array([[fx, 0.0, self.W / 2.0],
                         [0.0, fy, self.H / 2.0],
                         [0.0, 0.0, 1.0]])

    # ---- pixel projection, standard CV convention (matches gsplat output)
    def project_cv(self, pts: np.ndarray) -> np.ndarray:
        """Nx3 world points -> Nx2 pixel coords (z-forward CV convention)."""
        h = np.hstack([pts, np.ones((len(pts), 1))])
        pc = h @ self.view_cv().T
        z = np.clip(pc[:, 2], 1e-6, None)
        K = self.K()
        return np.stack([K[0, 0] * pc[:, 0] / z + K[0, 2],
                         K[1, 1] * pc[:, 1] / z + K[1, 2]], axis=1)


def _bbox(pts: np.ndarray):
    lo, hi = pts.min(axis=0), pts.max(axis=0)
    return lo, hi


def _footprint(points: np.ndarray, boxes, pad_frac: float = 0.03):
    """(center, half_diag, loft, lohi) framing the DEVICE LAYOUT.

    Favours the boxes' footprint when available (that is what the god-view
    must show), falling back to the point-cloud bbox. The margin is a
    FRACTION of the layout size (pad_frac, default 3%) so the racks fill
    most of the frame instead of a small central patch.
    """
    if boxes:
        cs = np.vstack([b.corners_2d() for b in boxes]).astype(np.float64)
        lo = cs.min(axis=0)
        hi = cs.max(axis=0)
        z_top = max((np.asarray(b.center[2]) + b.size[2] / 2.0 for b in boxes),
                    default=float("nan"))
    else:
        lo = points[:, :2].min(axis=0)
        hi = points[:, :2].max(axis=0)
        z_top = float("nan")
    pad = pad_frac * float(np.linalg.norm(hi - lo) / 2.0)
    lo = lo - pad
    hi = hi + pad
    center = (lo + hi) / 2.0
    half_diag = float(np.linalg.norm(hi - lo) / 2.0)
    return center, half_diag, lo, hi, z_top


def make_godview_cam(points: np.ndarray, boxes=(), W: int = 1280, H: int = 1024,
                     elev_deg: float = 55.0, azim_deg: float = 45.0,
                     nadir: bool = False, cam_z: float | None = None) -> Cam:
    """Bird's-eye camera auto-fitted so the whole DEVICE LAYOUT is in frame.

    nadir=False (default): oblique view, eye raised by `elev_deg`/`azim_deg`.

    nadir=True: true top-down (straight down) camera. The framing footprint
    is the BOXES' footprint (not the raw point-cloud bbox, which walls and
    floor smear across the whole room) so the racks fill the frame instead
    of a small patch in the middle. The camera height is derived from that
    footprint and raised until it is framed -- so it sits well above the
    racks. (The eye may end up above the ceiling; the Z cut removes ceiling
    gaussians before rasterization, so they cannot reappear overhead.)
    """
    center, half_diag, lo, hi, z_top = _footprint(points, boxes)

    if nadir:
        # look straight down (-z), up hint = +y in world (screen-up = +y)
        z_ref = float(z_top) if np.isfinite(z_top) else float(points[:, 2].max())
        z_floor = float(points[:, 2].min())
        up = np.array([0.0, 1.0, 0.0])  # screen up aligned with world +y
        # Analytic first guess for the height, then nudge up until the whole
        # footprint (including its rack-top corners) projects inside the frame.
        fov_half = math.radians(60.0 / 2.0)     # fovy is the VERTICAL half-angle
        span_x = float(hi[0] - lo[0])
        span_y = float(hi[1] - lo[1])
        # horizontal half-angle at the same fovy: pixels are square, so
        # tan(fx) = tan(fy) * W/H  (wider frame -> wider horizontal FOV)
        fx_half = math.atan(math.tan(fov_half) * (W / H))
        # on-screen x <- world x needs fx_half; on-screen y <- world y needs fov_half
        need_h = max(span_x / 2.0 / math.tan(fx_half),
                     span_y / 2.0 / math.tan(fov_half))
        # Base height frames the footprint *exactly* at hf=1.0, then keep a
        # little extra so the rack-TOP corners (projected at z_ref, which
        # spread outward under perspective) stay inside the frame too. The
        # rack footprint sits inside the padded framing box, so exact-fit on
        # the padded box is guaranteed in-frame; the z_ref term is what
        # clears the outward-spreading rack tops.
        base_z = max(float(cam_z) if (cam_z is not None and np.isfinite(cam_z)) else 0.0,
                     need_h + z_ref * 1.25)
        # 8 framing corners at BOTH floor and rack-top heights: the 3D
        # wireframe's bottom ring sits at floor level, and under perspective
        # the (closer, lower) floor corners lean OUTWARD vs the top ring --
        # frame them too or the wireframe's lower edge clips the border.
        corners = np.array([[x, y, z] for x in (lo[0], hi[0])
                            for y in (lo[1], hi[1])
                            for z in (z_floor, z_ref)])
        for hf in (1.0, 1.02, 1.05, 1.08, 1.12, 1.18, 1.25, 1.35, 1.5):
            eye_z = base_z * hf
            c = Cam(eye=np.array([center[0], center[1], eye_z]),
                    target=np.array([center[0], center[1], z_floor]),
                    up=up, fovy_deg=60.0, W=W, H=H)
            pc = np.hstack([corners, np.ones((len(corners), 1))]) @ c.view_cv().T
            if not np.all(pc[:, 2] > 0.1):
                continue
            uv = c.project_cv(corners)
            if (uv[:, 0].min() > 0.005 * W and uv[:, 0].max() < 0.995 * W and
                    uv[:, 1].min() > 0.005 * H and uv[:, 1].max() < 0.995 * H):
                return c
        return Cam(eye=np.array([center[0], center[1], base_z * 1.0]),
                   target=np.array([center[0], center[1], z_floor]),
                   up=up, fovy_deg=60.0, W=W, H=H)

    el, az = math.radians(elev_deg), math.radians(azim_deg)
    up = np.array([0.0, 0.0, 1.0])
    cam = None
    z_floor = float(points[:, 2].min())
    # 8 corners of the framing footprint (floor + rack-top heights)
    zc = z_ref if np.isfinite(z_ref) else float(points[:, 2].max())
    corners = np.array([[x, y, z] for x in (lo[0], hi[0])
                        for y in (lo[1], hi[1]) for z in (z_floor, zc)])
    for dist in [(half_diag + 1.0) * f for f in (1.0, 1.2, 1.5, 1.8, 2.2, 2.8, 3.5, 4.5, 6.0, 8.0, 11.0)]:
        eye = center + dist * np.array(
            [math.cos(el) * math.cos(az), math.cos(el) * math.sin(az), math.sin(el)])
        c = Cam(eye=eye, target=np.array([center[0], center[1], z_floor]),
                up=up, fovy_deg=60.0, W=W, H=H)
        cam = cam or c
        pc = np.hstack([corners, np.ones((len(corners), 1))]) @ c.view_cv().T
        if not np.all(pc[:, 2] > 0.1):       # some corner behind the camera
            continue
        uv = c.project_cv(corners)
        if (uv[:, 0].min() > 0.03 * W and uv[:, 0].max() < 0.97 * W and
                uv[:, 1].min() > 0.03 * H and uv[:, 1].max() < 0.97 * H):
            cam = c
            break
    return cam


def make_local_cam(boxes, extent: float = 1.2, W: int = 768, H: int = 768,
                   elev_deg: float = 18.0, azim_deg: float = 0.0) -> Cam:
    """Camera for one box (or a pair): the fine-detail counterpart to the
    god-view's coarse positioning.

    Accepts a single OrientedBox or a LIST of boxes (e.g. the two faces of
    a merge-pair adjudication) and frames the UNION of all their 3D corners
    -- with a verify-and-back-off loop so nothing clips out of view.

    The camera looks at the box from its FRONT (perpendicular to the row
    direction), rotated around the box by `azim_deg` (0 = front face, 90 =
    side face), tilted down by `elev_deg`. A slight tilt shows the face
    detail (doors/panels/LED); a steep one (e.g. ~70) shows the top face
    with little foreshortening -- the row-direction thickness stays
    measurable in continuous rows. `up` stays world-vertical so the rack
    renders upright.

    W/H default 768: each tile of the three-view composite the VLM
    adjudicates on carries ~5cm-scale misfits (wireframe overhang); at
    448px a 1m-wide rack resolves to ~2px/cm which the VLM cannot read.
    """

    if hasattr(boxes, "center"):    # tolerate a single OrientedBox
        boxes = [boxes]
    ref = boxes[0]
    c = np.asarray(ref.center, dtype=float)
    yaw = float(ref.yaw)
    # horizontal viewing direction: box front (cross axis) rotated by azim
    az = math.radians(azim_deg)
    base = np.array([-math.sin(yaw), math.cos(yaw)])          # front (cross)
    rot = np.array([[math.cos(az), -math.sin(az)],
                    [math.sin(az), math.cos(az)]])
    horiz = rot @ base
    # all 3D corners of all boxes: the union that must stay in frame
    corners = np.vstack([_box_corners_3d(b) for b in boxes])
    fy = math.tan(math.radians(60.0 / 2.0))
    fx = fy * (W / H)
    spans = corners.max(axis=0) - corners.min(axis=0)
    # first-guess distance from the union's extent (plus margin), then
    # verify by projection and back off until every corner is in frame
    dist0 = max(spans[2] / 2.0 / fy, (spans[0] + extent) / 2.0 / fx)
    for f in (1.0, 1.1, 1.25, 1.4, 1.6, 1.9, 2.2, 2.6, 3.0, 3.5):
        dist = dist0 * f
        eye = c + np.array([horiz[0] * dist, horiz[1] * dist,
                            dist * math.tan(math.radians(elev_deg))])
        cam = Cam(eye=eye, target=c, up=np.array([0.0, 0.0, 1.0]),
                  fovy_deg=60.0, W=W, H=H)
        pc = np.hstack([corners, np.ones((len(corners), 1))]) @ cam.view_cv().T
        if not np.all(pc[:, 2] > 0.1):       # some corner behind the camera
            continue
        uv = cam.project_cv(corners)
        if (uv[:, 0].min() > 0.02 * W and uv[:, 0].max() < 0.98 * W and
                uv[:, 1].min() > 0.02 * H and uv[:, 1].max() < 0.98 * H):
            return cam
    return cam


# ---------------------------------------------------------------- rasterizers
def _near_boxes_mask(gs: GaussianData, boxes, margin: float = 0.6) -> np.ndarray:
    """Boolean mask: gaussians whose xy lies inside any box's OBB (inflated
    by `margin`). Used by the LOCAL render to hide unrelated structure --
    other racks in front, walls -- so nothing occludes the box being
    adjudicated. The margin keeps a ring of immediate context around the
    box (its own noisy gaussians bleeding past the faces, plus the closest
    neighbouring structure) while dropping everything the adjudication
    does not need to see.
    """
    xy = gs.means[:, :2]
    m = np.zeros(len(gs), dtype=bool)
    for b in boxes:
        yaw = float(b.yaw)
        c, s = math.cos(yaw), math.sin(yaw)
        d = xy - np.asarray(b.center, dtype=float)[:2]
        along = d @ np.array([c, s])
        cross = d @ np.array([-s, c])
        size = np.asarray(b.size, dtype=float)
        m |= (np.abs(along) < size[0] / 2.0 + margin) & \
             (np.abs(cross) < size[1] / 2.0 + margin)
    return m


def _prep(gs: GaussianData, cut_z: float, cut_z_low: float = float("-inf")):
    """Common tensor-ready numpy arrays (ceiling + floor cuts applied).

    cut_z: keep gaussians BELOW this (removes ceiling / overhead trays).
    cut_z_low: keep gaussians ABOVE this (removes the floor and below, whose
    texture / reflections occlude the rack footprint in a top-down view).
    """
    m = np.ones(len(gs), dtype=bool)
    if np.isfinite(cut_z):
        m &= gs.means[:, 2] < cut_z
    if np.isfinite(cut_z_low):
        m &= gs.means[:, 2] > cut_z_low
    means = gs.means[m].astype(np.float64)
    scales = np.exp(gs.log_scales[m].astype(np.float64))
    quats = gs.quats[m].astype(np.float64)
    qn = np.linalg.norm(quats, axis=1, keepdims=True)
    quats = quats / np.clip(qn, 1e-9, None)
    opac = 1.0 / (1.0 + np.exp(-gs.raw_opacity[m].astype(np.float64)))  # sigmoid
    rgb = np.clip(0.5 + SH_C0 * gs.f_dc[m].astype(np.float64), 0.0, 1.0)
    return means, quats, scales, opac, rgb


def _subset_gs(gs: GaussianData, mask: np.ndarray) -> GaussianData:
    """GaussianData restricted to the masked gaussians (views, no copy of
    the big arrays beyond the boolean index)."""
    import copy as _copy
    sub = _copy.copy(gs)     # shallow: reuse untouched fields
    sub.means = gs.means[mask]
    sub.log_scales = gs.log_scales[mask]
    sub.quats = gs.quats[mask]
    sub.raw_opacity = gs.raw_opacity[mask]
    sub.f_dc = gs.f_dc[mask]
    return sub


def _try_gsplat(means, quats, scales, opac, rgb, V_cv, K, W, H):
    import torch
    from gsplat.rendering import rasterization
    dev = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    t = lambda a: torch.tensor(a, dtype=torch.float32, device=dev)
    # gsplat's rasterization expects opacities shaped (N,) in current versions.
    # Squeeze to a 1-D opac so shape mismatches from older (N,1) conventions
    # don't silently break; rasterization broadcasts a 1-D opacity fine.
    op = t(opac).squeeze(-1)
    out = rasterization(
        t(means), t(quats), t(scales),
        op, t(rgb),
        viewmats=t(V_cv).unsqueeze(0), Ks=t(K).unsqueeze(0),
        width=W, height=H, render_mode="RGB",
    )
    img = out[0][0].detach().cpu().numpy()
    return np.clip(img[..., :3], 0.0, 1.0)


def _getPerspective(znear, zfar, fovy, W, H):
    tanFovY = math.tan(math.radians(fovy) / 2.0)
    tanFovX = tanFovY * W / H
    P = np.zeros((4, 4))
    P[0, 0] = 1.0 / tanFovX
    P[1, 1] = 1.0 / tanFovY
    P[2, 2] = zfar / (zfar - znear)
    P[3, 2] = -(zfar * znear) / (zfar - znear)
    P[2, 3] = 1.0
    return P


def _try_official(means, quats, scales, opac, rgb, cam: Cam):
    """diff_gaussian_rasterization (the official 3DGS repo's CUDA lib).

    Mirrors gaussian_renderer/render() from INRIA's repo: tensors are stored
    TRANSPOSED because the CUDA kernel reads them column-major.
    """
    import torch
    from diff_gaussian_rasterization import (GaussianRasterizationSettings,
                                              GaussianRasterizer)
    dev = torch.device("cuda")
    t = lambda a: torch.tensor(a, dtype=torch.float32, device=dev)
    V_off = cam.view_official()
    full = _getPerspective(0.01, 1e6, cam.fovy_deg, cam.W, cam.H) @ V_off
    settings = GaussianRasterizationSettings(
        image_height=cam.H, image_width=cam.W,
        tanfovx_y=math.tan(math.radians(cam.fovy_deg) / 2.0),
        tanfovx_x=math.tan(math.radians(cam.fovy_deg) / 2.0) * cam.W / cam.H,
        bg=torch.zeros(3, device=dev), scale_modifier=1.0,
        viewmatrix=t(V_off).T, projmatrix=t(full).T,
        sh_degree=0, campos=t(cam.eye), prefiltered=False, debug=False)
    rasterizer = GaussianRasterizer(raster_settings=settings)
    means_t, quats_t = t(means), t(quats)
    screenspace = torch.zeros_like(means_t[:, :3].repeat(1, 1),
                                   requires_grad=True, device=dev) + 0
    screenspace = torch.zeros((len(means), 3), dtype=torch.float32,
                              device=dev, requires_grad=True)
    with torch.enable_grad():
        img, radii, _ = rasterizer(
            means3D=means_t, means2D=screenspace, shs=None,
            colors_precomp=t(rgb), opacities=t(opac).unsqueeze(1),
            scales=t(scales), rotations=t(quats), cov3D_precomp=None)
    return np.clip(img.detach().cpu().numpy().T, 0.0, 1.0)


def rasterize_gs(gs: GaussianData, cam: Cam, cut_z: float = float("inf"),
                 cut_z_low: float = float("-inf"),
                 keep_mask: np.ndarray | None = None):
    """(H,W,3) float image or None if no CUDA rasterizer is available.

    keep_mask: optional boolean mask over gs (e.g. _near_boxes_mask) to
    render only a subset -- used by the local view to hide unrelated
    structure that would occlude the adjudicated box.
    """
    if keep_mask is not None:
        gs = _subset_gs(gs, keep_mask)
    means, quats, scales, opac, rgb = _prep(gs, cut_z, cut_z_low)
    if len(means) == 0:
        return None
    try:
        return _try_gsplat(means, quats, scales, opac, rgb,
                           cam.view_cv(), cam.K(), cam.W, cam.H)
    except ImportError:
        pass
    except Exception as e:
        print(f"[gs] gsplat rasterization failed ({type(e).__name__}: {e}) "
              f"-> trying official rasterizer")
    try:
        return _try_official(means, quats, scales, opac, rgb, cam)
    except ImportError:
        print("[gs] neither gsplat nor diff_gaussian_rasterization installed "
              "-> falling back to scatter render "
              "(pip install gsplat on the GPU server)")
    except Exception as e:
        print(f"[gs] official rasterizer failed too ({type(e).__name__}: {e}) "
              f"-> falling back to scatter render")
    return None


# ---------------------------------------------------------------- overlay
def _box_corners_3d(box) -> np.ndarray:
    """8x3 world corners of an OrientedBox (z-rotation only)."""
    l, w, h = (s / 2.0 for s in box.size)
    local = np.array([[x, y, z] for x in (-l, l) for y in (-w, w) for z in (-h, h)])
    c, s = math.cos(box.yaw), math.sin(box.yaw)
    rot = np.array([[c, -s, 0.0], [s, c, 0.0], [0.0, 0.0, 1.0]])
    return local @ rot.T + np.asarray(box.center, dtype=float)


def overlay_boxes(img: np.ndarray, boxes, cam: Cam,
                  mode: str = "footprint") -> np.ndarray:
    """Overlay numbered boxes on a rendered image.

    mode="footprint" (god-view): a top-down *discovery* view -- the VLM only
    needs to know WHERE devices are (2D), so we draw only the box's top-face
    rectangle (thin, confidence-coloured) plus a numbered chip; the gaussian
    render underneath stays readable.

    mode="wire3d" (local evidence): the camera looks at the box from its
    front at a slight tilt, so we draw the FULL 12-edge 3D wireframe. A
    lone top rectangle would float mid-air over an oblique render; the full
    wireframe hugs the rack's visible faces and shows the VLM exactly which
    volume the candidate box claims.

    mode="wire3d_axes" (fit refinement): wire3d PLUS two arrows on the
    box's TOP face -- green along the box's local +x (length) axis, blue
    along local +y (depth) -- so the VLM can SEE the box's orientation and
    propose yaw / size corrections relative to those axes.
    """
    from PIL import Image, ImageDraw
    pil = Image.fromarray((np.clip(img, 0, 1) * 255).astype(np.uint8))
    dr = ImageDraw.Draw(pil)

    def _arrow(dr, cam, p0_w, p1_w, color):
        """3D segment p0_w->p1_w drawn as a pixel arrow with a head."""
        p0 = cam.project_cv(np.asarray(p0_w, dtype=float)[None])[0]
        p1 = cam.project_cv(np.asarray(p1_w, dtype=float)[None])[0]
        dr.line([tuple(p0), tuple(p1)], fill=color, width=3)
        d = np.array([p1[0] - p0[0], p1[1] - p0[1]], dtype=float)
        n = float(np.linalg.norm(d))
        if n < 1e-6:
            return
        d /= n
        perp = np.array([-d[1], d[0]])
        tip = np.asarray(p1, dtype=float)
        for s in (1.0, -1.0):
            head = tip - 9.0 * d + 4.0 * s * perp
            dr.line([tuple(tip), tuple(head)], fill=color, width=3)

    for i, b in enumerate(boxes):
        cs = _box_corners_3d(b)
        color = (255, 60, 50) if getattr(b.confidence, "value", "") != "low" \
            else (255, 190, 40)
        if mode in ("wire3d", "wire3d_axes"):
            # corner ordering (x,y,z) in ((-l,l),(-w,w),(-h,h)):
            # idx = 4*xi + 2*yi + zi, so 0..7. Full 12-edge wireframe.
            uv = cam.project_cv(cs)
            top_ring = [uv[1], uv[3], uv[7], uv[5]]
            bot_ring = [uv[0], uv[2], uv[6], uv[4]]
            for ring in (top_ring, bot_ring):
                dr.line([tuple(p) for p in ring] + [tuple(ring[0])],
                        fill=color, width=2)
            for a, bidx in ((0, 1), (2, 3), (6, 7), (4, 5)):
                dr.line([tuple(uv[a]), tuple(uv[bidx])], fill=color, width=2)
            # numbered chip at the bottom ring's near corner
            cx, cy = uv[0]
            chip = str(i)
            wpx = dr.textlength(chip, font=None)
            dr.rectangle([cx - 3, cy - 9, cx + wpx + 5, cy + 5], fill=(0, 0, 0))
            dr.text((cx + 2, cy - 8), chip, fill=(255, 255, 255))
            if mode == "wire3d_axes":
                # local axis arrows on the top face: green = +x (length),
                # blue = +y (depth). Drawn slightly PAST the face so they
                # stay visible against the wireframe.
                c = np.asarray(b.center, dtype=float)
                h = b.size[2] / 2.0
                cyy, syy = math.cos(b.yaw), math.sin(b.yaw)
                xlen = b.size[0] / 2.0 + 0.12
                ylen = b.size[1] / 2.0 + 0.12
                x_tip = c + np.array([xlen * cyy, xlen * syy, h])
                y_tip = c + np.array([-ylen * syy, ylen * cyy, h])
                top_c = c + np.array([0.0, 0.0, h])
                _arrow(dr, cam, top_c, x_tip, (0, 255, 80))
                _arrow(dr, cam, top_c, y_tip, (80, 160, 255))
            continue
        # ---- footprint mode (god-view) ----
        # Project the TOP ring (z=+h) in polygon order. The god view is a
        # discovery view and the rack's visible top is what the VLM sees as
        # "where the device is"; projecting the bottom ring would be pulled
        # outward under perspective for off-centre racks (a tall rack at the
        # frame edge leans its base away), so the box would look misaligned
        # with the rendered rack. The top ring matches the visible footprint.
        # Corner ordering for (x,y,z) in ((-l,l),(-w,w),(-h,h)): top ring =
        # 1,3,7,5. (1=(-l,-w) 3=(-l,+w) 7=(+l,+w) 5=(+l,-w)) -> perimeter.
        uv = cam.project_cv([cs[1], cs[3], cs[7], cs[5]])
        # thin footprint rectangle
        dr.line([tuple(uv[0]), tuple(uv[1]), tuple(uv[2]), tuple(uv[3]),
                 tuple(uv[0])], fill=color, width=2)
        # numbered chip on the rectangle's top-left edge, off the rack body
        cx, cy = uv[0]
        chip = str(i)
        wpx = dr.textlength(chip, font=None)
        dr.rectangle([cx - 3, cy - 9, cx + wpx + 5, cy + 5], fill=(0, 0, 0))
        dr.text((cx + 2, cy - 8), chip, fill=(255, 255, 255))
    return np.asarray(pil).astype(np.float32) / 255.0


def render_gs_view(gs: GaussianData, boxes, cam: Cam,
                   cut_z: float = float("inf"),
                   cut_z_low: float = float("-inf"),
                   overlay: str = "footprint",
                   isolate_boxes: bool = False,
                   isolate_margin: float = 0.6):
    """Full render: gaussians + numbered box overlay. None if no backend.

    isolate_boxes: keep ONLY the gaussians near `boxes` (their inflated
    OBBs) -- for the local evidence view, so unrelated structure (other
    racks in front, walls) cannot occlude the box being adjudicated.
    """
    keep = _near_boxes_mask(gs, boxes, margin=isolate_margin) \
        if (isolate_boxes and boxes) else None
    img = rasterize_gs(gs, cam, cut_z=cut_z, cut_z_low=cut_z_low,
                       keep_mask=keep)
    if img is None:
        return None
    if boxes:
        img = overlay_boxes(img, boxes, cam, mode=overlay)
    return img


def png_bytes(img: np.ndarray) -> bytes:
    import io
    from PIL import Image
    buf = io.BytesIO()
    Image.fromarray((np.clip(img, 0, 1) * 255).astype(np.uint8)).save(
        buf, format="png")
    return buf.getvalue()


# ------------------------------------------------------- render quality
def train_view_trust(cam: Cam, centers: np.ndarray, dirs: np.ndarray,
                     pos_scale: float = 2.0) -> float:
    """How well a candidate render view is covered by the TRAINING cameras.

    3DGS quality is anisotropic around the training path: a view close
    to a training camera AND looking a similar direction renders sharp;
    the same view extrapolated away blurs and grows floaters. The
    image-based quality score measures the SYMPTOM (blur / speckle)
    AFTER rendering; this measures the CAUSE (distance to the trained
    ray distribution) BEFORE rendering, immune to floaters that happen
    to look sharp, and available even when the rasterizer's output is
    ambiguous.

    trust = max_i  exp(-|E - C_i| / pos_scale) * max(cos(L, d_i), 0)^2
    Position AND direction both matter: standing exactly on a training
    spot but looking 180 deg away extrapolates rays the field never saw;
    looking the right way from 5 m off the path does too.
    """
    centers = np.asarray(centers, dtype=np.float64)
    dirs = np.asarray(dirs, dtype=np.float64)
    E = np.asarray(cam.eye, dtype=np.float64)[:3]
    L = np.asarray(cam.target, dtype=np.float64)[:3] - E
    L = L / (np.linalg.norm(L) + 1e-12)
    d = np.linalg.norm(centers - E, axis=1)                 # (N,)
    look = dirs / (np.linalg.norm(dirs, axis=1, keepdims=True) + 1e-12)
    cosang = np.clip(look @ L, 0.0, 1.0)                    # (N,)
    trust = float(np.max(np.exp(-d / pos_scale) * cosang ** 2))
    return min(max(trust, 0.0), 1.0)


def view_quality(img: np.ndarray) -> dict:
    """No-reference quality score for ONE rendered view, in [0,1] higher =
    better. 3DGS quality is anisotropic: views near the training cameras
    are sharp, extrapolated ones blur and grow floaters. Before handing a
    view to the VLM we score the candidates and keep the best.

    Three orthogonal signals, all pure numpy:
      sharpness  -- variance of the image Laplacian. The typical 3DGS
                    out-of-distribution failure is blur (underconstrained
                    gaussians inflate), which kills high frequencies first.
      coverage   -- fraction of pixels with actual content (background is
                    black). Too little = the view barely rendered anything
                    (undertrained direction); ~everything = a wall or floor
                    slab occludes the whole frame.
      speckle    -- fraction of foreground pixels isolated from any
                    neighbour (4-connectivity on a coarse grid). Small
                    disconnected blobs are the classic floater signature.
    """
    g = np.asarray(img, dtype=np.float32)[..., :3]
    gray = g @ np.array([0.299, 0.587, 0.114], dtype=np.float32)
    step = max(1, min(gray.shape) // 256)
    if step > 1:
        gray = gray[::step, ::step]
    if gray.size < 16:
        return {"score": 0.0, "sharpness": 0.0, "coverage": 0.0,
                "speckle": 1.0}

    # sharpness: Laplacian variance (log-compressed to a usable range)
    lap = (gray[:-2, 1:-1] + gray[2:, 1:-1] + gray[1:-1, :-2]
           + gray[1:-1, 2:] - 4.0 * gray[1:-1, 1:-1])
    lap_var = float(lap.var()) if lap.size else 0.0
    sharp_n = float(np.clip(np.log10(1.0 + 100.0 * lap_var) / 2.0, 0.0, 1.0))

    # coverage: foreground band score
    fg = gray > 0.05
    cov = float(fg.mean())
    if cov <= 0.2:
        cov_score = max(cov / 0.2, 0.0)
    elif cov <= 0.9:
        cov_score = 1.0
    else:
        cov_score = max(0.0, 1.0 - (cov - 0.9) / 0.1 * 0.7)

    # speckle: isolated foreground pixels on the coarse grid
    n_fg = int(fg.sum())
    if n_fg == 0:
        speck_n = 1.0
    else:
        pad = np.pad(fg, 1)
        neigh = (pad[:-2, 1:-1].astype(np.int8) + pad[2:, 1:-1]
                 + pad[1:-1, :-2] + pad[1:-1, 2:])
        isolated = int(((fg) & (neigh == 0)).sum())
        speck_n = float(np.clip(isolated / (0.15 * n_fg), 0.0, 1.0))

    score = 0.55 * sharp_n + 0.25 * cov_score + 0.20 * (1.0 - speck_n)
    return {"score": round(score, 4), "sharpness": round(sharp_n, 4),
            "coverage": round(cov, 4), "speckle": round(speck_n, 4)}


def box_visibility(gs: GaussianData, boxes, cam: Cam,
                   keep_mask: np.ndarray | None = None,
                   cut_z: float = float("inf"),
                   max_gaussians: int = 60_000) -> float:
    """Fraction of box-surface sample points with a CLEAR sightline from
    the camera. [0,1], higher = the box is actually visible.

    Catches the occluded-view case that image-quality scores CANNOT: a box
    against a wall, or flush against a neighbouring rack, gets its front
    view filled by the wall / neighbour -- which renders SHARP and scores
    well while the adjudicated box is invisible. Occlusion is geometric
    and must be tested geometrically: for each sample point on the boxes'
    6 faces, cast the sightline from the camera and ask whether any other
    gaussian (an ellipsoid of radius ~1.5x its max scale) intersects the
    segment IN FRONT of the point.

    Excluded from occluders: the boxes' OWN splats (inside any box with a
    small margin -- the device surface itself must not self-occlude) and
    gaussians above cut_z (the ceiling cut already removes them from the
    actual render, so they must not count as blockers either).
    """
    m = keep_mask if keep_mask is not None else np.ones(len(gs), dtype=bool)
    means = np.asarray(gs.means)[m]
    radii = np.exp(np.asarray(gs.log_scales)[m]).max(axis=1) * 1.5
    if cut_z is not None and np.isfinite(cut_z):
        below = means[:, 2] < cut_z
        means, radii = means[below], radii[below]
    # exclude the device's own splats: inside any box (with margin)
    inside = np.zeros(len(means), dtype=bool)
    for b in boxes:
        inside |= b.contains(means, margin=0.05)
    means, radii = means[~inside], radii[~inside]
    if len(means) > max_gaussians:      # cap for memory/speed, deterministic
        sel = np.random.default_rng(0).choice(len(means), max_gaussians,
                                              replace=False)
        means, radii = means[sel], radii[sel]
    # sample points: 3x3 grid on each of the 6 faces of every box
    pts = []
    for b in boxes:
        l, w, h = (s / 2.0 for s in np.asarray(b.size, dtype=float))
        faces = []
        for u in np.linspace(-l, l, 3):
            for v in np.linspace(-w, w, 3):
                faces += [(u, v, h), (u, v, -h)]
        for u in np.linspace(-l, l, 3):
            for z in np.linspace(-h, h, 3):
                faces += [(u, w, z), (u, -w, z)]
        for v in np.linspace(-w, w, 3):
            for z in np.linspace(-h, h, 3):
                faces += [(l, v, z), (-l, v, z)]
        pts.append(b.local_to_world(np.asarray(faces)))
    P = np.vstack(pts)                          # (n,3)
    if len(means) == 0:
        return 1.0
    C = np.asarray(cam.eye, dtype=float)
    D = P - C
    L = np.linalg.norm(D, axis=1)                # (n,)
    U = D / np.clip(L[:, None], 1e-9, None)
    visible = np.ones(len(P), dtype=bool)
    step = 20_000
    for s in range(0, len(means), step):
        G = means[s:s + step]                    # (m,3)
        R = radii[s:s + step]                    # (m,)
        GC = G[:, None, :] - C                   # (m,n,3)
        t = np.einsum("mnc,nc->mn", GC, U)       # proj along sightline
        perp2 = np.einsum("mnc,mnc->mn", GC, GC) - t ** 2
        occl = (t > 0.05) & (t < L[None, :] - 0.15) & \
               (perp2 < R[:, None] ** 2)
        visible &= ~occl.any(axis=0)
    return float(visible.mean())


def camera_clearance(gs: GaussianData, boxes, cam: Cam,
                     keep_mask: np.ndarray | None = None,
                     cut_z: float = float("inf")) -> float:
    """Signed distance [m] from the camera eye to the nearest gaussian
    structure it would render (the occluder set: keep_mask, below cut_z,
    outside the adjudicated boxes). Negative = the eye is INSIDE a splat.

    Catches the 'sandwiched device' failure the quality/visibility scores
    only half-cover: a rack flush between two neighbours has its SIDE view
    camera travel along the row and end up embedded in the neighbouring
    rack -- the render is then a wall of huge near-camera splats (a blurry
    mess) regardless of how well that direction was trained. This is a
    property of the camera POSITION, so it must be tested before scoring:
    a colliding camera is pulled back along its sight axis (see
    _pullback_cam) instead of being silently scored and picked."""
    m = keep_mask if keep_mask is not None else np.ones(len(gs), dtype=bool)
    means = np.asarray(gs.means)[m]
    radii = np.exp(np.asarray(gs.log_scales)[m]).max(axis=1)
    if cut_z is not None and np.isfinite(cut_z):
        below = means[:, 2] < cut_z
        means, radii = means[below], radii[below]
    inside = np.zeros(len(means), dtype=bool)
    for b in boxes:
        inside |= b.contains(means, margin=0.05)
    means, radii = means[~inside], radii[~inside]
    if len(means) == 0:
        return float("inf")
    d = np.linalg.norm(means - np.asarray(cam.eye, dtype=float), axis=1)
    return float((d - radii).min())


def _pullback_cam(gs: GaussianData, boxes, cam: Cam,
                  keep_mask: np.ndarray | None, cut_z: float,
                  min_clear: float = 0.10):
    """Move a structure-embedded camera to a clear position. Returns
    (cam, clearance).

    Two escape moves, both preserving the camera's AZIMUTH (the slot's
    role: a side view stays a side view):
      1. scale the eye about the target along the sight ray -- escapes a
         finite structure the camera just clipped into (near row end,
         opposing rack across a narrow aisle);
      2. additionally RAISE the eye -- a sandwiched mid-row rack's side
         camera sits inside a CONTINUOUS row, and no amount of backing off
         along the row axis exits it; looking down the row from above the
         rack tops is the only clear sightline, and at 768px per tile the
         smaller/steeper box stays readable.
    If nothing clears, the least-embedded attempt is returned with its
    (negative) clearance, which disqualifies the candidate downstream and
    gates the verdict confidence."""
    c = np.asarray(cam.target, dtype=float)
    eye0 = np.asarray(cam.eye, dtype=float)

    def _mk(f, lift):
        eye = c + (eye0 - c) * f
        eye = eye + np.array([0.0, 0.0, lift])
        return Cam(eye=eye, target=cam.target, up=cam.up,
                   fovy_deg=cam.fovy_deg, W=cam.W, H=cam.H)

    best, best_clr = cam, camera_clearance(gs, boxes, cam, keep_mask, cut_z)
    if best_clr >= min_clear:
        return best, best_clr
    attempts = [(f, 0.0) for f in (1.35, 1.8, 2.4, 3.2)] + \
               [(f, lift) for f in (1.8, 2.4)
                for lift in (0.8, 1.6)]
    for f, lift in attempts:
        cand = _mk(f, lift)
        clr = camera_clearance(gs, boxes, cand, keep_mask, cut_z)
        if clr > best_clr:
            best, best_clr = cand, clr
        if clr >= min_clear:
            return cand, clr
    return best, best_clr


def render_slot_candidates(gs, boxes, cam_fn, candidates, cut_z,
                            overlay: str, iso_margin: float,
                            fallback_candidates=(),
                            train_views=None):
    """Render several (elev, azim) candidates for ONE view slot, score each
    on the RAW render (before the wireframe overlay -- drawn lines would
    pollute the sharpness/speckle metrics), and return the best.

    Four independent signals per candidate:
      quality     -- image-based (sharpness/coverage/speckle): is this view
                     well-trained? A view extrapolated away from the
                     training cameras blurs and grows floaters.
      visibility  -- geometry-based (box_visibility): is the box actually
                     VISIBLE, or does a wall / flush neighbour fill the
                     frame? A sharp wall still scores high on quality, so
                     only the sightline test catches it.
      clearance   -- geometry-based (camera_clearance): is the CAMERA
                     itself outside the structure? A sandwiched device's
                     side view embeds the camera in the neighbouring rack
                     (a blurry wall of near splats); the camera is pulled
                     back along its sight axis until it clears.
      train       -- pose-based (train_view_trust, from COLMAP training
                     poses): distance of the candidate view to the
                     TRAINED ray distribution. Known BEFORE rendering
                     (the cause of blur, not the symptom), so when the
                     trust is low the image quality is HALVED toward the
                     fallback tier: an extrapolated view that happens to
                     look sharp on no-reference metrics (floaters with
                     texture) must not win the slot. None (no poses
                     given) leaves the pure image score untouched.

    Selection: among candidates where the box is visible (visibility >=
    0.25) AND the camera is outside the structure (clearance >= 0 after
    the pullback) pick the highest image quality; if every candidate fails,
    keep the least-occluded one (its low visibility / negative clearance
    flows into the confidence gating downstream -- the verdict is then
    distrusted instead of silently judged on a wall of near splats).

    fallback_candidates: a SECOND tier of steep (~58 deg) elevations at
    the slot's azimuths, rendered ONLY when the primary tier comes out
    weak -- ineligible, or eligible but blurry (score < 0.35: the
    pullback rescue tends to leave the camera far away in an extrapolated
    direction). The narrow-aisle case: two facing full-height rows with
    a ~0.5 m aisle between them have NO horizontal sightline to either
    row's inner face (the sightline must pass over a flush row of the
    same height), so the horizontal camera ends up far away, lifted, and
    blurry; the steep near camera looks down over the aisle instead --
    top face + aisle context, close enough to stay sharp. Normal-width
    aisles never trigger the fallback, so the front role (door/panel
    detail) is preserved where it is actually achievable.

    train_views: optional (centers Nx3, dirs Nx3) from read_colmap_views.

    cam_fn(elev_deg, azim_deg) -> Cam. The isolation mask is computed ONCE
    for all candidates (it depends only on the boxes, not the camera).
    Returns (img, quality, (elev, azim)) or (None, None, None).
    """
    keep = _near_boxes_mask(gs, boxes, margin=iso_margin) if boxes else None

    def _score_one(elev, azim):
        cam = cam_fn(elev, azim)
        cam, clr = _pullback_cam(gs, boxes, cam, keep, cut_z)
        vis = box_visibility(gs, boxes, cam, keep_mask=keep, cut_z=cut_z)
        img = rasterize_gs(gs, cam, cut_z=cut_z, keep_mask=keep)
        if img is None:
            return None
        q = view_quality(img)
        if train_views is not None:
            trust = train_view_trust(cam, train_views[0], train_views[1])
            q["train"] = round(trust, 3)
            # low pose-trust halves the image score toward the steep
            # fallback: extrapolation is the CAUSE of blur, and the
            # no-reference metrics can be fooled by textured floaters
            q["score"] = round(q["score"] * (0.5 + 0.5 * trust), 4)
        q["visibility"] = round(vis, 4)
        q["clearance"] = round(clr, 3) if np.isfinite(clr) else None
        if boxes:
            img = overlay_boxes(img, boxes, cam, mode=overlay)
        return (img, q, (elev, azim))

    def _select(pool):
        eligible = [s for s in pool
                    if s[1]["visibility"] >= 0.25
                    and (s[1]["clearance"] is None or s[1]["clearance"] >= 0.0)]
        if eligible:
            return max(eligible, key=lambda s: s[1]["score"]), True
        return max(pool, key=lambda s: s[1]["visibility"]), False

    scored = [s for s in (_score_one(e, a) for e, a in candidates) if s]
    if not scored:
        return (None, None, None)
    best, was_eligible = _select(scored)
    if fallback_candidates and (not was_eligible
                                 or best[1]["score"] < 0.35):
        for e, a in fallback_candidates:
            s = _score_one(e, a)
            if s is not None:
                scored.append(s)
        best, _ = _select(scored)
    return best
