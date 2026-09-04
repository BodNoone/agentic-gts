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
        # 4 footprint corners at rack-top height (worst case for overhang)
        corners = np.array([[x, y, z_ref] for x in (lo[0] , hi[0])
                            for y in (lo[1], hi[1])])
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


def make_local_cam(box, extent: float = 1.2, W: int = 448, H: int = 448,
                   elev_deg: float = 38.0) -> Cam:
    """Oblique camera for one box so the VLM sees the rack's side + top.

    A pure top-down view only shows the rack top and loses the
    height/door/side detail that distinguishes a rack from clutter. Here the
    camera sits up and to one side, looking down and IN toward the box. The
    horizontal direction is aligned with the box's yaw (its long axis), so we
    look along the row -- the info-rich face. `up` is world-vertical so the
    rack stays upright in the image.
    """
    c = np.asarray(box.center, dtype=float)
    top = c[2] + box.size[2] / 2.0
    # aim at the box's upper-mid so the whole rack is in view
    aim = np.array([c[0], c[1], c[2] + box.size[2] * 0.35])
    # horizontal offset along the box's long axis (its yaw direction)
    hdir = np.array([math.cos(box.yaw), math.sin(box.yaw), 0.0])
    horiz = np.asarray(box.size[0]) * 1.0 + extent
    eye = aim + hdir * horiz + np.array([0.0, 0.0, horiz * math.tan(math.radians(elev_deg)) * 1.6])
    return Cam(eye=eye, target=aim, up=np.array([0.0, 0.0, 1.0]),
               fovy_deg=60.0, W=W, H=H)


# ---------------------------------------------------------------- rasterizers
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
                 cut_z_low: float = float("-inf")):
    """(H,W,3) float image or None if no CUDA rasterizer is available."""
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


def overlay_boxes(img: np.ndarray, boxes, cam: Cam) -> np.ndarray:
    """Overlay numbered boxes on a rendered image.

    God-view is a top-down *discovery* view: the VLM only needs to know WHERE
    devices are (2D), not the 3D orientation/height. We therefore draw only
    the box's footprint rectangle (thin, semi-transparent, confidence-coloured)
    plus a small high-contrast numbered chip, so the gaussian render underneath
    stays readable. The chip is placed on the rectangle's edge (not its centre)
    so it never hides the rack body.
    """
    from PIL import Image, ImageDraw
    pil = Image.fromarray((np.clip(img, 0, 1) * 255).astype(np.uint8))
    dr = ImageDraw.Draw(pil)
    for i, b in enumerate(boxes):
        cs = _box_corners_3d(b)
        # Project the BOTTOM ring (z=-h) in polygon order so the rectangle is
        # a proper closed quad, not two crossed triangles. Corner ordering
        # for (x,y,z) in ((-l,l),(-w,w),(-h,h)): bottom ring = 0,2,6,4
        # (0=(-l,-w) 2=(-l,+w) 6=(+l,+w) 4=(+l,-w)) -> that is the perimeter.
        uv = cam.project_cv([cs[0], cs[2], cs[6], cs[4]])
        color = (255, 60, 50) if getattr(b.confidence, "value", "") != "low" \
            else (255, 190, 40)
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
                   cut_z_low: float = float("-inf")):
    """Full render: gaussians + numbered box overlay. None if no backend."""
    img = rasterize_gs(gs, cam, cut_z=cut_z, cut_z_low=cut_z_low)
    if img is None:
        return None
    if boxes:
        img = overlay_boxes(img, boxes, cam)
    return img


def png_bytes(img: np.ndarray) -> bytes:
    import io
    from PIL import Image
    buf = io.BytesIO()
    Image.fromarray((np.clip(img, 0, 1) * 255).astype(np.uint8)).save(
        buf, format="png")
    return buf.getvalue()
