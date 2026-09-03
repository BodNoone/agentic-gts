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


def make_godview_cam(points: np.ndarray, boxes=(), W: int = 1280, H: int = 1024,
                     elev_deg: float = 55.0, azim_deg: float = 45.0,
                     nadir: bool = False, cam_z: float | None = None) -> Cam:
    """Bird's-eye camera auto-fitted so the whole scene is in frame.

    nadir=False (default): oblique view, eye raised by `elev_deg`/`azim_deg`.

    nadir=True: true top-down (straight down) camera. `cam_z` is the camera
    height, typically just above the tallest box top but STILL BELOW the
    ceiling (cut_z), so overhead structure sits above the camera and is never
    projected -- it cannot occlude the racks. The camera looks straight down
    at the scene center; the view's up direction is aligned with the layout
    yaw so the rows run horizontally/vertically on screen.
    """
    lo, hi = _bbox(points)
    center = (lo + hi) / 2.0
    r = float(np.linalg.norm(hi[:2] - lo[:2]) / 2.0) + 1.0

    if nadir:
        # look straight down (-z), up hint = +y in world (screen-up = +y)
        # Camera height is driven by the FOOTPRINT size, not by box tops:
        # a god-view must overlook the whole room layout, so it sits well
        # above every structure. (The eye may end up above the ceiling --
        # that's fine, ceiling gaussians are removed by the Z cut before
        # rasterization, so they can never reappear overhead.)
        half_diag = float(np.linalg.norm(hi[:2] - lo[:2]) / 2.0)
        eye_z = 0.0 if (cam_z is None or not np.isfinite(cam_z)) else float(cam_z)
        # scale the base height with the room so a big room -> high camera
        eye_z = max(eye_z, half_diag * 1.15 + float(hi[2]))
        up = np.array([0.0, 1.0, 0.0])  # screen up aligned with world +y
        base = Cam(eye=np.array([center[0], center[1], eye_z]),
                   target=np.array([center[0], center[1], lo[2]]),
                   up=up, fovy_deg=60.0, W=W, H=H)
        corners = np.array([[x, y, hi[2]] for x in (lo[0], hi[0])
                            for y in (lo[1], hi[1])])
        # raise the camera until the whole footprint is inside the FOV
        for hf in (1.0, 1.15, 1.3, 1.5, 1.75, 2.0, 2.5, 3.0, 4.0, 5.5, 8.0):
            c = Cam(eye=np.array([center[0], center[1], eye_z * hf]),
                    target=np.array([center[0], center[1], lo[2]]),
                    up=up, fovy_deg=60.0, W=W, H=H)
            pc = np.hstack([corners, np.ones((len(corners), 1))]) @ c.view_cv().T
            if not np.all(pc[:, 2] > 0.1):
                continue
            uv = c.project_cv(corners)
            if (uv[:, 0].min() > 0.03 * W and uv[:, 0].max() < 0.97 * W and
                    uv[:, 1].min() > 0.03 * H and uv[:, 1].max() < 0.97 * H):
                return c
        return base

    el, az = math.radians(elev_deg), math.radians(azim_deg)
    up = np.array([0.0, 0.0, 1.0])
    cam = None
    # all 8 corners of the scene bbox must end up inside the frame
    corners = np.array([[x, y, z] for x in (lo[0], hi[0])
                        for y in (lo[1], hi[1]) for z in (lo[2], hi[2])])
    for dist in [r * f for f in (1.0, 1.2, 1.5, 1.8, 2.2, 2.8, 3.5, 4.5, 6.0, 8.0, 11.0)]:
        eye = center + dist * np.array(
            [math.cos(el) * math.cos(az), math.cos(el) * math.sin(az), math.sin(el)])
        c = Cam(eye=eye, target=center, up=up, fovy_deg=60.0, W=W, H=H)
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


def make_local_cam(box, extent: float = 1.0, W: int = 448, H: int = 448) -> Cam:
    """Nadir (straight-down) camera over one box, up-aligned with the box yaw."""
    c = np.asarray(box.center, dtype=float)
    top = c[2] + box.size[2] / 2.0
    eye = np.array([c[0], c[1], top + 2.0 * extent])
    up = np.array([math.cos(box.yaw), math.sin(box.yaw), 0.0])
    return Cam(eye=eye, target=np.array([c[0], c[1], 0.0]), up=up,
               fovy_deg=60.0, W=W, H=H)


# ---------------------------------------------------------------- rasterizers
def _prep(gs: GaussianData, cut_z: float):
    """Common tensor-ready numpy arrays (ceiling cut applied)."""
    m = np.isfinite(cut_z)
    mask = gs.means[:, 2] < cut_z if m else np.ones(len(gs), dtype=bool)
    means = gs.means[mask].astype(np.float64)
    scales = np.exp(gs.log_scales[mask].astype(np.float64))
    quats = gs.quats[mask].astype(np.float64)
    qn = np.linalg.norm(quats, axis=1, keepdims=True)
    quats = quats / np.clip(qn, 1e-9, None)
    opac = 1.0 / (1.0 + np.exp(-gs.raw_opacity[mask].astype(np.float64)))  # sigmoid
    rgb = np.clip(0.5 + SH_C0 * gs.f_dc[mask].astype(np.float64), 0.0, 1.0)
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


def rasterize_gs(gs: GaussianData, cam: Cam, cut_z: float = float("inf")):
    """(H,W,3) float image or None if no CUDA rasterizer is available."""
    means, quats, scales, opac, rgb = _prep(gs, cut_z)
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
    """Draw numbered box wireframes onto a rendered image (PIL, in-place copy)."""
    from PIL import Image, ImageDraw
    pil = Image.fromarray((img * 255).astype(np.uint8))
    dr = ImageDraw.Draw(pil)
    for i, b in enumerate(boxes):
        corners = _box_corners_3d(b)
        uv = cam.project_cv(corners)
        color = (255, 60, 50) if getattr(b.confidence, "value", "") != "low" \
            else (255, 190, 40)
        # local corner order: x(-,+), y(-,+), z(-,+) -> bottom ring 0,2,6,4 / top 1,3,7,5
        edges = [(0, 1), (0, 2), (0, 4), (1, 3), (1, 5), (2, 3), (2, 6),
                 (3, 7), (4, 5), (4, 6), (5, 7), (6, 7)]
        for a, b_ in edges:
            dr.line([tuple(uv[a]), tuple(uv[b_])], fill=color,
                    width=3 if a in (0, 2, 4, 6) else 2)
        c_uv = cam.project_cv(np.asarray(b.center, dtype=float)[None])[0]
        dr.text((c_uv[0] - 4, c_uv[1] - 6), str(i), fill=color)
    return np.asarray(pil).astype(np.float32) / 255.0


def render_gs_view(gs: GaussianData, boxes, cam: Cam,
                   cut_z: float = float("inf")):
    """Full render: gaussians + numbered box overlay. None if no backend."""
    img = rasterize_gs(gs, cam, cut_z=cut_z)
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
