"""Local VLM point prompts -> SAM mask -> 3D OBB refinement.

Contract:
  * Qwen3-VL returns point prompts in its native relative 0..1000 grid.
  * This module converts them ONCE to image pixels for SAM. They are not
    normalized again.
  * SAM creates pixel-accurate candidate masks.
  * The mask is lifted to the local 3DGS geometry by projecting Gaussian
    centers through the exact render camera. Geometry, not the VLM, measures
    the final metric OBB.

SAM is optional. The implementation supports Meta SAM2 and legacy Segment
Anything when installed. If no SAM backend/checkpoint is configured, local
mask refinement is skipped conservatively; it never damages existing boxes.
"""
from __future__ import annotations

import json
import math
import os
import re
from dataclasses import dataclass

import numpy as np

from agentic_gts.core.models import BoxSource, Confidence, OrientedBox, Scene


@dataclass
class PointGroup:
    positive_norm: list[tuple[float, float]]
    negative_norm: list[tuple[float, float]]
    hypothesis: str = "rack"
    confidence: float = 0.5

    def pixel_prompts(self, W: int, H: int) -> tuple[np.ndarray, np.ndarray]:
        pts = self.positive_norm + self.negative_norm
        labels = [1] * len(self.positive_norm) + [0] * len(self.negative_norm)
        if not pts:
            return np.empty((0, 2), dtype=np.float32), np.empty(0, dtype=np.int32)
        xy = np.asarray([[x / 1000.0 * (W - 1), y / 1000.0 * (H - 1)]
                         for x, y in pts], dtype=np.float32)
        return xy, np.asarray(labels, dtype=np.int32)


def parse_point_groups(text: str) -> list[PointGroup]:
    """Parse Qwen point-grounding JSON (relative 0..1000 coordinates).

    Accepted shapes:
      {"candidate_groups": [{"positive": [[x,y]], "negative": ...}]}
      [{"positive_points": ..., "negative_points": ...}]
      {"positive": ..., "negative": ...}
    Points may be [x,y] or {"x": x, "y": y}. Values in 0..1 are accepted
    as normalized fractions and converted to 0..1000 for robustness.
    """
    if not text:
        return []
    data = None
    # Qwen may prepend reasoning. Scan every object/array opener with the
    # standard JSON decoder and keep the LAST complete value (the final answer).
    dec = json.JSONDecoder()
    values = []
    for m in re.finditer(r"[\[{]", text):
        try:
            value, _ = dec.raw_decode(text[m.start():])
            values.append(value)
        except json.JSONDecodeError:
            continue
    if values:
        # Prefer a complete top-level structure carrying prompt-group keys;
        # later values may just be nested [x,y] arrays encountered by the scan.
        for value in reversed(values):
            if (isinstance(value, dict) and
                    any(k in value for k in ("candidate_groups", "groups",
                                              "positive", "positive_points"))):
                data = value
                break
            if (isinstance(value, list) and value and
                    all(isinstance(x, dict) for x in value)):
                data = value
                break
    if data is None:
        return []
    if isinstance(data, dict):
        items = data.get("candidate_groups") or data.get("groups") or [data]
    elif isinstance(data, list):
        items = data
    else:
        return []

    def _points(raw):
        out = []
        for p in raw or []:
            try:
                if isinstance(p, dict):
                    x, y = float(p["x"]), float(p["y"])
                else:
                    x, y = float(p[0]), float(p[1])
            except (KeyError, IndexError, TypeError, ValueError):
                continue
            if max(abs(x), abs(y)) <= 1.0:
                x, y = x * 1000.0, y * 1000.0
            if 0 <= x <= 1000 and 0 <= y <= 1000:
                out.append((x, y))
        return out

    groups = []
    for item in items if isinstance(items, list) else []:
        if not isinstance(item, dict):
            continue
        pos = _points(item.get("positive") or item.get("positive_points"))
        neg = _points(item.get("negative") or item.get("negative_points"))
        if not pos:
            continue
        try:
            conf = float(item.get("confidence", 0.5))
        except (TypeError, ValueError):
            conf = 0.5
        groups.append(PointGroup(pos, neg, str(item.get("hypothesis", "rack")),
                                 min(max(conf, 0.0), 1.0)))
    return groups


class SamPredictorAdapter:
    """Lazy SAM2/SAM1 predictor with a uniform point-prompt interface."""

    def __init__(self, checkpoint: str | None = None,
                 model_cfg: str | None = None):
        self.checkpoint = checkpoint or os.environ.get("SAM_CHECKPOINT")
        self.model_cfg = model_cfg or os.environ.get("SAM_MODEL_CFG")
        self._predictor = None

    @property
    def available(self) -> bool:
        return bool(self.checkpoint)

    def _load(self):
        if self._predictor is not None:
            return
        if not self.checkpoint:
            raise RuntimeError("SAM_CHECKPOINT not configured")
        # SAM2 first (recommended). SAM_MODEL_CFG is required by build_sam2.
        try:
            from sam2.build_sam import build_sam2
            from sam2.sam2_image_predictor import SAM2ImagePredictor
            if not self.model_cfg:
                raise RuntimeError("SAM_MODEL_CFG is required for SAM2")
            model = build_sam2(self.model_cfg, self.checkpoint)
            try:
                import torch
                if torch.cuda.is_available():
                    model = model.to("cuda")
            except Exception:
                pass
            self._predictor = SAM2ImagePredictor(model)
            return
        except ImportError:
            pass
        # Legacy SAM fallback.
        try:
            from segment_anything import sam_model_registry, SamPredictor
            model_type = os.environ.get("SAM_MODEL_TYPE", "vit_h")
            model = sam_model_registry[model_type](checkpoint=self.checkpoint)
            try:
                import torch
                if torch.cuda.is_available():
                    model = model.to("cuda")
            except Exception:
                pass
            self._predictor = SamPredictor(model)
            return
        except ImportError as e:
            raise RuntimeError("install SAM2 or segment-anything") from e

    def predict(self, image: np.ndarray, point_coords: np.ndarray,
                point_labels: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        self._load()
        u8 = (np.clip(image[..., :3], 0, 1) * 255).astype(np.uint8)
        self._predictor.set_image(u8)
        masks, scores, _ = self._predictor.predict(
            point_coords=point_coords, point_labels=point_labels,
            multimask_output=True)
        return np.asarray(masks, dtype=bool), np.asarray(scores, dtype=float)


def _subset_or_none(gs, keep: np.ndarray):
    """gs restricted to the keep mask, or None when nothing survives."""
    if keep is None or not keep.any():
        return None
    from agentic_gts.output.gs_render import _subset_gs
    return _subset_gs(gs, keep)


def _view_frustum_mask(gs, box: OrientedBox, cam, near_margin: float = 0.5,
                       ring_margin: float = 0.25,
                       behind_slack: float = 0.4) -> np.ndarray:
    """Occlusion-free isolation for the front/side local views.

    The box OBB ring (near_margin) is ALWAYS kept -- that is the device
    plus its own noisy gaussians bleeding past the faces. Everything
    else must EARN its way into the render by being INSIDE the camera's
    view frustum of the box: project the box's 3D corners, take the
    pixel-space hull of the box, and keep only gaussians whose
    projection falls inside it (inflated by ring_margin in world
    terms). This drops whatever stands between the camera and the box
    (a facing row across the aisle, a near wall) -- the 'garbage
    angles' the user reported: the rack rendered as its visible SIDE
    because the front was occluded -- while keeping a thin ring of
    context (the user asked for a little surrounding environment).

    behind_slack: gaussians up to this far BEHIND the box's far face
    are kept even when the frustum test rejects them (their projections
    fall inside the box hull anyway; the slack covers the row's own
    back band and splat radii).
    """
    pts = np.asarray(gs.means, dtype=float)
    m = np.zeros(len(pts), dtype=bool)

    # 1. the always-keep ring: box OBB + near_margin
    corners3 = np.asarray(box.corners_2d())
    c = np.asarray(box.center, dtype=float)
    yaw = float(box.yaw)
    ca, sa = math.cos(yaw), math.sin(yaw)
    d = pts[:, :2] - c[:2]
    along = d @ np.array([ca, sa])
    cross = d @ np.array([-sa, ca])
    size = np.asarray(box.size, dtype=float)
    m |= ((np.abs(along) < size[0] / 2.0 + near_margin) &
          (np.abs(cross) < size[1] / 2.0 + near_margin) &
          (pts[:, 2] > c[2] - size[2] / 2.0 - 0.3) &
          (pts[:, 2] < c[2] + size[2] / 2.0 + 0.3))

    # 2. frustum-of-the-box test for everything else
    uv = cam.project_cv(pts)
    H, W = cam.H, cam.W
    in_img = ((uv[:, 0] >= -0.02 * W) & (uv[:, 0] <= 1.02 * W) &
              (uv[:, 1] >= -0.02 * H) & (uv[:, 1] <= 1.02 * H))
    # the box's 8 real 3D corners (4 footprint corners at both z faces)
    z_lo, z_hi = c[2] - size[2] / 2.0, c[2] + size[2] / 2.0
    box_pts = np.vstack([np.column_stack([corners3, np.full(4, z_lo)]),
                         np.column_stack([corners3, np.full(4, z_hi)])])
    box_uv = cam.project_cv(box_pts)
    # pixel hull bounds (conservative AABB of the box's own corners),
    # inflated by ring_margin expressed in pixels (~size[1] maps to the
    # box's face height in the frame; scale the margin the same way)
    face_px = max(float(np.ptp(box_uv[:, 0])), float(np.ptp(box_uv[:, 1])),
                  1.0)
    pad_px = ring_margin / max(size[1], 0.1) * face_px
    u0 = float(box_uv[:, 0].min()) - pad_px
    u1 = float(box_uv[:, 0].max()) + pad_px
    v0 = float(box_uv[:, 1].min()) - pad_px
    v1 = float(box_uv[:, 1].max()) + pad_px
    in_box_hull = ((uv[:, 0] >= u0) & (uv[:, 0] <= u1) &
                   (uv[:, 1] >= v0) & (uv[:, 1] <= v1))
    # depth: keep a bit behind the box (its back band + splat radius),
    # but anything CLOSER to the camera than the box's near face minus
    # slack is an occluder candidate -- only frustum survivors pass
    V = cam.view_cv()
    pc = np.hstack([pts, np.ones((len(pts), 1))]) @ V.T
    depth = pc[:, 2]
    bpc = np.hstack([box_pts, np.ones((8, 1))]) @ V.T
    box_near = float(bpc[:, 2].min()) - behind_slack
    box_far = float(bpc[:, 2].max()) + behind_slack
    near_ok = (depth >= box_near) | (m & (depth > 0.0))
    far_ok = depth <= box_far + 0.3
    m |= (in_img & in_box_hull & near_ok & far_ok & (depth > 0.0))
    return m


def render_local_views(scene: Scene, box: OrientedBox,
                       out_dir: str | None = None) -> list[dict]:
    """Render front + side local views, isolated to the box neighborhood."""
    gs_ply = scene.meta.get("gs_ply")
    if not gs_ply:
        return []
    from agentic_gts.tools.gs_io import read_gaussian_ply
    from agentic_gts.output.gs_render import (make_local_cam, rasterize_gs,
                                              render_gs_view, png_bytes)
    gs = read_gaussian_ply(gs_ply)
    cut_hi = box.center[2] + box.size[2] / 2.0 + 0.10
    cut_lo = box.center[2] - box.size[2] / 2.0 + 0.05
    # FRONT must face the device's front face: the camera direction is
    # the box's cross (short) axis. When the fitted box has its length
    # on the cross axis (size[0] < size[1] -- a yaw that runs along the
    # row, or a 90-deg flip from PCA), the view at azim=0 looks along
    # the LONG edge instead: swap the two so 'front' is always
    # perpendicular to the long edge, i.e. facing the device's face
    # (user report: many front views were the visible SIDE).
    azim_front = 0.0 if box.size[0] >= box.size[1] else 90.0
    # user-directed view pair: FRONT (the face: doors, panels) +
    # OBLIQUE (elevated ~58 deg off vertical, 30 deg around the box:
    # top face + two faces + context -- the footprint and the device
    # outline are both measurable, which a pure side view along the
    # row cannot show).
    slots = (("front", 18.0, azim_front),
             ("oblique", 58.0, azim_front + 30.0))
    out = []
    for name, elev, azim in slots:
        cam = make_local_cam([box], extent=1.4, W=768, H=768,
                             elev_deg=elev, azim_deg=azim)
        # occlusion-free isolation (user report: the box's front was
        # occluded by structure between the camera and the box, so the
        # view degenerated to a visible side / garbage angle). The
        # frustum mask keeps the box's ring plus a thin context band,
        # and drops every gaussian standing in front of the box.
        keep = _view_frustum_mask(gs, box, cam)
        sub = _subset_or_none(gs, keep)
        raw = rasterize_gs(sub, cam, cut_z=cut_hi, cut_z_low=cut_lo) \
            if sub is not None else None
        if raw is None:
            raw = render_gs_view(gs, [box], cam, cut_z=cut_hi,
                                 cut_z_low=cut_lo, overlay=None,
                                 isolate_boxes=True, isolate_margin=0.8)
            if raw is None:
                continue
            prompt_img = render_gs_view(
                gs, [box], cam, cut_z=cut_hi, cut_z_low=cut_lo,
                overlay="wire3d", isolate_boxes=True, isolate_margin=0.8)
        else:
            prompt_img = render_gs_view(
                sub, [box], cam, cut_z=cut_hi, cut_z_low=cut_lo,
                overlay="wire3d")
        path = None
        prompt_path = None
        if out_dir:
            os.makedirs(out_dir, exist_ok=True)
            path = os.path.join(out_dir, f"mask_{box.box_id}_{name}.png")
            with open(path, "wb") as f:
                f.write(png_bytes(raw))
            if prompt_img is not None:
                prompt_path = os.path.join(
                    out_dir, f"mask_prompt_{box.box_id}_{name}.png")
                with open(prompt_path, "wb") as f:
                    f.write(png_bytes(prompt_img))
        out.append({"name": name, "image": raw,
                    "prompt_image": prompt_img if prompt_img is not None else raw,
                    "cam": cam, "path": path, "prompt_path": prompt_path})
    return out


def _mask_to_points(scene: Scene, box: OrientedBox, mask: np.ndarray, cam,
                    margin: float = 0.8) -> np.ndarray:
    """Lift mask to visible local 3DGS centers with a small z-buffer.

    Selecting every center whose projection lands in the mask also selects
    surfaces hidden behind the visible rack, inflating the fitted OBB. Keep
    only points close to the nearest projected depth in each pixel.
    """
    pts = np.asarray(scene.points, dtype=float)
    region = box.contains(pts, margin=margin)
    pts = pts[region]
    if not len(pts):
        return pts
    uv = cam.project_cv(pts)
    x = np.rint(uv[:, 0]).astype(int)
    y = np.rint(uv[:, 1]).astype(int)
    h = np.hstack([pts, np.ones((len(pts), 1))])
    pc = h @ cam.view_cv().T
    depth = pc[:, 2]
    valid = ((x >= 0) & (x < mask.shape[1]) &
             (y >= 0) & (y < mask.shape[0]) & (depth > 0.05))
    keep = np.zeros(len(pts), dtype=bool)
    ids = np.where(valid)[0]
    if not len(ids):
        return pts[:0]
    pix = y[ids] * mask.shape[1] + x[ids]
    zbuf = np.full(mask.shape[0] * mask.shape[1], np.inf, dtype=float)
    np.minimum.at(zbuf, pix, depth[ids])
    visible = depth[ids] <= zbuf[pix] + 0.12
    selected = ids[visible & mask[y[ids], x[ids]]]
    keep[selected] = True
    return pts[keep]


def fit_mask_points(points: np.ndarray, old: OrientedBox) -> OrientedBox | None:
    """Fit yaw + footprint + height from SAM-selected 3D points."""
    if len(points) < 40:
        return None
    xy = points[:, :2]
    cxy = np.median(xy, axis=0)
    xc = xy - cxy
    cov = xc.T @ xc / max(len(xc) - 1, 1)
    vals, vecs = np.linalg.eigh(cov)
    v = vecs[:, int(np.argmax(vals))]
    pca_yaw = math.atan2(float(v[1]), float(v[0]))
    # A side-view mask often has more depth than rack width, so raw PCA may
    # pick local y and rotate the rack by 90 degrees. The existing rough box
    # reliably tells us which eigen-axis is the row/length axis. Choose
    # between the PCA axis and its perpendicular by nearest angular distance.
    choices = [pca_yaw + k * math.pi / 2 for k in range(4)]
    yaw = min(choices, key=lambda a: abs(math.atan2(
        math.sin(a - old.yaw), math.cos(a - old.yaw))))
    # resolve 180-degree symmetry to the representation nearest old.yaw
    while yaw - old.yaw > math.pi / 2:
        yaw -= math.pi
    while yaw - old.yaw < -math.pi / 2:
        yaw += math.pi
    axis = np.array([math.cos(yaw), math.sin(yaw)])
    cross = np.array([-math.sin(yaw), math.cos(yaw)])
    a, d = xy @ axis, xy @ cross
    a0, a1 = np.percentile(a, [1.0, 99.0])
    d0, d1 = np.percentile(d, [1.0, 99.0])
    z0, z1 = np.percentile(points[:, 2], [1.0, 99.0])
    L, W, H = float(a1 - a0), float(d1 - d0), float(z1 - z0)
    if L < 0.15 or W < 0.10 or H < 0.30:
        return None
    centre2 = axis * ((a0 + a1) / 2.0) + cross * ((d0 + d1) / 2.0)
    return OrientedBox(center=(float(centre2[0]), float(centre2[1]),
                               float((z0 + z1) / 2.0)),
                       size=(L, W, H), yaw=yaw,
                       box_id=old.box_id, device_type=old.device_type,
                       source=BoxSource.AGENT_FIX,
                       confidence=Confidence.MID, row_id=old.row_id,
                       meta={**old.meta, "sam_refined": True})


def score_candidate(mask_score: float, points: np.ndarray,
                    new: OrientedBox, old: OrientedBox) -> float:
    """SAM quality + 3D support + conservative change score."""
    support = min(len(points) / 300.0, 1.0)
    iou = old.iou_2d(new)
    yaw_delta = abs(math.atan2(math.sin(new.yaw - old.yaw),
                               math.cos(new.yaw - old.yaw)))
    yaw_score = max(0.0, 1.0 - yaw_delta / math.radians(45))
    return 0.45 * float(mask_score) + 0.25 * support + 0.20 * iou + 0.10 * yaw_score


def confirm_device_type(judge, box: OrientedBox, views: list,
                        ) -> dict | None:
    """Type-level guard for ONE box: is the wrapped object a server rack?

    The grounding guards only reject hallucinated EMPTY regions; a real
    structure mislabelled a rack (pillar / UPS / AC / wall) passes them
    all. This asks the VLM one yes/no question on the front local view
    (wireframe overlay shows which object is meant). `views` comes from
    the caller's render_local_views call (shared with refine_box --
    one render, two questions).

    Returns None when there is no signal (no views, or the judge is mock
    -- never penalise for missing evidence); else {"is_rack": bool,
    "confidence": float}. The CALLER decides policy; the standing
    contract is mark-LOW + human review, never deletion.
    """
    if getattr(judge, "backend", "mock") == "mock":
        return None
    front = next((v for v in views if v["name"] == "front"), None)
    if front is None:
        return None
    verdict = judge.adjudicate_rack_confirm(
        front["prompt_image"], box,
        png_path=front["prompt_path"] or front["path"])
    p = verdict.params or {}
    if "is_rack" not in p:
        return None
    return {"is_rack": bool(p["is_rack"]),
            "confidence": float(p.get("confidence",
                                      verdict.confidence or 0.5))}


def _save_sam_debug(view: dict | None, coords, labels, mask, pts3,
                    box, fitted, out_dir: str, tag: str) -> None:
    """Composite debug render for ONE SAM candidate (user request):
      panel 1: the view image with the VLM's prompt points drawn
               (green = positive, red = negative)
      panel 2: the same image with the SAM mask overlaid (cyan tint +
               solid edge)
      panel 3: the back-projected 3D points top-down (z colored), the
               OLD box (dashed blue) and the FITTED box (solid red)
    view=None renders panel 3 only (the multi-view union candidate has
    no single owning view). Never raises: debug output must not break
    the refinement."""
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        from PIL import Image, ImageDraw

        # ---- panel 3 (always available): back-projected points ----
        fig = plt.figure(figsize=(4.8, 4.8), dpi=160)
        ax = fig.add_axes([0.04, 0.04, 0.92, 0.92])
        if pts3 is not None and len(pts3):
            ax.scatter(pts3[:, 0], pts3[:, 1], s=2, c=pts3[:, 2],
                       cmap="viridis", vmin=0.0)
        for b, style, color, label in ((box, "--", "deepskyblue", "old"),
                                       (fitted, "-", "red", "fitted")):
            if b is None:
                continue
            cs = np.asarray(b.corners_2d())
            ax.plot(np.append(cs[:, 0], cs[0, 0]),
                    np.append(cs[:, 1], cs[0, 1]), style, color=color,
                    lw=2, label=label)
        ax.set_aspect("equal")
        ax.set_xticks([])
        ax.set_yticks([])
        if fitted is not None or box is not None:
            ax.legend(loc="upper right", fontsize=6)
        fig.canvas.draw()
        p3 = Image.fromarray(
            np.asarray(fig.canvas.buffer_rgba())[..., :3])
        plt.close(fig)

        panels = [p3]
        titles = ["3. back-projected pts (top-down)"]

        if view is not None and view.get("image") is not None:
            img = np.asarray(view["image"], dtype=np.float32)
            img = (np.clip(img, 0, 1) * 255).astype(np.uint8)[..., :3]
            H, W = img.shape[:2]
            # ---- panel 1: prompt points on the clean view ----
            p1 = Image.fromarray(img.copy())
            d1 = ImageDraw.Draw(p1)
            for (x, y), lab in zip(coords or [], labels or []):
                r = max(6, W // 128)
                color = (0, 255, 0) if lab > 0 else (255, 40, 40)
                d1.ellipse((x - r, y - r, x + r, y + r),
                           fill=color, outline=(255, 255, 255), width=2)
            # ---- panel 2: mask overlay on the same view ----
            p2 = Image.fromarray(img.copy())
            if mask is not None:
                m = np.asarray(mask, dtype=bool)
                if m.shape == (H, W):
                    # semi-transparent cyan fill
                    alpha = (m.astype(np.uint8) * 90)
                    tint = np.zeros((H, W, 3), dtype=np.uint8)
                    tint[..., 0] = 0
                    tint[..., 1] = 220
                    tint[..., 2] = 255
                    a3 = alpha[..., None]
                    p2 = Image.fromarray(
                        (np.asarray(p2) * (255 - a3) // 255
                         + tint * a3 // 255).astype(np.uint8))
                    # solid edge: mask minus its erosion (no scipy needed)
                    er = m.copy()
                    er[1:] &= m[:-1]
                    er[:-1] &= m[1:]
                    er[:, 1:] &= m[:, :-1]
                    er[:, :-1] &= m[:, 1:]
                    edge = m & ~er
                    d2 = ImageDraw.Draw(p2)
                    ys, xs = np.nonzero(edge)
                    for xx, yy in zip(xs.tolist(), ys.tolist()):
                        d2.point((xx, yy), fill=(0, 220, 255))
            panels = [p1, p2, p3]
            titles = ["1. VLM prompt pts (+=green -=red)",
                      "2. SAM mask", "3. back-projected pts (top-down)"]

        # ---- tile horizontally with label strips ----
        lab_h = 20
        pad = 6
        Wt = sum(p.width for p in panels) + pad * (len(panels) + 1)
        Ht = max(p.height for p in panels) + lab_h + pad
        comp = Image.new("RGB", (Wt, Ht), (0, 0, 0))
        dd = ImageDraw.Draw(comp)
        x = pad
        for p, t in zip(panels, titles):
            comp.paste(p, (x, lab_h + pad))
            dd.text((x + 4, 4), t, fill=(255, 255, 255))
            x += p.width + pad
        os.makedirs(out_dir, exist_ok=True)
        comp.save(os.path.join(out_dir, f"sam_debug_{tag}.png"))
    except Exception as e:
        print(f"[mask-refine] debug render {tag} failed "
              f"({type(e).__name__}: {e})")


def refine_box(scene: Scene, box: OrientedBox, judge, sam: SamPredictorAdapter,
               out_dir: str | None = None,
               views: list | None = None) -> tuple[OrientedBox | None, dict]:
    """Run Qwen point grounding + SAM + 3D fitting for one box."""
    if views is None:
        views = render_local_views(scene, box, out_dir)
    audit = {"box_id": box.box_id, "views": [], "accepted": False}
    if not views or not sam.available:
        audit["reason"] = "no local GS views or SAM checkpoint"
        return None, audit
    candidates = []
    best_points_per_view = []
    for view in views:
        verdict = judge.adjudicate_sam_points(
            view["prompt_image"], box, view["name"],
            png_path=view["prompt_path"] or view["path"])
        groups = verdict.params.get("groups", []) if verdict.params else []
        va = {"view": view["name"], "image": view["path"],
              "answer": verdict.raw or verdict.detail, "groups": groups}
        view_candidates = []
        for gi, g in enumerate(groups):
            group = PointGroup(g["positive"], g.get("negative", []),
                               g.get("hypothesis", "rack"),
                               float(g.get("confidence", 0.5)))
            coords, labels = group.pixel_prompts(view["image"].shape[1],
                                                  view["image"].shape[0])
            if not len(coords):
                continue
            masks, scores = sam.predict(view["image"], coords, labels)
            for mi, (mask, ms) in enumerate(zip(masks, scores)):
                pts3 = _mask_to_points(scene, box, mask, view["cam"])
                fitted = fit_mask_points(pts3, box)
                if out_dir:
                    # debug composite: prompt points + mask + lifted pts
                    _save_sam_debug(view, coords, labels, mask, pts3, box,
                                    fitted, out_dir,
                                    f"{box.box_id}_{view['name']}_g{gi}_m{mi}")
                if fitted is None:
                    continue
                s = score_candidate(float(ms), pts3, fitted, box)
                candidates.append((s, fitted, view["name"], len(pts3), float(ms)))
                view_candidates.append((s, pts3, float(ms), gi, mi))
        if view_candidates:
            view_candidates.sort(key=lambda x: x[0], reverse=True)
            best_points_per_view.append(view_candidates[0])
        if out_dir:
            try:
                with open(os.path.join(
                        out_dir, f"sam_points_{box.box_id}_{view['name']}.json"),
                          "w", encoding="utf-8") as f:
                    json.dump(va, f, ensure_ascii=False, indent=2)
            except OSError:
                pass
        audit["views"].append(va)
    # Multi-view union candidate: front provides width/height, side provides
    # depth. It is usually more complete than either visible surface alone.
    if len(best_points_per_view) >= 2:
        union_pts = np.vstack([x[1] for x in best_points_per_view])
        fitted = fit_mask_points(union_pts, box)
        if out_dir:
            # union debug: panel 3 only (no single owning view)
            _save_sam_debug(None, None, None, None, union_pts, box, fitted,
                            out_dir, f"{box.box_id}_union")
        if fitted is not None:
            ms = float(np.mean([x[2] for x in best_points_per_view]))
            s = score_candidate(ms, union_pts, fitted, box) + 0.08
            candidates.append((min(s, 1.0), fitted, "front+side",
                               len(union_pts), ms))
    if not candidates:
        audit["reason"] = "no SAM mask yielded a valid 3D box"
        return None, audit
    candidates.sort(key=lambda x: x[0], reverse=True)
    score, best, view_name, npts, mask_score = candidates[0]
    audit.update({"accepted": score >= 0.45, "score": round(score, 4),
                  "view": view_name, "points": npts,
                  "sam_score": round(mask_score, 4),
                  "box": best.to_dict()})
    return (best if score >= 0.45 else None), audit
