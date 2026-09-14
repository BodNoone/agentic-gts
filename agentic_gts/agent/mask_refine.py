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

    @staticmethod
    def _build_sam2_model(model_cfg: str, checkpoint: str):
        """build_sam2, tolerant of a model_cfg that is a FILE PATH.

        build_sam2 feeds the cfg to hydra's compose(config_name=...), which
        only searches the sam2 package's own configs dir and interprets the
        name as a path RELATIVE to it -- an absolute path like
        /home/bod/code/sam2.1_hiera_b+.yaml loses its leading slash and
        becomes 'home/bod/code/...' (MissingConfigException). When the
        cfg is an existing file, register ITS directory as the hydra
        search path and pass only the basename; anything else (e.g.
        'sam2.1_hiera_b+.yaml', the package-relative name) goes through
        unchanged.
        """
        from sam2.build_sam import build_sam2
        if os.path.isfile(model_cfg):
            import hydra
            from hydra.core.global_hydra import GlobalHydra
            gh = GlobalHydra.instance()
            if gh.is_initialized():
                gh.clear()   # sam2's __init__ already bound sam2.configs
            cfg_dir = os.path.dirname(os.path.abspath(model_cfg))
            with hydra.initialize_config_dir(config_dir=cfg_dir,
                                             version_base="1.2"):
                return build_sam2(os.path.basename(model_cfg), checkpoint)
        return build_sam2(model_cfg, checkpoint)

    def _load(self):
        if self._predictor is not None:
            return
        if not self.checkpoint:
            raise RuntimeError("SAM_CHECKPOINT not configured")
        # SAM2 first (recommended). SAM_MODEL_CFG is required by build_sam2.
        try:
            from sam2.sam2_image_predictor import SAM2ImagePredictor
            if not self.model_cfg:
                raise RuntimeError("SAM_MODEL_CFG is required for SAM2")
            model = self._build_sam2_model(self.model_cfg, self.checkpoint)
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


def _open_side(gs, box: OrientedBox, reach: float = 3.0):
    """The box's open side (the aisle) and the clear corridor width.

    A rack in a row has an aisle on one face and the back gap (wall /
    the next row's back, typically 0.3-0.8m) on the other; a camera on
    the closed side photographs the rack's back squeezed against
    structure. Compare the free corridor on each side of the box -- in
    the device-height band and the row strip -- and return the open
    direction (unit world 2D vector) plus the corridor width in metres
    (capped at `reach`).
    """
    pts = np.asarray(gs.means, dtype=float)
    c = np.asarray(box.center, dtype=float)
    yaw = float(box.yaw)
    axis = np.array([math.cos(yaw), math.sin(yaw)])       # local x
    cross = np.array([-math.sin(yaw), math.cos(yaw)])     # local y
    size = np.asarray(box.size, dtype=float)
    # the FACE axis is perpendicular to the long edge (same convention as
    # the azim_front swap in render_local_views)
    if size[0] >= size[1]:
        face_axis, face_half, long_half = cross, size[1] / 2.0, size[0] / 2.0
        u, v = cross, axis          # u: face axis, v: row axis
    else:
        face_axis, face_half, long_half = axis, size[0] / 2.0, size[1] / 2.0
        u, v = axis, cross
    d = pts[:, :2] - c[:2]
    du = d @ u
    dv = d @ v
    z = pts[:, 2]
    z_top = c[2] + size[2] / 2.0
    band = (z > 0.25) & (z < max(min(z_top - 0.2, 2.0), 0.5))
    best_vec, best_corridor = face_axis.copy(), -1.0
    for s in (1.0, -1.0):
        beyond = du * s - face_half        # distance past the s-side face
        m = (np.abs(dv) < long_half + 1.0) & (beyond > 0.10) & \
            (beyond < reach) & band
        corridor = float(beyond[m].min()) if m.any() else reach
        if corridor > best_corridor:
            best_corridor, best_vec = corridor, s * face_axis
    return best_vec, min(best_corridor, reach)


def _box_only_mask(gs, box: OrientedBox, pad: float = 0.15) -> np.ndarray:
    """Boolean mask over gs: True only for gaussians INSIDE the box's OBB
    (plus `pad` metres of slack, since the fitted OBB clips a few cm off
    the device's own face gaussians).

    The local views exist to show the VLM and SAM exactly ONE device.
    Keeping the rest of the scene (the earlier 'normal aisle photo'
    attempt) re-introduced haze whenever the camera stood inside
    structure: big low-opacity training floaters fog the whole frame
    from any position. Hiding everything outside the box removes both
    the occluders AND the fog source in one rule -- what renders is
    exactly the device under adjudication, on a clean background.
    """
    return box.contains(np.asarray(gs.means, dtype=float), margin=pad)


def render_local_views(scene: Scene, box: OrientedBox,
                       out_dir: str | None = None) -> list[dict]:
    """Render front + diagonal local views: ONE device, isolated.

    The views feed the VLM and SAM, which only need the target box --
    everything else in the scene (the facing row the camera may stand
    inside, floor floaters, training haze) is noise to them. Two
    placement rules produce the clean pair:
      * the camera stands on the box's OPEN side (the aisle), picked by
        comparing free corridor width on either side -- the alternative
        (whichever way local +y happens to point) photographs the
        rack's back against the wall half the time;
      * the eye stays INSIDE that corridor (standoff mode) and widens
        its lens to frame the box instead of backing off;
      * every gaussian OUTSIDE the box's OBB (plus a small slack) is
        hidden from the render: no occluders, and no fog from structure
        the camera happened to end up inside (the earlier 'keep the
        environment' attempt still hazed out whenever the eye stood in
        a big low-opacity floater).
    """
    gs_ply = scene.meta.get("gs_ply")
    if not gs_ply:
        return []
    from agentic_gts.tools.gs_io import read_gaussian_ply
    from agentic_gts.output.gs_render import (make_local_cam, rasterize_gs,
                                              render_gs_view, png_bytes)
    gs = read_gaussian_ply(gs_ply)
    # camera side: the open corridor (aisle), not whichever way local +y
    # points. FRONT must additionally face the big face: perpendicular to
    # the long edge (a yaw running along the row, or a 90-deg flip from
    # PCA, otherwise makes azim=0 look along the LONG edge -- the visible
    # SIDE, an earlier user report).
    open_vec, corridor = _open_side(gs, box)
    yaw = float(box.yaw)
    cross_dir = np.array([-math.sin(yaw), math.cos(yaw)])  # local +y
    axis_dir = np.array([math.cos(yaw), math.sin(yaw)])    # local +x
    if box.size[0] >= box.size[1]:
        azim_front, face_dir = 0.0, cross_dir
    else:
        azim_front, face_dir = 90.0, axis_dir
    if float(open_vec @ face_dir) < 0.0:
        azim_front += 180.0            # the aisle is on the other side
    # view pair, both at GROUND level (elev 18 deg, rack height -- no
    # top-down component: the local views must show the device's
    # vertical surfaces, which the ground-level 3DGS training observed
    # well): FRONT (the face: doors, panels) + the DIAGONAL halfway
    # between front and side (front+45 deg: corner view showing two
    # adjacent faces at once -- the device's 3D extent along BOTH axes
    # is measurable, which neither the face-on nor the pure side view
    # alone can give).
    slots = (("front", 18.0, azim_front),
             ("oblique", 18.0, azim_front + 45.0))
    # standoff: ~80% into the corridor, never further than 2.2m; the
    # camera widens its lens to frame, it does not back off
    standoff = float(np.clip(0.8 * corridor, 0.6, 2.2))
    out = []
    for name, elev, azim in slots:
        cam = make_local_cam([box], W=768, H=768, elev_deg=elev,
                             azim_deg=azim, standoff=standoff)
        # render ONLY the device: every gaussian outside the box's OBB
        # (plus slack) is hidden -- occluders and fog sources alike
        sub = _subset_or_none(gs, _box_only_mask(gs, box))
        raw = rasterize_gs(sub, cam) if sub is not None else None
        if raw is None:
            raw = render_gs_view(gs, [box], cam, overlay=None,
                                 isolate_boxes=True, isolate_margin=0.8)
            if raw is None:
                continue
            prompt_img = render_gs_view(
                gs, [box], cam, overlay="wire3d", isolate_boxes=True,
                isolate_margin=0.8)
        else:
            prompt_img = render_gs_view(
                sub, [box], cam, overlay="wire3d")
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
    """Type-level guard for ONE box: is the wrapped object DC equipment
    (server rack / IT cabinet / air-conditioning unit)?

    The grounding guards only reject hallucinated EMPTY regions; a real
    structure mislabelled equipment (pillar / UPS / wall) passes them
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


def _overlay_mask(img_u8: np.ndarray, m: np.ndarray,
                  alpha: int = 115) -> np.ndarray:
    """Semi-transparent cyan fill over the RGB image: a proper
    rgb+mask overlay. NB: promote to int32 BEFORE multiplying --
    uint8 * uint8 wraps modulo 256 (150*165 -> 174), which made the
    masked region dark garbage with only the solid edge line
    surviving: the 'contour drawing' look the user reported."""
    alpha = int(alpha)
    tint = np.zeros(img_u8.shape, dtype=np.int32)
    tint[..., 1] = 220
    tint[..., 2] = 255
    a3 = m.astype(np.int32)[..., None] * alpha
    base = np.asarray(img_u8, dtype=np.int32)
    return np.clip(base * (255 - a3) // 255 + tint * a3 // 255,
                   0, 255).astype(np.uint8)


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
            # NB: coords/labels are numpy arrays -- `coords or []` would
            # evaluate the array's truth value (ValueError: ambiguous).
            for (x, y), lab in zip(coords if coords is not None else (),
                                   labels if labels is not None else ()):
                r = max(6, W // 128)
                color = (0, 255, 0) if lab > 0 else (255, 40, 40)
                d1.ellipse((x - r, y - r, x + r, y + r),
                           fill=color, outline=(255, 255, 255), width=2)
            # ---- panel 2: mask overlay on the same view ----
            p2 = Image.fromarray(img.copy())
            if mask is not None:
                m = np.asarray(mask, dtype=bool)
                if m.ndim == 3:            # SAM2 returns (C, H, W)
                    m = m.reshape(-1, H, W)[0]
                if m.shape == (H, W):
                    p2 = Image.fromarray(_overlay_mask(img, m))
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


def _augment_spread(view_img: np.ndarray, coords: np.ndarray,
                    labels: np.ndarray, min_points: int = 6,
                    min_spread: float = 0.45, max_add: int = 4):
    """Guard against clustered VLM point prompts (user report: points
    bunched on one door/panel make SAM segment a LOCAL part of the rack).

    When the positive points span less than `min_spread` of the image
    diagonal (or there are fewer than `min_points`), add points sampled
    from the device's interior pixels: the local views render ONLY the
    box's own gaussians, so every non-background pixel belongs to the
    target. Candidates are picked farthest-point style -- each new point
    maximizes its distance to the points already in the set -- so the
    additions spread the coverage instead of re-clustering.

    Returns (coords, labels) with the added positives appended.
    """
    coords = np.asarray(coords, dtype=np.float64).reshape(-1, 2)
    labels = np.asarray(labels).reshape(-1)
    if not len(coords):
        return coords, labels
    pos = coords[labels > 0]
    H, W = view_img.shape[:2]
    diag = math.hypot(H, W)

    def _spread(ps):
        return float(np.linalg.norm(ps.max(axis=0) - ps.min(axis=0))) \
            if len(ps) else 0.0

    if (len(pos) >= min_points and
            _spread(pos) >= min_spread * diag):
        return coords, labels

    # device foreground: non-background pixels (local views are box-only
    # renders, so fg == the target device), eroded so sampled points sit
    # safely inside the silhouette, away from boundaries
    img = np.asarray(view_img, dtype=np.float32)
    if img.ndim == 2:
        img = img[..., None]
    gray = img[..., :3].mean(axis=2)
    if float(gray.max()) > 1.5:            # uint8-scale input
        gray = gray / 255.0
    fg = gray > 0.08
    if not fg.any():
        return coords, labels
    inside = fg.copy()
    for dy in (-2, 0, 2):
        for dx in (-2, 0, 2):
            inside &= np.roll(np.roll(fg, dy, axis=0), dx, axis=1)
    ys, xs = np.nonzero(inside if inside.sum() > 50 else fg)
    cands = np.column_stack([xs, ys]).astype(np.float64)

    pts = [p for p in pos]
    for _ in range(max_add):
        if len(pts) >= min_points and _spread(np.asarray(pts)) >= \
                min_spread * diag:
            break
        d = np.min(np.linalg.norm(
            cands[:, None, :] - np.asarray(pts)[None, :, :], axis=2),
            axis=1) if pts else \
            np.linalg.norm(cands - cands.mean(axis=0), axis=1)
        pts.append(cands[int(np.argmax(d))])
    if not pts:
        return coords, labels
    new_pos = np.asarray(pts, dtype=np.float64)
    return (np.vstack([coords, new_pos[len(pos):]]),
            np.concatenate([labels,
                            np.ones(len(pts) - len(pos), dtype=labels.dtype)]))


def _calibrate_coord_scale(img: np.ndarray, coords: np.ndarray,
                           labels: np.ndarray) -> np.ndarray:
    """Re-interpret VLM point coords if they are ABSOLUTE PIXELS, not the
    0-1000 grid (user report: points mostly off the device, pulled toward
    one corner).

    pixel_prompts() divides by 1000 -- Qwen3-VL's native relative grid.
    But a backend answering in ABSOLUTE PIXELS instead (Qwen2.5-VL's
    convention: coords on the resized input image) then has every point
    shrunk to ~0.77x of its intended position on a 768px view, biased
    toward the top-left corner. Two facts disambiguate:
      * a raw value beyond the image size cannot be a pixel -> the grid
        interpretation is certain, keep it;
      * otherwise, the device IS the foreground (box-only local views
        render it on black): the interpretation landing more POSITIVE
        points on it wins, and only a CLEAR win (>= 0.25) re-maps -- a
        tie keeps the grid (current behaviour, no regression for
        compliant Qwen3-VL servers).
    """
    if not len(coords) or not (labels > 0).any():
        return coords
    H, W = img.shape[:2]
    raw = coords / np.array([W - 1, H - 1], dtype=np.float64) * 1000.0
    if raw.max() > max(W, H) + 2:
        return coords                  # beyond pixel range: grid, for sure
    fg = img[..., :3].mean(axis=2) > 0.08
    if not fg.any():
        return coords
    pos = raw[labels > 0]

    def _on_fg(pts):
        x = np.clip(np.rint(pts[:, 0]), 0, W - 1).astype(int)
        y = np.clip(np.rint(pts[:, 1]), 0, H - 1).astype(int)
        return float(fg[y, x].mean())

    as_grid = _on_fg(pos / 1000.0 * np.array([W - 1, H - 1]))
    as_pix = _on_fg(np.clip(pos, 0, [W - 1, H - 1]))
    if as_pix - as_grid < 0.25:
        return coords
    print("[mask-refine] VLM points look like ABSOLUTE PIXELS, not the "
          "0-1000 grid; re-mapped (check which model the server runs)")
    return np.clip(raw, 0, [W - 1, H - 1]).astype(np.float32)


def _pull_points_inward(view_img: np.ndarray, coords: np.ndarray,
                        labels: np.ndarray,
                        margin_frac: float = 0.06) -> tuple[np.ndarray,
                                                            np.ndarray]:
    """Pull POSITIVE prompts off the target's boundary (user report:
    points sitting on the device's edge hit attached cables / conduit /
    ladders, and SAM then segments those connected things in too).

    The local views render ONLY the box's own gaussians, so the
    foreground silhouette IS the device. Erode it by `margin_frac` of
    the image's shorter side and snap every positive that falls outside
    the eroded interior to its nearest interior pixel. Negatives are
    left alone (they are SUPPOSED to sit on the surroundings). Points
    already deep inside pass through unchanged.
    """
    coords = np.asarray(coords, dtype=np.float64).reshape(-1, 2)
    labels = np.asarray(labels).reshape(-1)
    if not len(coords) or not (labels > 0).any():
        return coords, labels
    img = np.asarray(view_img, dtype=np.float32)
    if img.ndim == 2:
        img = img[..., None]
    gray = img[..., :3].mean(axis=2)
    if float(gray.max()) > 1.5:            # uint8-scale input
        gray = gray / 255.0
    fg = gray > 0.08
    if not fg.any():
        return coords, labels
    H, W = fg.shape
    r = max(2, int(margin_frac * min(H, W)))
    er = fg.copy()
    for _ in range(r):
        er = er & np.roll(er, 1, axis=0) & np.roll(er, -1, axis=0) \
               & np.roll(er, 1, axis=1) & np.roll(er, -1, axis=1)
        if er.sum() < 20:                  # thin device: stop eroding
            break
    if er.sum() < 20:
        er = fg                            # degenerate: no interior left
    if er.all():
        return coords, labels
    ys, xs = np.nonzero(er)
    inside = np.column_stack([xs, ys]).astype(np.float64)
    out = coords.copy()
    for i in np.nonzero(labels > 0)[0]:
        iy, ix = int(round(coords[i, 1])), int(round(coords[i, 0]))
        if (0 <= iy < H and 0 <= ix < W) and er[iy, ix]:
            continue
        j = int(np.argmin(np.linalg.norm(inside - coords[i], axis=1)))
        out[i] = inside[j]
    return out, labels


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
        # CLEAN image to the VLM: the wireframe overlay (prompt_image)
        # is the Stage-A box, which is often oversized/misplaced -- the
        # VLM anchors its points on the frame and lands them off the
        # device (user report: mostly off-device points). Same principle
        # as global grounding: no box prompts in the input image. The
        # rack TYPE-CONFIRM call keeps the wireframe (it judges the box
        # fit); point generation judges the DEVICE.
        verdict = judge.adjudicate_sam_points(
            view["image"], box, view["name"],
            png_path=view["path"])
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
            # a backend answering in absolute pixels (Qwen2.5-VL's
            # convention) gets shrunk ~0.77x toward the top-left by the
            # 0-1000 grid conversion; re-map when the evidence says so
            coords = _calibrate_coord_scale(view["image"], coords, labels)
            # edge-sitting positives hit attached cables/ladders -> SAM
            # segments them in; pull them back onto the device interior
            coords, labels = _pull_points_inward(view["image"], coords,
                                                 labels)
            # clustered VLM points make SAM segment a local part; spread
            # them with device-interior samples when coverage is poor
            coords, labels = _augment_spread(view["image"], coords, labels)
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
