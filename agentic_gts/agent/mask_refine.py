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
class BoxGroup:
    bbox_norm: tuple[float, float, float, float]   # x1,y1,x2,y2 on 0..1000
    hypothesis: str = "rack"
    confidence: float = 0.5

    def pixel_box(self, W: int, H: int) -> np.ndarray:
        x1, y1, x2, y2 = self.bbox_norm
        return np.array([x1 / 1000.0 * (W - 1), y1 / 1000.0 * (H - 1),
                          x2 / 1000.0 * (W - 1), y2 / 1000.0 * (H - 1)],
                         dtype=np.float32)


def parse_box_groups(text: str) -> list[BoxGroup]:
    """Parse Qwen box-grounding JSON (relative 0..1000 coordinates).

    Accepted shapes:
      {"candidate_groups": [{"bbox_2d": [x1,y1,x2,y2], ...}]}
      [{"bbox_2d" | "bbox" | "box": [...]}]
      {"bbox_2d": [...]}

    Box grounding is Qwen3-VL's NATIVE task format (bbox_2d), unlike
    point placement which the model does poorly -- the reason the
    pipeline switched from point prompts to a pure box prompt for SAM.
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
        # Pick the strongest structure, not the LAST-scanned fragment:
        # the scan above also enters every INNER object, so a reversed
        # first-hit would return the final single item of a multi-box
        # reply (truncating 1-3 candidates to the last one). Priority:
        # candidate_groups dict > official cookbook ARRAY of
        # {"bbox_2d", "label"} items > one bare box dict.
        def _has_bbox(v) -> bool:
            return (isinstance(v, dict)
                    and any(k in v for k in ("bbox_2d", "bbox", "box")))

        best, best_rank = None, -1
        for value in values:
            if (isinstance(value, dict)
                    and ("candidate_groups" in value or "groups" in value)):
                rank = 3
            elif (isinstance(value, list) and value
                    and all(isinstance(x, dict) for x in value)
                    and any(_has_bbox(x) for x in value)):
                rank = 2
            elif _has_bbox(value):
                rank = 1
            else:
                continue
            if rank > best_rank:
                best, best_rank = value, rank
        data = best
    if data is None:
        return []
    if isinstance(data, dict):
        items = data.get("candidate_groups") or data.get("groups") or [data]
    elif isinstance(data, list):
        items = data
    else:
        return []

    def _bbox(raw):
        if raw is None:
            return None
        if isinstance(raw, dict):
            raw = raw.get("bbox_2d") or raw.get("bbox") or raw.get("box")
        try:
            x1, y1, x2, y2 = (float(v) for v in raw[:4])
        except (TypeError, ValueError):
            return None
        if max(abs(x1), abs(y1), abs(x2), abs(y2)) <= 1.0:
            x1, y1, x2, y2 = x1 * 1000.0, y1 * 1000.0, \
                x2 * 1000.0, y2 * 1000.0
        x1, x2 = min(x1, x2), max(x1, x2)
        y1, y2 = min(y1, y2), max(y1, y2)
        if not (0 <= x1 < x2 <= 1000 and 0 <= y1 < y2 <= 1000):
            return None
        return (x1, y1, x2, y2)

    groups = []
    for item in items if isinstance(items, list) else []:
        if not isinstance(item, dict):
            continue
        bbox = _bbox(item.get("bbox_2d") or item.get("bbox")
                     or item.get("box"))
        if bbox is None:
            continue
        try:
            conf = float(item.get("confidence", 0.5))
        except (TypeError, ValueError):
            conf = 0.5
        # official cookbook emits "label"; the earlier custom draft
        # asked for "hypothesis" -- accept both
        label = str(item.get("label") or item.get("hypothesis")
                    or "rack")[:20]
        groups.append(BoxGroup(bbox, label, min(max(conf, 0.0), 1.0)))
    return groups


class SamPredictorAdapter:
    """Lazy SAM2/SAM1 predictor with a uniform box-prompt interface."""

    def __init__(self, checkpoint: str | None = None,
                 model_cfg: str | None = None):
        self.checkpoint = checkpoint or os.environ.get("SAM_CHECKPOINT")
        self.model_cfg = model_cfg or os.environ.get("SAM_MODEL_CFG")
        self._predictor = None
        self._last_img = None

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

    def predict(self, image: np.ndarray,
                box: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        """Segment with a pure BOX prompt (pixel coords [x1,y1,x2,y2]).

        The box prompt is SAM's canonical interaction: coarse anchoring
        from the VLM, fine boundaries from the mask. multimask_output
        keeps the 3 granularity candidates (object / part / supart)
        competing downstream.

        Efficiency: the image is encoded ONCE per distinct array OBJECT
        (identity check, not content). A multi-group view -- a joined
        row the local grounding split into G cabinets -- runs G box
        prompts over the SAME rendering, and the Hiera encoder (not the
        lightweight mask head) is SAM's dominant cost: re-encoding per
        prompt multiplied that cost by G. A different image always
        re-encodes, so correctness never depends on the cache.
        """
        self._load()
        if self._last_img is not image:
            u8 = (np.clip(image[..., :3], 0, 1) * 255).astype(np.uint8)
            self._predictor.set_image(u8)
            self._last_img = image
        masks, scores, _ = self._predictor.predict(
            box=np.asarray(box, dtype=np.float32),
            multimask_output=True)
        return np.asarray(masks, dtype=bool), np.asarray(scores, dtype=float)


def json_default(o):
    """json.dump default handler: numpy scalars/arrays and paths.

    Audit dicts carry box.to_dict() output, whose `meta` transparently
    forwards whatever earlier stages stored there -- numpy floats, int64
    counts, small arrays. Plain json.dump raises TypeError on those
    (np.int64 is not an int subclass on Windows), which used to kill the
    whole mask_refine.json / type_confirm.json save."""
    if isinstance(o, np.generic):
        return o.item()
    if isinstance(o, np.ndarray):
        return o.tolist()
    if isinstance(o, os.PathLike):
        return os.fspath(o)
    raise TypeError(f"not JSON serializable: {type(o).__name__}")


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

    Structure detection is opacity-MASS based (5cm bins along the face
    normal, a bin counts as blocking when its summed opacity >= 1.0): a
    wall -- even one FLUSH against the box face or rendered diffuse and
    low-opacity -- is a solid mass of gaussians, while an aisle holds at
    most isolated floaters that must not count as blocking. The earlier
    single-nearest-point test with a 0.10 m dead zone was blind to a
    flush wall: that side measured a full-width corridor, the camera
    walked straight through the wall and rendered the view from OUTSIDE
    the room (a fog of structure behind the wall, user report).

    ROOM-INTERIOR prior (user rule): a PERIPHERAL wall-adjacent box
    faces the cloud's interior -- extremely diffuse walls (every 5cm
    bin under the mass threshold) still slip past the opacity test and
    measure a wide corridor, so geometry vetoes them: when a side both
    points AWAY from the cloud's interior (dot < -0.35 vs the median
    centre) and the cloud ENDS just past that face (< 1.2 m from the
    face to the 99th-pct extent of the points beyond it), that side is
    the wall, whatever the corridor said. Only a WIDE measured corridor
    (>= 1.5 m) is vetoed: a narrow one means a real facing structure
    was detected and stands (e.g. the 0.4m back gap of mid-room
    back-to-back rows must keep winning against the far aisle).
    """
    pts = np.asarray(gs.means, dtype=float)
    op = 1.0 / (1.0 + np.exp(-np.asarray(gs.raw_opacity, dtype=float)))
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
    bin_edges = np.arange(0.02, reach + 0.05, 0.05)
    med_xy = np.median(pts[:, :2], axis=0)
    to_interior = med_xy - c[:2]
    tc = float(np.linalg.norm(to_interior))
    best_vec, best_corridor = face_axis.copy(), -1.0
    for s in (1.0, -1.0):
        beyond = du * s - face_half        # distance past the s-side face
        m = (np.abs(dv) < long_half + 1.0) & (beyond > 0.02) & \
            (beyond < reach) & band
        if m.any():
            hist, _ = np.histogram(beyond[m], bins=bin_edges, weights=op[m])
            blocked = np.nonzero(hist >= 1.0)[0]
            # left edge of the first blocked bin: slightly conservative
            # (camera stands a touch closer), never through the wall
            corridor = (float(bin_edges[blocked[0]])
                       if len(blocked) else reach)
        else:
            corridor = reach
        # interior veto: this side is the wall of a peripheral box
        if corridor >= 1.5 and tc > 0.1:
            out = s * face_axis
            if float(out @ to_interior) / tc < -0.35:
                # how far the cloud continues PAST this face (all
                # heights/positions: a wall spans the room): a real
                # aisle opens into the room's interior structure
                beyond_all = (pts[:, :2] - c[:2]) @ out - face_half
                past = beyond_all[beyond_all > 0.0]
                edge_gap = (float(np.percentile(past, 99.0))
                            if len(past) else 0.0)
                if edge_gap < 1.2:
                    corridor = 0.0
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
    # between front and side. The roles are DECOUPLED (refine_box):
    # front is the sole voter on instance division (door seams / height
    # / color are legible face-on); the diagonal's foreshortening makes
    # adjacent cabinets visually merge, so it NEVER grounds -- it only
    # completes DEPTH via SAM prompts projected from the front fit.
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


def _save_sam_debug(view: dict | None, box_prompt, mask, pts3,
                    box, fitted, out_dir: str, tag: str) -> None:
    """Composite debug render for ONE SAM candidate (user request):
      panel 1: the view image with the VLM's box prompt drawn (green)
      panel 2: the same image with the SAM mask overlaid (cyan tint +
               solid edge)
      panel 3: the back-projected 3D points top-down (z colored), the
               OLD box (dashed blue) and the FITTED box (solid red)
    view=None renders panel 3 only (the multi-view union candidate has
    no single owning view). Never raises: debug output must not break
    the refinement.
    """
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
            # ---- panel 1: the VLM's box prompt on the clean view ----
            p1 = Image.fromarray(img.copy())
            d1 = ImageDraw.Draw(p1)
            if box_prompt is not None:
                x1, y1, x2, y2 = (float(v) for v in box_prompt[:4])
                d1.rectangle((x1, y1, x2, y2), outline=(0, 255, 0), width=3)
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
            titles = ["1. VLM box prompt (green)",
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


def _apply_depth_from_oblique(instances: list, pts: np.ndarray,
                              seed: "OrientedBox",
                              min_pts: int = 20) -> list[dict]:
    """Update each front instance's DEPTH dimension from oblique-view
    back-projected points (the ONLY thing the oblique view contributes).

    The oblique grounding may see a joined row as ONE instance -- its
    foreshortening merges adjacent cabinets -- so its instance division
    is NEVER adopted. Only the points matter, sliced per instance: an
    oblique mask's points whose ALONG coordinate falls inside a front
    instance's along span belong to that cabinet, and their cross-axis
    (depth) extent measures it (the 45-deg view sees front AND side
    faces). Along/height stay front-measured: oblique perspective
    distorts along-row spans, which is exactly why the front view alone
    votes on division.
    """
    recs = []
    yaw = float(seed.yaw)
    axis = np.array([math.cos(yaw), math.sin(yaw)])
    cross = np.array([-math.sin(yaw), math.cos(yaw)])
    if pts is None or len(pts) == 0:
        return recs
    along_all = pts[:, :2] @ axis
    cross_all = pts[:, :2] @ cross
    sc = np.asarray(seed.center, dtype=float)
    seed_cross_c = float(sc[:2] @ cross)
    seed_half_d = float(np.asarray(seed.size)[1]) / 2.0
    for inst in instances:
        f = inst["fitted"]
        fc = np.asarray(f.center, dtype=float)
        along_c = float(fc[:2] @ axis)
        half = float(f.size[0]) / 2.0
        m = np.abs(along_all - along_c) <= half + 0.10
        sel = pts[m]
        rec = {"points": int(len(sel))}
        recs.append(rec)
        if len(sel) < min_pts:
            rec["reason"] = "too few oblique points in along span"
            continue
        c_lo, c_hi = np.percentile(cross_all[m], [2.0, 98.0])
        depth = float(c_hi - c_lo)
        if not (0.3 <= depth <= 2.5):
            rec["reason"] = f"implausible depth {depth:.2f}m"
            continue
        mid = 0.5 * (c_lo + c_hi)
        # the measured cabinet centre must still live inside the seed's
        # cross span (padded) -- the Stage-A box bounds the device
        if abs(mid - seed_cross_c) > seed_half_d + 0.30:
            rec["reason"] = "depth centre too far from seed"
            continue
        # rebuild: front's along/height/centre-height, oblique-measured
        # depth and cross centre
        cxy = axis * along_c + cross * mid
        inst["fitted"] = OrientedBox(
            center=(float(cxy[0]), float(cxy[1]), float(fc[2])),
            size=(float(f.size[0]), depth, float(f.size[2])),
            yaw=yaw)
        inst["pts"] = np.vstack([inst["pts"], sel])
        rec["accepted"] = True
        rec["depth"] = round(depth, 3)
    return recs


def refine_box(scene: Scene, box: OrientedBox, judge, sam: SamPredictorAdapter,
               out_dir: str | None = None,
               views: list | None = None) -> tuple[list, dict]:
    """Run VLM box grounding + SAM + 3D fitting for one box.

    Box-only prompting (no points): point placement is a WEAK Qwen3-VL
    skill (user report: prompts mostly off the device despite a clean
    input image), while box grounding is the model's NATIVE task -- and
    the box prompt is SAM's canonical interaction, forgiving of prompt
    error where points are brittle. The VLM draws the coarse box; SAM
    snaps the mask to the device inside it; the 3D fit comes from the
    mask's back-projected gaussians.

    DECOUPLED views (user decision): instance division is voted on by
    the FRONT view ONLY -- door seams / height / color differences are
    legible face-on, and letting a second view vote too produced
    count inconsistencies (oblique's foreshortening merges adjacent
    cabinets). The oblique view DOES run its own local grounding (same
    prompt), but its groups are never adopted as instances -- all its
    back-projected points form a pool, sliced per front instance by
    along-row span, from which ONLY the depth dimension is measured
    (_apply_depth_from_oblique): whether oblique grounds 1 merged box
    or N boxes, each front instance takes just its cabinet's depth.

    MULTI-instance: the front grounding separates a joined row into one
    group per visually distinct cabinet (different height / color).
    Returns the list of accepted instances (score >= 0.45) -- more than
    one means the grounding split the row, and the caller replaces the
    old box with all of them.
    """
    if views is None:
        views = render_local_views(scene, box, out_dir)
    audit = {"box_id": box.box_id, "views": [], "accepted": False}
    if not views or not sam.available:
        audit["reason"] = "no local GS views or SAM checkpoint"
        return [], audit
    # the voter view grounds instances; the others only complete depth.
    # Fallback order keeps ONE voter at all times (front preferred;
    # without a front slot the first view votes, still one vote).
    voter = next((v for v in views if v["name"] == "front"), None)
    if voter is None:
        voter = views[0]
    depth_views = [v for v in views if v is not voter]

    # ---- pass 1 (voter): VLM grounding -> SAM -> fitted instances ----
    # CLEAN image to the VLM: the wireframe overlay (prompt_image) is
    # the Stage-A box, which is often oversized/misplaced -- the VLM
    # anchors on the frame instead of the device. Same principle as
    # global grounding: no box prompts in the input image.
    verdict = judge.adjudicate_sam_boxes(
        voter["image"], box, voter["name"], png_path=voter["path"])
    groups = verdict.params.get("groups", []) if verdict.params else []
    va = {"view": voter["name"], "image": voter["path"], "role": "voter",
          "answer": verdict.raw or verdict.detail, "groups": groups}
    H, W = voter["image"].shape[:2]
    instances = []
    for gi, g in enumerate(groups):
        group = BoxGroup(tuple(g["bbox"]), g.get("hypothesis", "rack"),
                         float(g.get("confidence", 0.5)))
        box_pix = group.pixel_box(W, H)
        if not (box_pix[2] - box_pix[0] > 4 and box_pix[3] - box_pix[1] > 4):
            continue                  # degenerate/absent box
        masks, scores = sam.predict(voter["image"], box_pix)
        best = None
        for mi, (mask, ms) in enumerate(zip(masks, scores)):
            pts3 = _mask_to_points(scene, box, mask, voter["cam"])
            fitted = fit_mask_points(pts3, box)
            if out_dir:
                # debug composite: prompt box + mask + lifted pts
                _save_sam_debug(voter, box_pix, mask, pts3, box,
                                fitted, out_dir,
                                f"{box.box_id}_{voter['name']}_g{gi}_m{mi}")
            if fitted is None:
                continue
            s = score_candidate(float(ms), pts3, fitted, box)
            if best is None or s > best[0]:
                best = (s, pts3, fitted, float(ms))
        if best is not None and best[0] >= 0.45:
            s, pts, fitted, ms = best
            instances.append({"score": s, "pts": pts, "fitted": fitted,
                              "mask_score": ms, "label": group.hypothesis,
                              "view": voter["name"]})
    if out_dir:
        try:
            with open(os.path.join(
                    out_dir, f"sam_boxes_{box.box_id}_{voter['name']}.json"),
                      "w", encoding="utf-8") as f:
                json.dump(va, f, ensure_ascii=False, indent=2,
                          default=json_default)
        except OSError:
            pass
    audit["views"].append(va)

    # ---- pass 2 (depth views): independent grounding, DEPTH-ONLY use ----
    # The oblique view runs the SAME local grounding prompt as front,
    # but its groups are never adopted as instances (foreshortening
    # merges adjacent cabinets). All groups' back-projected points form
    # a pool; _apply_depth_from_oblique slices it per front instance
    # (along span) and measures ONLY that cabinet's depth.
    for dv in (depth_views if instances else []):
        verdict = judge.adjudicate_sam_boxes(
            dv["image"], box, dv["name"], png_path=dv["path"])
        groups = verdict.params.get("groups", []) if verdict.params else []
        dva = {"view": dv["name"], "image": dv["path"],
               "role": "depth_grounding", "groups": groups,
               "instances": []}
        H, W = dv["image"].shape[:2]
        pool, pool_ms = [], 0.0
        for gi, g in enumerate(groups):
            group = BoxGroup(tuple(g["bbox"]), g.get("hypothesis", "rack"),
                             float(g.get("confidence", 0.5)))
            box_pix = group.pixel_box(W, H)
            if not (box_pix[2] - box_pix[0] > 4
                    and box_pix[3] - box_pix[1] > 4):
                continue                  # degenerate/absent box
            masks, scores = sam.predict(dv["image"], box_pix)
            best = None
            for mi, (mask, ms) in enumerate(zip(masks, scores)):
                pts3 = _mask_to_points(scene, box, mask, dv["cam"])
                if out_dir:
                    _save_sam_debug(dv, box_pix, mask, pts3, box,
                                    None, out_dir,
                                    f"{box.box_id}_{dv['name']}_g{gi}_m{mi}")
                if len(pts3) < 20:
                    continue
                if best is None or ms > best[1]:
                    best = (pts3, float(ms))
            if best is not None:
                pool.append(best[0])
                pool_ms = max(pool_ms, best[1])
        if pool:
            recs = _apply_depth_from_oblique(
                instances, np.vstack(pool), box)
            dva["instances"] = recs
            for inst, rec in zip(instances, recs):
                if not rec.get("accepted"):
                    continue    # depth measurement rejected: front fit stands
                inst["view"] = inst["view"] + f'+{dv["name"]}(depth)'
                inst["mask_score"] = 0.5 * (inst["mask_score"] + pool_ms)
                inst["score"] = min(
                    score_candidate(inst["mask_score"], inst["pts"],
                                    inst["fitted"], box) + 0.08, 1.0)
        if out_dir:
            try:
                with open(os.path.join(
                        out_dir, f"sam_boxes_{box.box_id}_{dv['name']}.json"),
                          "w", encoding="utf-8") as f:
                    json.dump(dva, f, ensure_ascii=False, indent=2,
                              default=json_default)
            except OSError:
                pass
        audit["views"].append(dva)

    if not instances:
        audit["reason"] = "no SAM mask yielded a valid 3D box"
        return [], audit
    instances.sort(key=lambda e: e["score"], reverse=True)
    audit.update({"accepted": True,
                  "instances": [
                      {"score": round(e["score"], 4), "view": e["view"],
                       "points": len(e["pts"]),
                       "sam_score": round(e["mask_score"], 4),
                       "label": e["label"], "box": e["fitted"].to_dict()}
                      for e in instances]})
    return instances, audit
