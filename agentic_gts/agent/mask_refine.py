"""Local VLM box grounding -> SAM mask -> split-correction of one seed box.

Contract (user-directed):
  * The FRONT local view's VLM boxes feed SAM as pure box prompts; the
    back-projected mask SURFACE guides HOW the seed box splits along the
    row (each visually distinct cabinet its own span). The seed's yaw /
    height / depth are trusted -- pieces are splits OF the seed, never
    free re-fits.
  * The SIDE view (profile along the row axis) corrects each piece's
    THICKNESS: an open cabinet door sticks out horizontally beyond the
    body there, separable only from the side (the front view cannot).
    The VLM prompt excludes doors and the strong-bin estimator drops
    whatever door tail still leaks through.

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


_QUALITY_RE = re.compile(
    r"quality\s*[:：]\s*[\"']?(good|poor|true|false|yes|no)\b",
    re.IGNORECASE)


def reply_view_quality(text: str) -> str:
    """Parse the VLM's per-view quality verdict ('good' | 'poor').

    The grounding prompt asks for a first line 'quality: good|poor'
    judged IN THE SAME CALL as the box grounding (user direction: no
    extra call, no extra budget). A garbage view -- fogged, blurred,
    washed out -- has its boxes DROPPED however confidently the model
    drew them: a haze invites hallucinated structure. A reply with no
    marker (the model skipped the line) reads GOOD -- dropping a view
    loses signal, so it takes an explicit poor verdict. The LAST
    match wins when the model states its verdict more than once.
    """
    verdict = "good"
    for m in _QUALITY_RE.finditer(text or ""):
        verdict = m.group(1).lower()
    return "poor" if verdict in ("poor", "false", "no") else "good"


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
      [{"bbox_2d" | "bbox" | "box": [...]}]   (official cookbook array)
      {"bbox_2d": [...]}                       (single box)
      JSONL of single boxes -- one {"bbox_2d": ...} object per line.
      The prompt's example is a single bare dict, so models often
      emit one object per instance; the old pick-ONE-structure logic
      kept only the first line's box (user report: the answer held
      many instances, groups had one).

    Box grounding is Qwen3-VL's NATIVE task format (bbox_2d), unlike
    point placement which the model does poorly -- the reason the
    pipeline switched from point prompts to a pure box prompt for SAM.
    """
    if not text:
        return []
    # scan every object/array opener with the standard JSON decoder;
    # thinking is already stripped upstream, prose fragments mostly
    # fail to decode
    dec = json.JSONDecoder()
    frags = []
    for m in re.finditer(r"[\[{]", text):
        try:
            value, n = dec.raw_decode(text[m.start():])
        except json.JSONDecodeError:
            continue
        frags.append((m.start(), m.start() + n, value))
    if not frags:
        return []
    # keep only TOP-LEVEL fragments: the scan also enters every INNER
    # object/array of a valid reply, and those nested decodes are
    # partial copies of the same items
    tops = [v for s, e, v in frags
            if not any((s2, e2) != (s, e) and s2 <= s and e <= e2
                       for s2, e2, _ in frags)]

    def _items(v):
        if isinstance(v, dict):
            if "candidate_groups" in v or "groups" in v:
                inner = v.get("candidate_groups") or v.get("groups")
                if isinstance(inner, list):
                    return [i for i in inner if isinstance(i, dict)]
                return []
            return [v]
        if isinstance(v, list):
            return [x for x in v if isinstance(x, dict)]
        return []

    # three candidate pools; the fullest wins (a thin draft loses to
    # the full final answer, and JSONL lines merge into one pool)
    pools = []
    wrapped = [v for v in tops if isinstance(v, dict)
               and ("candidate_groups" in v or "groups" in v)]
    if wrapped:
        pools.append(max((_items(v) for v in wrapped), key=len))
    arr_items = [i for v in tops if isinstance(v, list) for i in _items(v)]
    if arr_items:
        pools.append(arr_items)
    dict_items = [i for v in tops
                  if isinstance(v, dict) and "candidate_groups" not in v
                  and "groups" not in v for i in _items(v)]
    if dict_items:
        pools.append(dict_items)
    if not pools:
        return []
    items = max(pools, key=len)

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

    ROOM-BOUNDARY override (user rule: a peripheral wall-adjacent box
    faces the room's interior): the corridor measurement stays blind to
    an EXTREMELY diffuse wall -- when every 5cm bin stays under the
    mass threshold, the wall side measures a full-width corridor and
    wins. What a diffuse wall cannot fake is far-field CONTENT: past a
    real aisle the room continues (devices, floor, facing rows -- a
    large opacity mass beyond 1.2 m from the face), while past a wall
    there is only faint smear. When one side carries >= 3x the other's
    far-field mass (floor 20), that side is the room side and wins
    outright, whatever its measured corridor. Symmetric far mass
    (back-to-back rows mid-room) falls back to the corridor pick.

    Returns (open_vec, corridor): the open direction (unit world 2D
    vector) and the corridor width in metres (capped at `reach`).
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
    strip = np.abs(dv) < long_half + 1.0
    bin_edges = np.arange(0.02, reach + 0.05, 0.05)
    side_corridor, side_far = {}, {}
    for s in (1.0, -1.0):
        beyond = du * s - face_half        # distance past the s-side face
        m = strip & (beyond > 0.02) & (beyond < reach) & band
        corridor = reach
        if m.any():
            hist, _ = np.histogram(beyond[m], bins=bin_edges, weights=op[m])
            blocked = np.nonzero(hist >= 1.0)[0]
            if len(blocked):
                # left edge of the first blocked bin: slightly
                # conservative (camera stands a touch closer), never
                # through the wall
                corridor = float(bin_edges[blocked[0]])
        side_corridor[s] = corridor
        far = strip & (beyond > 1.2) & band
        side_far[s] = float(op[far].sum())
    f1, f2 = side_far[1.0], side_far[-1.0]
    if f1 >= 3.0 * f2 and f1 >= 20.0:
        s = 1.0
    elif f2 >= 3.0 * f1 and f2 >= 20.0:
        s = -1.0
    else:
        s = 1.0 if side_corridor[1.0] >= side_corridor[-1.0] else -1.0
    return s * face_axis, min(side_corridor[s], reach)


def _free_row_end(gs, box: OrientedBox, reach: float = 3.0):
    """The row's FREE end (the side-view camera's ground) and its
    corridor width.

    The side view looks ALONG the row from beyond one row end; the
    old slot picked that end blindly (front + 90), so a cabinet whose
    SIDE face is flush against a wall put the eye inside the wall and
    the render came out a smooth veil (user report). Same measurement
    as _open_side, but along the ROW axis beyond the row's end faces:
    5cm opacity-mass bins (a bin blocks when its summed opacity >=
    1.0 -- a wall is a solid mass however diffuse it renders), with
    the far-field override (past a real open end the room continues;
    past a wall there is only faint smear).

    Returns (end_sign, corridor, v): end_sign +-1 along the row axis
    v (the direction OUT of the free end), corridor the clear width
    past that end (capped at reach), v the unit row axis.
    """
    pts = np.asarray(gs.means, dtype=float)
    op = 1.0 / (1.0 + np.exp(-np.asarray(gs.raw_opacity, dtype=float)))
    c = np.asarray(box.center, dtype=float)
    yaw = float(box.yaw)
    axis = np.array([math.cos(yaw), math.sin(yaw)])       # local x
    cross = np.array([-math.sin(yaw), math.cos(yaw)])     # local y
    size = np.asarray(box.size, dtype=float)
    # the ROW axis carries the LONG side (same convention as _open_side)
    if size[0] >= size[1]:
        v, long_half = axis, size[0] / 2.0
    else:
        v, long_half = cross, size[1] / 2.0
    u = cross if size[0] >= size[1] else axis
    face_half = min(size[0], size[1]) / 2.0
    d = pts[:, :2] - c[:2]
    du = d @ u
    dv = d @ v
    z = pts[:, 2]
    z_top = c[2] + size[2] / 2.0
    band = (z > 0.25) & (z < max(min(z_top - 0.2, 2.0), 0.5))
    strip = np.abs(du) < face_half + 1.0
    bin_edges = np.arange(0.02, reach + 0.05, 0.05)
    end_corridor, end_far = {}, {}
    for s in (1.0, -1.0):
        beyond = dv * s - long_half    # distance past the s-side end face
        m = strip & (beyond > 0.02) & (beyond < reach) & band
        corridor = reach
        if m.any():
            hist, _ = np.histogram(beyond[m], bins=bin_edges, weights=op[m])
            blocked = np.nonzero(hist >= 1.0)[0]
            if len(blocked):
                corridor = float(bin_edges[blocked[0]])
        end_corridor[s] = corridor
        far = strip & (beyond > 1.2) & band
        end_far[s] = float(op[far].sum())
    f1, f2 = end_far[1.0], end_far[-1.0]
    if f1 >= 3.0 * f2 and f1 >= 20.0:
        s = 1.0
    elif f2 >= 3.0 * f1 and f2 >= 20.0:
        s = -1.0
    else:
        s = 1.0 if end_corridor[1.0] >= end_corridor[-1.0] else -1.0
    return s, min(end_corridor[s], reach), v


def _side_azim(box: OrientedBox, azim_front: float,
               end_sign: float, row_v) -> float:
    """The side-view azimuth whose EYE stands out of the FREE row end.

    make_local_cam's eye displacement at azimuth A is R(A) @ cross
    (dist > 0 along it); for the side candidates that is R(azim_front
    +- 90) @ cross. 2D rotations commute, so R(azim_front + 90) @
    cross = R(azim_front) @ R(90) @ cross -- computed from the SAME
    math make_local_cam uses, never from a remembered sign convention
    (the trap that once inverted _front_azim).
    """
    az = math.radians(azim_front)
    rot = np.array([[math.cos(az), -math.sin(az)],
                    [math.sin(az), math.cos(az)]])
    base = np.array([-math.sin(float(box.yaw)),
                     math.cos(float(box.yaw))])
    plus = rot @ np.array([-base[1], base[0]])   # R(azim_front + 90) @ cross
    if float(np.asarray(plus) @ np.asarray(row_v)) * end_sign < 0.0:
        return azim_front - 90.0
    return azim_front + 90.0


def _box_only_mask(gs, box: OrientedBox, pad: float = 0.30,
                   wall_vec=None, face_half: float | None = None,
                   wall_pad: float = 0.05,
                   z_pad: float = 0.15) -> np.ndarray:
    """Boolean mask over gs: True only for gaussians INSIDE the box's OBB
    (plus `pad` metres of HORIZONTAL slack, since the globally grounded
    OBB carries a placement offset of up to ~15cm -- a tight slack cut
    a strip of the device off the local views, user report).

    The slack is HORIZONTAL-ONLY: `z_pad` (kept at the old 0.15m)
    governs the vertical axis, so widening `pad` does not reach up and
    pull the cable trays / ceiling haze above the device into the
    render.

    The local views exist to show the VLM and SAM exactly ONE device.
    Keeping the rest of the scene (the earlier 'normal aisle photo'
    attempt) re-introduced haze whenever the camera stood inside
    structure: big low-opacity training floaters fog the whole frame
    from any position. Hiding everything outside the box removes both
    the occluders AND the fog source in one rule -- what renders is
    exactly the device under adjudication, on a clean background.

    wall_vec / face_half / wall_pad (user report: the side view of a
    wall-flush row still fogged with the camera already on the free
    end -- the fog was IN the mask, not at the camera): a wall FLUSH
    against the box's closed lateral face has its gaussian means
    within the slack, so it rendered as a full-height sheet behind the
    rack no matter where the eye stood. On the WALLED side only, the
    outside-face slack shrinks to `wall_pad` (0.05m): the wall's means
    (>= 5cm past the face) drop out while the device's own bled face
    gaussians (a couple of cm, plus a small placement offset) survive.
    The open side keeps the full slack.
    """
    means = np.asarray(gs.means, dtype=float)
    local = box.world_to_local(means)
    half = np.asarray(box.size, dtype=float) / 2.0
    m = np.all(np.abs(local[:, :2]) <= half[:2] + pad, axis=1)
    m &= np.abs(local[:, 2]) <= half[2] + z_pad
    if wall_vec is not None and face_half is not None:
        off = (means[:, :2] - np.asarray(box.center, dtype=float)[:2]) \
            @ np.asarray(wall_vec, dtype=float)
        m &= off <= face_half + wall_pad
    return m


def _front_azim(box: OrientedBox, open_vec) -> float:
    """The front-view azimuth that puts make_local_cam's EYE on the box's
    open side.

    Sign trap (the bug that kept wall-adjacent boxes rendering from
    outside the room): make_local_cam's horizontal direction at azim A
    is R(A) @ base with base = local +y -- so azim 0 stands on the
    local +y side, but azim 90 stands on the local -x side, NOT +x.
    The flip must therefore compare open_vec against the side the
    camera would actually STAND ON, not against the face normal it is
    supposed to look at; comparing against +x inverted the flip for
    boxes whose long edge is on the cross axis.
    """
    yaw = float(box.yaw)
    cross = np.array([-math.sin(yaw), math.cos(yaw)])   # local +y
    axis = np.array([math.cos(yaw), math.sin(yaw)])    # local +x
    if box.size[0] >= box.size[1]:
        azim, cam_side = 0.0, cross
    else:
        azim, cam_side = 90.0, -axis
    if float(np.asarray(open_vec) @ cam_side) < 0.0:
        azim += 180.0       # the aisle is on the other side
    return azim


_SIDE_LETTER_GLYPHS = {
    "A": ["01110", "10001", "10001", "11111", "10001", "10001", "10001"],
    "B": ["11110", "10001", "10001", "11110", "10001", "10001", "11110"],
    "C": ["01110", "10001", "10000", "10000", "10000", "10001", "01110"],
    "D": ["11110", "10001", "10001", "10001", "10001", "10001", "11110"],
    "E": ["11111", "10000", "10000", "11110", "10000", "10000", "11111"],
    "F": ["11111", "10000", "10000", "11110", "10000", "10000", "10000"],
}

# Oblique side-view offset (user report: a long cable ladder beside a
# low device fully occluded the straight profile; a slight azimuth
# offset peeks past it without biasing the 3D thickness read).
_SIDE_OBLIQUE_DEG = 15.0


def _side_panel_image(imgs: list, scale: int = 16,
                      gap: int = 10) -> np.ndarray:
    """Composite side-view candidates into ONE labeled panel image.

    Panels sit side by side on a dark canvas with a big YELLOW A/B/C
    stenciled into each panel's top-left corner (pure-numpy bitmap
    glyphs -- no PIL font dependency). This is the single image the
    VLM arbitrates on (user direction: let the VLM pick the clearest
    side view instead of more placement rules).
    """
    H, W = imgs[0].shape[:2]
    n = len(imgs)
    canvas = np.full((H, n * W + (n - 1) * gap, 3), 0.03, dtype=np.float32)
    for i, im in enumerate(imgs):
        a = np.asarray(im, dtype=np.float32)[..., :3]
        canvas[:, i * (W + gap):(i + 1) * W + i * gap] = a
    for i in range(n):
        g = _SIDE_LETTER_GLYPHS[chr(65 + i)]
        x0 = i * (W + gap) + 14
        for r, row in enumerate(g):
            for c, v in enumerate(row):
                if v == "1":
                    canvas[14 + r * scale:14 + (r + 1) * scale,
                           x0 + c * scale:x0 + (c + 1) * scale] = \
                        (1.0, 0.9, 0.0)
    return canvas


def render_local_views(scene: Scene, box: OrientedBox,
                       out_dir: str | None = None,
                       judge=None) -> list[dict]:
    """Render front + side local views: ONE device, isolated.

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

    SIDE view (user direction: rules keep misjudging which end is
    clear, fog persists -- let the VLM look): TWO oblique candidates
    -- +15 deg at the rule-picked free end and +15 deg at the
    opposite end (the ends differ by 180 deg, so the same offset
    swings them to opposite lateral sides; the straight perpendicular
    profile is retired: it is the view a long cable ladder or clutter
    beside the device blocks), each passes the cheap gradient-energy
    gate, and the survivors are composited into one labeled A/B
    panel image for ONE tiny VLM call that picks the clearest. No
    judge / call failure / one survivor -> rule order.
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
    # SIDE, an earlier user report). _front_azim compares the open side
    # against the side the camera would actually STAND ON.
    open_vec, corridor = _open_side(gs, box)
    azim_front = _front_azim(box, open_vec)
    # view set, all at GROUND level (elev 18 deg, rack height -- no
    # top-down component: the local views must show the device's
    # vertical surfaces, which the ground-level 3DGS training observed
    # well): FRONT (the face: doors, panels -- a voter on instance
    # division, door seams / height / color are legible face-on) +
    # BACK (the mirrored face: the front aisle is sometimes a NARROW
    # corridor -- poor standoff, foreshortened row ends -- while the
    # back is open; the same cabinets, a second, often cleaner vote)
    # + SIDE (along the row axis: the depth/height PROFILE, where an
    # open door sticks out horizontally beyond the cabinet body and
    # the true thickness is measurable -- the front view cannot
    # separate a door, user report). NO geometric pre-gating (the
    # corridor-based back skip was reverted, user direction): every
    # view is rendered, then judged on its RENDER -- a view too poor
    # to judge does not participate in the refinement, whichever
    # view it is (a wall-adjacent box can fog up its BACK or its SIDE
    # render just the same).
    # standoff: ~80% into the corridor, never further than 2.2m; the
    # camera widens its lens to frame, it does not back off
    standoff = float(np.clip(0.8 * corridor, 0.6, 2.2))
    # WALLED lateral side (from _open_side's pick): the closed lateral
    # face's outside slack shrinks in _box_only_mask below -- a flush
    # wall's means sit within the horizontal slack and render as a
    # sheet behind the rack (user report: side still fogged with the
    # camera already on the free end -- the fog was IN the mask).
    if box.size[0] >= box.size[1]:
        face_half = float(box.size[1]) / 2.0
    else:
        face_half = float(box.size[0]) / 2.0
    ov = np.asarray(open_vec, dtype=float).ravel()
    wall_vec = -ov / (float(np.linalg.norm(ov)) + 1e-12)
    # SIDE slot placement (user report: the side view rendered as a
    # veil when the cabinet's SIDE face was flush against a wall): the
    # old azim = front + 90 stands the eye beyond ONE row end, picked
    # blindly -- the walled end puts the eye inside the wall. Pick the
    # FREE end instead (_free_row_end, same opacity-mass corridor
    # measurement as _open_side), and cap the side standoff by that
    # end's corridor (never past a measured wall: the 0.35 floor could
    # overrun a corridor under 0.44m). The candidate azimuth whose EYE
    # DISPLACEMENT (make_local_cam's own rotation math: R(azim) @ cross)
    # points out of the free end wins -- computed, not remembered, so
    # the sign trap that once inverted _front_azim cannot recur here.
    azim_side = azim_front + 90.0
    standoff_side = standoff
    try:
        end_sign, end_corridor, row_v = _free_row_end(gs, box)
        azim_side = _side_azim(box, azim_front, end_sign, row_v)
        standoff_side = float(min(np.clip(0.8 * end_corridor, 0.35, 2.2),
                                  max(end_corridor - 0.05, 0.10)))
        print(f"[mask-refine] side view: free end {end_sign:+.0f} "
              f"(corridor {end_corridor:.2f}m), wall-masked lateral "
              f"side, standoff {standoff_side:.2f}m")
    except Exception as e:
        print(f"[mask-refine] free-row-end pick failed "
              f"({type(e).__name__}: {e}) -> default side slot")
    # render ONLY the device: every gaussian outside the box's OBB
    # (plus slack; the WALLED lateral side's slack shrunk to 0.03m --
    # occluders and fog sources alike) is hidden. One subset serves
    # every view (the mask does not depend on the camera).
    sub = _subset_or_none(gs, _box_only_mask(
        gs, box, wall_vec=wall_vec, face_half=face_half))

    def _render_one(elev: float, azim: float, so: float):
        cam = make_local_cam([box], W=768, H=768, elev_deg=elev,
                             azim_deg=azim, standoff=so)
        raw = rasterize_gs(sub, cam) if sub is not None else None
        if raw is None:
            raw = render_gs_view(gs, [box], cam, overlay=None,
                                 isolate_boxes=True, isolate_margin=0.8)
            if raw is None:
                return None, None, cam
            prompt_img = render_gs_view(
                gs, [box], cam, overlay="wire3d", isolate_boxes=True,
                isolate_margin=0.8)
        else:
            prompt_img = render_gs_view(sub, [box], cam, overlay="wire3d")
        return raw, prompt_img, cam

    def _save_view(raw, name: str, prompt_img=None):
        path = prompt_path = None
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
        return path, prompt_path

    out = []
    for name, elev, azim, so in (("front", 18.0, azim_front, standoff),
                                 ("back", 18.0, azim_front + 180.0,
                                  standoff)):
        raw, prompt_img, cam = _render_one(elev, azim, so)
        if raw is None:
            continue
        # QUALITY GATE (user rule): judge the RENDER, not the geometry.
        # A wall-adjacent box fogs up whichever camera lands in
        # structure; a poor view that slipped through would feed the
        # VLM a haze and poison the split. The image is still saved
        # (audit: the user SEES which view was dropped and why).
        ok, why = _view_quality(raw)
        if not ok:
            print(f"[mask-refine] {name} view dropped: {why}")
            if out_dir:
                os.makedirs(out_dir, exist_ok=True)
                with open(os.path.join(
                        out_dir,
                        f"mask_{box.box_id}_{name}_dropped.png"),
                        "wb") as f:
                    f.write(png_bytes(raw))
            continue
        path, prompt_path = _save_view(raw, name, prompt_img)
        out.append({"name": name, "image": raw,
                    "prompt_image": prompt_img if prompt_img is not None
                    else raw,
                    "cam": cam, "path": path, "prompt_path": prompt_path})

    # ---- SIDE view: candidates + VLM arbitration ----
    # The side slot looks along the row from beyond one end; every
    # geometric pick so far (blind azim, free-end corridor, wall-masked
    # slack) still fogged on real scenes (user reports, twice). Render
    # a small candidate set instead and let the EVIDENCE decide: the
    # cheap gradient gate kills any candidate that rendered a veil,
    # then one tiny VLM call picks the clearest survivor.
    # TWO OBLIQUE candidates only (user direction: one per end is
    # enough -- the straight perpendicular profile is retired, it is
    # the view a long cable ladder or clutter beside the device
    # blocks). The ends' azimuths differ by 180 deg, so the SAME
    # +15 deg offset on each swings them to OPPOSITE lateral sides:
    # the pair covers both ends AND both peek directions at once.
    # Safe for the thickness read: it comes from the 3D
    # back-projected points' cross-axis span, not from pixel extents,
    # so the obliquity cannot bias it.
    alt_azim = (azim_front + 90.0
                if abs(azim_side - (azim_front - 90.0)) < 1e-6
                else azim_front - 90.0)
    side_cands = [(azim_side + _SIDE_OBLIQUE_DEG, standoff_side),
                  (alt_azim + _SIDE_OBLIQUE_DEG, standoff_side)]
    survivors = []
    for ci, (sa, ss) in enumerate(side_cands):
        raw, prompt_img, cam = _render_one(18.0, sa, ss)
        if raw is None:
            continue
        ok, why = _view_quality(raw)
        if not ok:
            print(f"[mask-refine] side candidate {chr(65 + ci)} "
                  f"dropped: {why}")
            if out_dir:
                os.makedirs(out_dir, exist_ok=True)
                with open(os.path.join(
                        out_dir, f"mask_{box.box_id}_side_"
                        f"{chr(65 + ci)}_dropped.png"), "wb") as f:
                    f.write(png_bytes(raw))
            continue
        survivors.append((raw, prompt_img, cam))
    if not survivors:
        print("[mask-refine] side view dropped: no candidate passed "
              "the quality gate")
        return out
    pick = 0
    if len(survivors) >= 2 and judge is not None \
            and getattr(judge, "backend", "mock") != "mock":
        try:
            panel = _side_panel_image([s[0] for s in survivors])
            panel_path = None
            if out_dir:
                os.makedirs(out_dir, exist_ok=True)
                panel_path = os.path.join(
                    out_dir, f"side_pick_{box.box_id}.png")
                with open(panel_path, "wb") as f:
                    f.write(png_bytes(panel))
            v = judge.adjudicate_side_pick(panel, box,
                                           n_panels=len(survivors),
                                           png_path=panel_path)
            p = v.params.get("pick") if v.params else None
            if p is not None and 0 <= int(p) < len(survivors):
                pick = int(p)
        except Exception as e:
            print(f"[mask-refine] side VLM arbitration failed "
                  f"({type(e).__name__}: {e}) -> rule order")
    raw, prompt_img, cam = survivors[pick]
    print(f"[mask-refine] side view: candidate {chr(65 + pick)} of "
          f"{len(survivors)} survivor(s) chosen"
          f"{' (VLM)' if len(survivors) >= 2 else ''}")
    for ci, (r_, _pi, _c) in enumerate(survivors):
        if ci != pick and out_dir:
            os.makedirs(out_dir, exist_ok=True)
            with open(os.path.join(
                    out_dir,
                    f"mask_{box.box_id}_side_"
                    f"{chr(65 + ci)}_unused.png"), "wb") as f:
                f.write(png_bytes(r_))
    path, prompt_path = _save_view(raw, "side", prompt_img)
    out.append({"name": "side", "image": raw,
                "prompt_image": prompt_img if prompt_img is not None
                else raw,
                "cam": cam, "path": path, "prompt_path": prompt_path})
    return out


def _view_quality(img: np.ndarray) -> tuple[bool, str]:
    """Gross quality judgement of one local-view render (user rule: a
    view too poor to judge must not participate in the refinement).

    Two signals, matched to what the failure modes actually look like:
      * EMPTY -- nothing rendered (a broken placement): near-zero
        visible coverage.
      * VEIL -- the camera stood inside structure (a flush wall's
        diffuse gaussians, the narrow gap of a wall-adjacent box):
        the frame fills with a SMOOTH semi-bright veil. The
        discriminator is GRADIENT ENERGY, not the value spread: a veil
        is smooth (at most slow gradients, mean squared gradient
        ~1e-7) while a device render is full of crisp steps --
        cabinet contours, panel seams, door frames (~1e-3), two to
        three orders of magnitude apart, so the threshold has margin
        both ways. Unlike a std/coverage histogram test, gradient
        energy is not fooled by a TEXTURED veil (its own variation
        counts toward std), by a GRADIENT veil (a brightness ramp
        spreads the histogram); and it does not reject a legitimate
        close-up of a flat-panel cabinet (uniform values, low std, but
        its seams and contour still carry edges).

    Returns (ok, reason); reason is "" when ok.
    """
    if img is None or not np.asarray(img).size:
        return False, "no image"
    lum = np.clip(np.asarray(img, dtype=float)[..., :3], 0.0, 1.0)
    lum = lum.mean(axis=2)
    cov = float((lum > 0.10).mean())
    if cov < 0.02:
        return False, f"empty frame (device coverage {cov:.1%})"
    gx = np.diff(lum, axis=1)
    gy = np.diff(lum, axis=0)
    edge = float(np.mean(gx * gx) + np.mean(gy * gy))
    if edge < 1e-5:
        return False, (f"smooth veil, no structure (edge energy "
                       f"{edge:.2e}, coverage {cov:.1%})")
    return True, ""


def _mask_to_points(scene: Scene, box: OrientedBox, mask: np.ndarray, cam,
                    margin: float = 0.25, z_buffer: bool = True,
                    exclude: np.ndarray | None = None) -> np.ndarray:
    """Lift mask to local 3DGS centers.

    z_buffer=True (front view): only points close to the nearest
    projected depth in each pixel -- the visible SURFACE. Selecting every
    center whose projection lands in the mask also selects surfaces
    hidden behind the visible rack, inflating the fitted box.

    z_buffer=False (side view): every region point projecting into the
    mask, at ANY depth. The side camera looks ALONG the row, so all
    cabinets of the seed overlap in projection; keeping only the nearest
    surface would starve the inner cabinets of depth-measurement points.

    exclude (pixel mask, same shape): points whose projection lands in
    this mask are DROPPED -- the open-door subtraction. The door class
    is detected by the VLM as its own positive instance (user finding:
    the model detects "open cabinet door" reliably but cannot exclude
    it via a negative instruction); its SAM mask pixel-subtracts the
    door from every device mask, so door points never reach the
    span/thickness pools -- box refinement only ever fits devices.

    margin: the candidate region is the seed OBB grown by this much.
    Backprojected points far outside the seed are NOISE -- a mask edge
    bleeding onto the floor / neighbouring structure picks up their
    pixels, and the fitted box balloons toward them (user report:
    backprojected points well past the initial box). 0.25 m allows the
    legitimate case (a slightly conservative grounding box growing to
    the true surface, which the nadir point-fit places within ~0.2 m)
    while dropping the bleed the old 0.8 m margin let through.
    """
    pts = np.asarray(scene.points, dtype=float)
    region = box.contains(pts, margin=margin)
    pts = pts[region]
    if not len(pts):
        return pts
    uv = cam.project_cv(pts)
    x = np.rint(uv[:, 0]).astype(int)
    y = np.rint(uv[:, 1]).astype(int)
    valid = ((x >= 0) & (x < mask.shape[1]) &
             (y >= 0) & (y < mask.shape[0]))
    if exclude is not None:
        valid &= ~exclude[np.clip(y, 0, exclude.shape[0] - 1),
                          np.clip(x, 0, exclude.shape[1] - 1)]
    if not z_buffer:
        selected = np.where(valid)[0]
        keep = np.zeros(len(pts), dtype=bool)
        keep[selected[mask[y[selected], x[selected]]]] = True
        return pts[keep]
    h = np.hstack([pts, np.ones((len(pts), 1))])
    pc = h @ cam.view_cv().T
    depth = pc[:, 2]
    valid &= (depth > 0.05)
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


def _absorb_single(single: dict, fines: list, axis, along0: float) -> bool:
    """Fold the single-instance face's span into the multi-instance
    face's split (user rule: one face grounds ONE instance, the other
    grounds SEVERAL -> the several stand; the row is one whole and the
    single box is that whole unresolved).

    NOTHING of the single survives (user report: the back view's
    unresolved whole-row box swallowed the top cable connections into
    its SAM mask, and the point contribution dragged the pieces'
    P97.5 height to the cable bundle even after the extent was
    dropped) -- extent dropped, points dropped, only the collision
    flag returned. Returns True when the single overlapped any fine
    span (absorbed); False when it overlapped none (the caller keeps
    it -- recall: a region the multi face never grounded)."""
    for f in fines:
        if min(single["hi"], f["hi"]) - max(single["lo"], f["lo"]) > 0:
            return True
    return False


def _merge_cross_view(spans: list, axis=None, along0: float = 0.0) -> list:
    """Reconcile the SAME cabinets voted from BOTH faces.

    The back view is the front's mirror: the SAME physical cabinets,
    so a cabinet grounded in both views yields two spans overlapping
    by nearly the full cabinet width (>= 50% of the shorter -- the
    relative threshold that adjacent cabinets, overlapping only by
    the seam-placement difference, never reach).

    USER RULE -- instance COUNT decides first: when one face grounds
    ONE instance while the other grounds SEVERAL, the several stand
    (the row is one whole; the single box is that whole unresolved).
    The single face's span is fully DISCARDED -- extent dropped AND
    points dropped (user report: the single face's whole-row mask
    carried the top cable connections into the pieces' point pools
    and the P97.5 height read the cable bundle).

    Graph merge handles the rest: mutual single-overlap pairs union
    (both faces saw one cabinet), a span bridging TWO OR MORE spans
    is dropped (a coarse whole-row vote from an N-vs-M disagreement),
    singletons pass through (a cabinet legible from only one face
    still splits the row).
    """
    by_view = {}
    for s in spans:
        by_view.setdefault(s.get("view"), []).append(s)
    if len(by_view) == 2:
        a, b = list(by_view.values())
        if len(a) == 1 and len(b) > 1:
            if _absorb_single(a[0], b, axis, along0):
                print(f"[mask-refine] cross-view: single-instance face "
                      f"absorbed into the {len(b)}-instance face's split")
                spans = b
        elif len(b) == 1 and len(a) > 1:
            if _absorb_single(b[0], a, axis, along0):
                print(f"[mask-refine] cross-view: single-instance face "
                      f"absorbed into the {len(a)}-instance face's split")
                spans = a
    n = len(spans)
    adj = [set() for _ in range(n)]
    for i in range(n):
        for j in range(i + 1, n):
            a, b = spans[i], spans[j]
            ov = min(a["hi"], b["hi"]) - max(a["lo"], b["lo"])
            shorter = min(a["hi"] - a["lo"], b["hi"] - b["lo"])
            if ov >= 0.5 * shorter:
                adj[i].add(j)
                adj[j].add(i)
    keep = [i for i in range(n) if len(adj[i]) <= 1]
    kept_set = set(keep)
    out, used = [], set()
    for i in keep:
        if i in used:
            continue
        used.add(i)
        s = dict(spans[i])
        # union only a MUTUAL pair: i's single neighbour j is also
        # kept and j's single neighbour is i
        if adj[i]:
            j = next(iter(adj[i]))
            if j in kept_set and j not in used:
                used.add(j)
                p = spans[j]
                s["lo"] = min(s["lo"], p["lo"])
                s["hi"] = max(s["hi"], p["hi"])
                if p["pts"] is not None:
                    s["pts"] = (np.vstack([s["pts"], p["pts"]])
                                if s["pts"] is not None else p["pts"])
                s["ms"] = max(s["ms"], p["ms"])
                np_p = len(p["pts"]) if p["pts"] is not None else 0
                np_s = len(s["pts"]) if s["pts"] is not None else 0
                if np_p > np_s:
                    s["label"] = p["label"]
        out.append(s)
    dropped = n - len(out)
    if dropped:
        print(f"[mask-refine] cross-view: dropped {dropped} coarse "
              f"bridging span(s) -- the finer face's split stands")
    return out


def _merge_spans(spans: list) -> list:
    """Reconcile along-row spans: duplicates merge, seams normalise.

    Two overlap regimes, treated OPPOSITELY (user rule: never merge
    two instances just because their spans overlap):

    * DUPLICATE -- the VLM double-boxed the SAME cabinet. Detected on
      the VLM's OWN pixel boxes (2D IoU >= 0.5), which the mask spans
      cannot provide: masks bleed. Union into one span.
    * MASK BLEED -- two DISTINCT VLM instances whose SAM masks each
      overshoot the cabinet seam by a few centimetres (joined cabinets
      have no visual gap, so each mask edge lands inside the
      neighbour). Both spans survive, cut at the overlap midpoint --
      the seam. The old rule merged ANY overlap > 0.10 m, which
      collapsed the whole row back into one span and the split never
      happened (user report: joined rows stayed joined).
    """

    def _iou(a, b):
        if not a or not b:
            return 0.0
        ix = min(a[2], b[2]) - max(a[0], b[0])
        iy = min(a[3], b[3]) - max(a[1], b[1])
        if ix <= 0 or iy <= 0:
            return 0.0
        inter = ix * iy
        aa = (a[2] - a[0]) * (a[3] - a[1])
        ab = (b[2] - b[0]) * (b[3] - b[1])
        return inter / (aa + ab - inter)

    out = []
    for s in sorted(spans, key=lambda t: t["lo"]):
        dup = next((p for p in out
                    if s["lo"] < p["hi"]
                    and _iou(p.get("pix"), s.get("pix")) >= 0.5), None)
        if dup is not None:
            # same instance double-boxed by the VLM: union
            if len(s["pts"]) > len(dup["pts"]):
                dup["label"] = s["label"]
                dup["pix"] = s.get("pix")
            dup["lo"] = min(dup["lo"], s["lo"])
            dup["hi"] = max(dup["hi"], s["hi"])
            dup["pts"] = np.vstack([dup["pts"], s["pts"]])
            dup["ms"] = max(dup["ms"], s["ms"])
            continue
        for p in out:                      # distinct instances: cut the
            if s["lo"] < p["hi"]:          # mask bleed at the seam
                seam = 0.5 * (p["hi"] + s["lo"])
                p["hi"] = seam
                s["lo"] = seam
        out.append(dict(s))
    return out


def _anchored_top(v: np.ndarray, cell: float = 0.05, dens_frac: float = 0.20,
                  body_frac: float = 0.35, max_gap: int = 2, floor: int = 5):
    """Top of the density-CONNECTED column anchored at the bottom.

    Why not a percentile (or _robust_span): a device body is a column
    of points CONTIGUOUS from the ground up -- every z-slice between
    its floor and its top carries surface. Overhead clutter (cable
    trays, their vertical supports, hanging bundles) is a FLOATING
    layer: dense at its own z, but separated from the body by a
    near-empty gap. A percentile lets the clutter tail drag the top up
    (P99.5 + >0.5% clutter = hijacked); a strong-bin span is blind to
    it too (a tray concentrates into a strong bin). Walking 5cm bins
    upward from the anchor, keeping the run alive only through thin
    gaps (<= max_gap bins), stops at the first real void -- the body's
    top -- wherever the floating layer sits.

    The walk threshold is anchored to the CONFIRMED BODY, not to the
    whole column's median (user report: some heights far above the
    device tops). 3DGS haze DIFFUSES through the whole column -- every
    bin above the cabinet is non-empty, often at 20-40% of the body
    density. Those haze bins drag the all-bin median down, the old
    static threshold fell below the haze density, and the walk
    connected straight through to the floater layer. The body-relative
    threshold (running median of the bins already confirmed as body)
    keeps the bar at the TRUE body density: a haze tail at 30% of the
    body cannot pass `body_frac`, wherever the column median sits.

    Returns the z of the run's upper edge, or None (too few points /
    no dense run).
    """
    v = np.asarray(v, dtype=float)
    if len(v) < floor:
        return None
    lo, hi = float(v.min()), float(v.max())
    # explicit bin COUNT, not arange(stop): arange's ceil((stop-start)/
    # step) also drifts in fp ((3.6-2.45)/0.05 -> 22.999... -> 23
    # edges), and an edge landing a hair BELOW hi makes np.histogram
    # silently DROP every v == hi -- a face sheet sitting exactly on a
    # bin boundary vanishes (_robust_span's span collapses to the far
    # face, _anchored_top's top under-measures). Two extra bins of
    # margin: the last edge is always >= hi + cell; trailing bins are
    # empty-or-real, empty ones never count as strong/dense.
    nb = int(np.floor((hi - lo) / cell)) + 2
    edges = lo + cell * np.arange(nb + 1)
    if len(edges) < 3:
        return None
    hist, _ = np.histogram(v, bins=edges)
    occ = hist[hist > 0]
    if not len(occ):
        return None
    thr0 = max(dens_frac * float(np.median(occ)), 1.0)
    top, gap = None, 0
    body: list[int] = []
    for i, c in enumerate(hist):
        # body-anchored threshold once the walk has confirmed body
        # bins; the static median only STARTS the anchor (bottom bins
        # are body + floor, never haze)
        thr = (max(body_frac * float(np.median(body)), 1.0)
               if body else thr0)
        if c >= thr:
            top, gap = float(edges[i + 1]), 0
            body.append(int(c))
        elif top is not None:
            gap += 1
            if gap >= max_gap:
                break
    return top


def _robust_span(v: np.ndarray, cell: float = 0.05,
                 strong_frac: float = 0.4, floor: int = 3):
    """Strong-bin span of a 1D sample: the extent covered by histogram
    bins holding at least `strong_frac` of the PEAK bin's count.

    Why not percentiles: an open cabinet door contributes a SPREAD-OUT
    tail of points beyond the body (the door panel stands roughly
    perpendicular to the front face, so its points smear over the door's
    full swing range) -- often more than the 2% a P2-P98 cut trims. The
    body's front/back shells concentrate into tall narrow bins while a
    door tail smears into a low plateau, so a peak-relative threshold
    keeps the body and drops the door. Returns None when no bin reaches
    the floor (too few points)."""
    v = np.asarray(v, dtype=float)
    if len(v) < floor:
        return None
    lo, hi = float(v.min()), float(v.max())
    # explicit bin COUNT, not arange(stop): arange's ceil((stop-start)/
    # step) also drifts in fp ((3.6-2.45)/0.05 -> 22.999... -> 23
    # edges), and an edge landing a hair BELOW hi makes np.histogram
    # silently DROP every v == hi -- a face sheet sitting exactly on a
    # bin boundary vanishes (_robust_span's span collapses to the far
    # face, _anchored_top's top under-measures). Two extra bins of
    # margin: the last edge is always >= hi + cell; trailing bins are
    # empty-or-real, empty ones never count as strong/dense.
    nb = int(np.floor((hi - lo) / cell)) + 2
    edges = lo + cell * np.arange(nb + 1)
    if len(edges) < 3:
        return None
    hist, _ = np.histogram(v, bins=edges)
    peak = int(hist.max())
    strong = np.where(hist >= max(strong_frac * peak, floor))[0]
    if not len(strong):
        return None
    return float(edges[strong[0]]), float(edges[strong[-1] + 1])


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
            # vmin=0.0 hardcoded the ground at z~0, but with --boxes
            # input (no ground alignment) every back-projected z can be
            # NEGATIVE: autoscaled vmax < vmin then makes Normalize
            # raise 'minvalue must be less than or equal to maxvalue'
            # and the whole debug render is lost. Clamp: vmin==vmax is
            # tolerated (flat colour), vmin > vmax never reached.
            zmax = float(pts3[:, 2].max())
            ax.scatter(pts3[:, 0], pts3[:, 1], s=2, c=pts3[:, 2],
                       cmap="viridis", vmin=min(0.0, zmax))
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


def _rebuild(f: "OrientedBox", center=None, size=None) -> "OrientedBox":
    """Copy-preserving rebuild: the _apply_* passes REPLACE fitted
    boxes; identity and meta (box_id, device_type, row_id, ...) must
    survive the replacement -- a plain OrientedBox(center=..., size=...)
    dropped them."""
    return OrientedBox(
        center=tuple(center) if center is not None else tuple(f.center),
        size=tuple(size) if size is not None else tuple(f.size),
        yaw=f.yaw, box_id=f.box_id, device_type=f.device_type,
        source=f.source, confidence=f.confidence, row_id=f.row_id,
        meta=dict(f.meta))


def _apply_depth_from_side(instances: list, pts: np.ndarray,
                           seed: "OrientedBox",
                           min_pts: int = 20) -> list[dict]:
    """Correct each split piece's THICKNESS (cross-axis extent) from
    side-view back-projected points.

    The side camera looks ALONG the row axis, so the (depth, height)
    profile of every cabinet projects into the same image region; the
    pool contains ALL pieces' points (lifted WITHOUT the z-buffer) and
    is sliced per piece by along-row span. Two things differ from the
    old oblique-view rule:

    * the view is a true PROFILE (90 deg off the front), where an open
      door sticks out horizontally beyond the cabinet body -- the front
      view cannot separate it (user report), the side view can;
    * the depth estimator is the strong-bin span (_robust_span), which
      drops the door's spread-out tail that a P2-P98 percentile cut
      kept inflating the thickness.

    Along/height stay seed-measured: the side view corrects ONLY the
    thickness and the cross-axis centre of each piece.
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
        m = np.abs(along_all - along_c) <= half + 0.05
        sel = pts[m]
        rec = {"points": int(len(sel))}
        recs.append(rec)
        span = _robust_span(cross_all[m])
        if span is None or len(sel) < min_pts:
            rec["reason"] = "too few side points in along span"
            continue
        c_lo, c_hi = span
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
        # rebuild: seed's along/height, side-measured thickness + centre
        cxy = axis * along_c + cross * mid
        inst["fitted"] = _rebuild(
            f, center=(float(cxy[0]), float(cxy[1]), float(fc[2])),
            size=(float(f.size[0]), depth, float(f.size[2])))
        inst["pts"] = sel
        inst["depth_ok"] = "side"
        rec["accepted"] = True
        rec["depth"] = round(depth, 3)
    return recs


def _pick_piece_top(z_mask: float | None, z_col: float | None,
                    seed_top: float,
                    strict_mask: bool = False) -> tuple[float | None, str | None]:
    """Height-source arbitration per split piece: the SAM-mask z
    (primary) vs the raw-column anchored top (guard + fallback).

    Mask points are the only semantically CLEAN z source (user
    directive after column readings kept scattering high/low): SAM
    isolated this device -- neighbour rows, haze and overhead trays
    are outside the mask, and cloud sparsity inside the body does not
    shorten pixels. Its one historical failure is TRUNCATION (a view
    cut at the frame, a VLM box covering only part of the cabinet,
    the side pass's vertically-short slice -- "very low boxes").
    Column guards exactly that: a mask z grossly below the row's tall
    cabinet (>45% under the seed top) while a SANE column sits well
    above it is far more likely a cut mask than a real half-height
    cabinet standing under a clean column -- the column wins the
    piece. Everything else: mask wins, including short cabinets the
    VLM split out of a mixed row and pieces TALLER than an
    under-measured seed (a broken density walk can never drag a
    column up past its gap). The guard's sane-column ceiling is
    seed_top + 1.20 (was +0.60): the seed itself under-measuring was
    untouchable before -- the column band was cut at seed + 0.60, so
    z_col could never even SEE the true top, let alone pass the cap
    (user report: boxes far below the real height with no rescue).
    The column is ANCHORED (trays/ceiling rejected by the first real
    void), so the extra headroom does not let clutter back in.

    strict_mask (MESH geometry, user directive: box heights must
    strictly follow the SAM mask back-projected MESH points): the
    column guard is DISABLED. In a mesh the trays are physically
    connected to the rack tops -- the anchored walk has NO void to
    stop at and z_col reads the TRAY top, so the very guard meant to
    rescue truncated masks instead OVERRIDES the correct mask value
    with the tray height (mesh runs: heights still wrong). The mask
    over exact mesh points is both semantically and geometrically
    clean -- it wins unconditionally when valid.

    Returns (z_top, source) -- source in {"mask", "col-guard", "col"}
    for the audit trail, or (None, None).
    """
    col_ok = (z_col is not None and 0.50 <= z_col <= 4.50
              and 0.45 * seed_top <= z_col <= seed_top + 1.20)
    if z_mask is not None and 0.50 <= z_mask <= 4.50:
        if (not strict_mask and z_mask < 0.70 * seed_top and col_ok
                and z_col > z_mask + 0.25):
            return z_col, "col-guard"
        return z_mask, "mask"
    if col_ok:
        return z_col, "col"
    return None, None


def _apply_height_and_geom_depth(instances: list, seed: "OrientedBox",
                                 scene: Scene) -> int:
    """Per-piece HEIGHT correction + geometry THICKNESS fallback.

    Height: split pieces inherit the seed's height, but the seed is the
    region fit -- the row's TALLEST cabinet (P99.5 over the whole
    rect). A row the VLM split precisely BECAUSE cabinets differ in
    height (prompt rule) must get each piece's OWN height. The z source
    is the RAW-CLOUD COLUMN under the piece's footprint (along span x
    fitted cross bounds), NOT the piece's mask points: a VLM box that
    covered only part of the cabinet, or the side pass REPLACING pts
    with a profile slice whose mask was vertically short, truncates
    the mask points' z-range and the height collapsed with it (user
    report: some boxes came out very low). The raw column cannot
    under-measure -- the cabinet's full height is in the cloud. The
    bottom stays the seed's bottom (devices stand on the ground).

    Thickness fallback: the side view is the primary thickness source
    (VLM excludes open doors), but when it is missing or rejected a
    piece, the only thing left is the seed's depth -- fit over the
    row's UNION footprint, wrong for every piece of a front-back
    STAGGERED row (user report: the actual thickness cannot be assigned
    per device). The fallback slices the RAW CLOUD by the piece's
    along-span and runs the same strong-bin estimator: the cabinet's
    front/back shells concentrate into strong bins while an open
    door's swing smears into a low plateau -- the statistical door
    exclusion the side view provides explicitly survives approximately.

    Returns the number of geometry-fallback depth corrections.
    """
    if not instances:
        return 0
    yaw = float(seed.yaw)
    axis = np.array([math.cos(yaw), math.sin(yaw)])
    cross = np.array([-math.sin(yaw), math.cos(yaw)])
    sc = np.asarray(seed.center, dtype=float)
    seed_cross_c = float(sc[:2] @ cross)
    seed_half_d = float(np.asarray(seed.size)[1]) / 2.0
    seed_bottom = float(sc[2] - seed.size[2] / 2.0)
    seed_top = float(sc[2] + seed.size[2] / 2.0)
    # raw-cloud device band for the geometry fallback (same band the
    # region fit used: floor texture out, ceiling out). The ceiling cap
    # sits 0.60 above the seed top, not 0.10: a seed whose top came in
    # LOW (bootstrap under-measure) must not
    # chain its error into the piece columns -- the height pass's
    # ANCHORED measurement rejects floating overhead layers (trays,
    # ceiling) anyway, so the headroom is safe.
    P = np.asarray(scene.points, dtype=float)
    band = P[(P[:, 2] > 0.30) & (P[:, 2] <= seed_top + 0.60)] \
        if len(P) else P
    # HEIGHT column band, cut HIGHER than the thickness band: the
    # column must be able to SEE past a LOW seed (the seed top is the
    # row's tallest per the REGION fit -- a broken density walk can
    # hand stageC an under-measured seed, and a band capped at
    # seed + 0.60 then blinds the column exactly when the rescue is
    # needed: z_col can never read the true top, _pick_piece_top's
    # col ceiling (+1.20) is unreachable, and the box stays far below
    # the real height (user report). The extra headroom is safe for
    # HEIGHT: the anchored walk rejects floating overhead layers by
    # the first real void. The THICKNESS band stays at +0.60 -- its
    # strong-bin span must not see trays.
    hband = P[(P[:, 2] > 0.30) & (P[:, 2] <= seed_top + 1.50)] \
        if len(P) else P
    n_geom = 0
    for inst in instances:
        f = inst["fitted"]
        # ---- thickness: geometry fallback for uncorrected pieces ----
        # (FIRST, so the height pass slices the column with the piece's
        # FINAL cross bounds)
        if not inst.get("depth_ok"):
            fc = np.asarray(f.center, dtype=float)
            along_c = float(fc[:2] @ axis)
            half = float(f.size[0]) / 2.0
            if len(band) >= 100:
                along_b = band[:, :2] @ axis
                sel = band[np.abs(along_b - along_c) <= half + 0.05]
                if len(sel) >= 40:
                    span = _robust_span(sel[:, :2] @ cross)
                    if span is not None:
                        c_lo, c_hi = span
                        depth = float(c_hi - c_lo)
                        mid = 0.5 * (c_lo + c_hi)
                        if (0.3 <= depth <= 2.5
                                and abs(mid - seed_cross_c)
                                <= seed_half_d + 0.30):
                            cxy = axis * along_c + cross * mid
                            f = _rebuild(
                                f, center=(float(cxy[0]), float(cxy[1]),
                                          float(f.center[2])),
                                size=(float(f.size[0]), depth,
                                      float(f.size[2])))
                            inst["fitted"] = f
                            inst["depth_ok"] = "geometry"
                            inst["depth"] = round(depth, 3)
                            n_geom += 1
        # ---- height: the piece's own SAM mask points are PRIMARY ----
        # (user directive: after the haze fix the column still reads
        # some pieces high / some low). The mask is the only
        # SEMANTICALLY clean z source: SAM isolated this device, so
        # no neighbour seep (the column's cross+0.15 slice bleeds the
        # taller facing row), no haze above the top, no trays resting
        # ON the top (density-connected, the walk cannot cut them),
        # and no mid-body sparsity gap (a sparse zone terminates a
        # density walk early -- mask pixels do not care about cloud
        # density). The historical mask failure ("very low boxes":
        # truncated z from a view cut / a VLM box covering only part
        # of the cabinet / the vertically-short side slice) is
        # GUARDED, not ignored: a mask z grossly below the row's tall
        # cabinet together with a SANE raw column above it flags
        # truncation and the column wins that piece (_pick_piece_top).
        pts_m = inst.get("pts")
        z_mask = (float(np.percentile(np.asarray(pts_m)[:, 2], 97.5))
                  if pts_m is not None and len(pts_m) >= 20 else None)
        inst["z_mask_top"] = (round(z_mask, 3)
                             if z_mask is not None else None)
        # raw column: the cross-check + fallback + sparse-cloud rescue
        fc = np.asarray(f.center, dtype=float)
        along_c = float(fc[:2] @ axis)
        cross_c = float(fc[:2] @ cross)
        half = float(f.size[0]) / 2.0 + 0.05
        half_d = float(f.size[1]) / 2.0 + 0.15
        col = (hband[(np.abs(hband[:, :2] @ axis - along_c) <= half)
                     & (np.abs(hband[:, :2] @ cross - cross_c) <= half_d)]
               if len(hband) >= 100 else hband)
        if len(col) < 40:
            col = pts_m if pts_m is not None and len(pts_m) >= 20 else None
            col = col if col is not None else []
        z_col = None
        if len(col):
            # ANCHORED column top, not a percentile / strong-bin span:
            # the column's edges bleed a few NEIGHBOUR points past the
            # span seam, and overhead clutter (trays and their supports
            # -- dense enough for a strong bin) floats above the body
            # separated by a near-empty gap. The anchored walk from the
            # ground keeps the density-connected run and stops at the
            # first real void, whichever sits above it.
            zspan = _anchored_top(np.asarray(col)[:, 2])
            z_col = float(zspan) if zspan is not None else None
        inst["z_col_top"] = round(z_col, 3) if z_col is not None else None
        # MESH geometry: strict mask (user directive) -- see
        # _pick_piece_top. GS keeps the column truncation guard.
        z_top, z_src = _pick_piece_top(
            z_mask, z_col, seed_top,
            strict_mask=bool(scene.meta.get("geometry_is_mesh")))
        if z_top is not None:
            h = z_top - seed_bottom
            if abs(h - float(f.size[2])) > 0.05:
                inst["fitted"] = _rebuild(
                    f,
                    center=(float(f.center[0]), float(f.center[1]),
                            seed_bottom + h / 2.0),
                    size=(float(f.size[0]), float(f.size[1]), h))
                inst["height"] = round(h, 3)
                inst["z_src"] = z_src
    return n_geom


def _build_split_pieces(spans: list, seed: "OrientedBox") -> list:
    """Turn along-row spans into SPLIT PIECES of the seed box.

    Each piece keeps the seed's yaw, height, depth and cross/z centres
    -- the front view only guides HOW the seed splits; only the
    along-row extent and position come from the measured span. Piece
    identity/meta copy the seed (the caller's adoption assigns ids).
    """
    yaw = float(seed.yaw)
    axis = np.array([math.cos(yaw), math.sin(yaw)])
    sc = np.asarray(seed.center, dtype=float)
    out = []
    for s in spans:
        length = float(s["hi"] - s["lo"])
        along_mid = 0.5 * (s["lo"] + s["hi"])
        c2 = sc[:2] + axis * along_mid
        piece = OrientedBox(
            center=(float(c2[0]), float(c2[1]), float(sc[2])),
            size=(length, float(seed.size[1]), float(seed.size[2])),
            yaw=yaw, box_id=seed.box_id, device_type=seed.device_type,
            source=BoxSource.AGENT_FIX, confidence=seed.confidence,
            row_id=seed.row_id, meta={**seed.meta, "sam_refined": True})
        if s["pts"] is not None:
            score = 0.55 * float(s["ms"]) + 0.45 * min(len(s["pts"]) / 200.0,
                                                       1.0)
        else:
            score = 0.5      # fallback piece: neutral, never skips the
                            # type-confirm (label unknown, score < 0.6)
        out.append({"score": score, "pts": s["pts"], "fitted": piece,
                    "mask_score": float(s["ms"]), "label": s["label"],
                    "view": "front"})
    return out


def _is_door(label) -> bool:
    """The open-door positive class: the VLM labels its boxes
    "open cabinet door" (user finding: the model DETECTS the open door
    reliably as its own class, but cannot exclude it via a negative
    instruction)."""
    return "door" in str(label or "").lower()


def _is_ladder(label) -> bool:
    """The cable-ladder positive class: a vertical ladder rack / cable
    tray running up beside or behind the device (user report: ladders
    included in front/back device boxes dragged the mask P97.5 height
    to the ladder top -- the ladder gets the SAME subtractive
    treatment as the open door)."""
    l = str(label or "").lower()
    return "ladder" in l or "cable tray" in l


def _is_top_cable(label) -> bool:
    """The top-cable positive class: a bundle of cables running across
    or connected to the TOP of the device (user report: the FRONT view
    grounded them inside the device box -- the back view did not --
    and the multi-view mask union dragged the P97.5 height to the
    cable bundle). Like the door and the ladder, the cable pixels are
    SUBTRACTED from every device back-projection."""
    return "cable" in str(label or "").lower()


def _is_subtractive(label) -> bool:
    """Subtractive classes: labelled structures whose SAM masks are
    pixel-REMOVED from every device back-projection -- the device
    spans, thickness pools and height reads never see them."""
    return _is_door(label) or _is_ladder(label) or _is_top_cable(label)


def _door_union(image: np.ndarray, groups: list, sam: SamPredictorAdapter
                ) -> np.ndarray | None:
    """Pixel union of the SAM masks of every SUBTRACTIVE-class box
    (open cabinet door, cable ladder, top cable) -- the subtractive
    layer for device back-projection. None when the VLM found none
    in this view (the common case)."""
    u: np.ndarray | None = None
    for g in groups:
        if not _is_subtractive(g.get("hypothesis")):
            continue
        group = BoxGroup(tuple(g["bbox"]), g.get("hypothesis", "door"),
                         float(g.get("confidence", 0.5)))
        H, W = image.shape[:2]
        box_pix = group.pixel_box(W, H)
        if not (box_pix[2] - box_pix[0] > 4 and box_pix[3] - box_pix[1] > 4):
            continue
        masks, scores = sam.predict(image, box_pix)
        if not len(masks):
            continue
        m = masks[int(np.argmax(scores))]
        u = m.copy() if u is None else (u | m)
    return u


def _voter_spans(scene: Scene, box: OrientedBox, view: dict, judge,
                 sam: SamPredictorAdapter, out_dir: str | None,
                 audit: dict) -> list[dict]:
    """One face voter (front or back): VLM grounding -> SAM masks ->
    back-projected surface -> along-row spans, per-view reconciled.

    Extracted from refine_box when the back view joined (user report:
    the front aisle is sometimes a narrow corridor -- the back face
    votes too). CLEAN image to the VLM: no wireframe overlay, same
    principle as global grounding. Door masks pixel-subtract from every
    back-projection; the per-view span list carries the VLM pixel box
    for the duplicate-detection _merge_spans does. The audit entry
    (answer, groups, spans) lands in audit["views"] and on disk as
    sam_boxes_<id>_<name>.json.
    """
    voter = view
    verdict = judge.adjudicate_sam_boxes(
        voter["image"], box, voter["name"], png_path=voter["path"])
    groups = verdict.params.get("groups", []) if verdict.params else []
    quality = (verdict.params.get("view_quality", "good")
               if verdict.params else "good")
    va = {"view": voter["name"], "image": voter["path"], "role": "voter",
          "answer": verdict.raw or verdict.detail, "groups": groups,
          "quality": quality, "spans": []}
    if quality == "poor":
        # VLM quality verdict, judged in the SAME call as the grounding
        # (user direction: drop garbage views): a fogged/blurred render
        # gets its boxes dropped however confident they look -- a haze
        # invites hallucinated structure. The audit entry keeps role
        # and groups for review.
        print(f"[mask-refine] {voter['name']} view dropped: "
              f"VLM judges the render quality poor")
        va["role"] = "voter-dropped"
        va["groups"] = groups = []
    H, W = voter["image"].shape[:2]
    yaw = float(box.yaw)
    axis = np.array([math.cos(yaw), math.sin(yaw)])
    along0 = float(np.asarray(box.center, dtype=float)[:2] @ axis)
    # the door class first: its SAM masks form the subtractive layer
    # every device back-projection is pixel-cleaned with
    doors = _door_union(voter["image"], groups, sam)
    spans = []
    for gi, g in enumerate(groups):
        if _is_subtractive(g.get("hypothesis")):
            continue          # door / ladder: subtraction only, never
                             # a span -- box refinement fits devices only
        group = BoxGroup(tuple(g["bbox"]), g.get("hypothesis", "rack"),
                         float(g.get("confidence", 0.5)))
        box_pix = group.pixel_box(W, H)
        if not (box_pix[2] - box_pix[0] > 4 and box_pix[3] - box_pix[1] > 4):
            continue                  # degenerate/absent box
        masks, scores = sam.predict(voter["image"], box_pix)
        best = None
        for mi, (mask, ms) in enumerate(zip(masks, scores)):
            # surface only (z-buffer): the face the VLM grounded; door
            # pixels subtracted so the open door never stretches the
            # span (user report: door interference on box size)
            pts3 = _mask_to_points(scene, box, mask, voter["cam"],
                                    exclude=doors)
            if out_dir:
                _save_sam_debug(voter, box_pix, mask, pts3, box,
                                None, out_dir,
                                f"{box.box_id}_{voter['name']}_g{gi}_m{mi}")
            if len(pts3) < 20:
                continue
            if best is None or ms > best[0]:
                best = (float(ms), pts3)
        if best is None:
            continue
        ms, pts3 = best
        along = pts3[:, :2] @ axis - along0
        lo, hi = np.percentile(along, [2.0, 98.0])
        spans.append({"lo": float(lo), "hi": float(hi), "pts": pts3,
                      "ms": ms, "label": group.hypothesis,
                      "pix": tuple(float(v) for v in box_pix),
                      "view": voter["name"]})
    spans = _merge_spans(spans)
    # clip each span's points to its (possibly seam-cut) extent: the
    # bleed points past the seam belong to the NEIGHBOUR piece, not
    # this one (keeps point counts / scores per-instance honest)
    for s in spans:
        if s["pts"] is None:
            continue
        a = s["pts"][:, :2] @ axis - along0
        s["pts"] = s["pts"][(a >= s["lo"]) & (a <= s["hi"])]
    spans = [s for s in spans
             if (s["hi"] - s["lo"]) >= 0.30 and len(s["pts"]) >= 40]
    va["spans"] = [{"lo": round(s["lo"], 3), "hi": round(s["hi"], 3),
                    "points": len(s["pts"]), "label": s["label"]}
                   for s in spans]
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
    return spans


def refine_box(scene: Scene, box: OrientedBox, judge, sam: SamPredictorAdapter,
               out_dir: str | None = None,
               views: list | None = None) -> tuple[list, dict]:
    """Split-correct one seed box from local VLM grounding + SAM masks.

    Box-only prompting (no points): point placement is a WEAK Qwen3-VL
    skill (user report: prompts mostly off the device despite a clean
    input image), while box grounding is the model's NATIVE task -- and
    the box prompt is SAM's canonical interaction, forgiving of prompt
    error where points are brittle.

    Corrections are applied ON the seed box, never as a free re-fit
    (user decision): the front view's back-projected mask surface only
    guides HOW the seed splits (each visually distinct cabinet its own
    along-row span; the seed's yaw / height / depth / cross centre are
    TRUSTED), and the SIDE view -- the profile along the row axis --
    corrects each piece's THICKNESS.

    Open doors are a positive CLASS, not a negative instruction (user
    finding: the VLM detects "open cabinet door" reliably but cannot
    exclude it from a device box on request): every view's door masks
    pixel-subtract from the device back-projection, so door points
    never enter the span or thickness pools -- box refinement only
    ever fits devices. The strong-bin estimator (_robust_span) drops
    whatever door tail still leaks through as the safety net.

    Returns the list of pieces -- more than one means the front
    grounding split the row, and the caller replaces the old box with
    all of them.
    """
    if views is None:
        views = render_local_views(scene, box, out_dir, judge=judge)
    audit = {"box_id": box.box_id, "views": [], "accepted": False}
    if not views or not sam.available:
        audit["reason"] = "no local GS views or SAM checkpoint"
        return [], audit
    voters = [v for v in views if v["name"] in ("front", "back")]
    # NO side-view fallback as a span voter: the side camera looks
    # ALONG the row, so its masks span the CROSS axis -- spans from it
    # would split the row by THICKNESS (an earlier bug). With both
    # faces dropped by the quality gate there is simply no span vote.
    side = next((v for v in views if v["name"] == "side"), None)
    yaw = float(box.yaw)
    axis = np.array([math.cos(yaw), math.sin(yaw)])
    sc = np.asarray(box.center, dtype=float)
    along0 = float(sc[:2] @ axis)

    # ---- pass 1 (front + back): grounding -> SAM surface -> spans ----
    # TWO face voters now (user report: the front aisle is sometimes a
    # NARROW corridor -- poor standoff, foreshortened row ends, weak
    # render -- while the back side is open and photographs cleanly).
    # Each face grounds independently; the SAME cabinet's spans union
    # across views (_merge_cross_view), a cabinet legible from only
    # one face still splits the row. CLEAN image to the VLM: the
    # wireframe overlay (prompt_image) is the Stage-A box, which is
    # often oversized/misplaced -- the VLM anchors on the frame
    # instead of the device. Same principle as global grounding: no
    # box prompts in the input image.
    spans = []
    for voter in voters:
        spans.extend(_voter_spans(scene, box, voter, judge, sam,
                                  out_dir, audit))
    spans = _merge_cross_view(spans, axis, along0)
    front_ok = bool(spans)
    if not front_ok:
        # no usable front division: the seed stays WHOLE and is still
        # eligible for the side-view thickness correction below
        spans = [{"lo": -float(box.size[0]) / 2.0,
                  "hi": float(box.size[0]) / 2.0,
                  "pts": None, "ms": 0.5, "label": "unknown"}]

    # ---- pass 2 (side): thickness correction per piece ----
    # The side camera looks ALONG the row: every cabinet's (depth,
    # height) profile projects into the same image region, so the mask
    # points (lifted WITHOUT the z-buffer) form one pool containing all
    # pieces, sliced per piece by along span. The VLM prompt already
    # excludes open doors -- face-on in this profile -- and the strong-
    # bin estimator drops whatever door tail still leaks through.
    depth_ok = False
    if side is not None:
        verdict = judge.adjudicate_sam_boxes(
            side["image"], box, side["name"], png_path=side["path"])
        groups = verdict.params.get("groups", []) if verdict.params else []
        quality = (verdict.params.get("view_quality", "good")
                   if verdict.params else "good")
        dva = {"view": side["name"], "image": side["path"],
               "role": "depth_profile", "groups": groups,
               "quality": quality, "instances": []}
        if quality == "poor":
            # same VLM quality verdict as the face voters (user
            # direction: drop garbage views): a fogged side render must
            # not feed the thickness pool -- the seed/geometry fallback
            # takes over instead
            print(f"[mask-refine] {side['name']} view dropped: "
                  f"VLM judges the render quality poor")
            dva["role"] = "depth_profile-dropped"
            dva["groups"] = groups = []
        H, W = side["image"].shape[:2]
        # the side view is where an open door sticks out HORIZONTALLY
        # beyond the body -- subtract its mask before any thickness
        # point enters the pool (belt and braces on top of the
        # strong-bin estimator)
        doors = _door_union(side["image"], groups, sam)
        pool, pool_ms = [], 0.0
        for gi, g in enumerate(groups):
            if _is_subtractive(g.get("hypothesis")):
                continue          # subtraction only, never a pool
            group = BoxGroup(tuple(g["bbox"]), g.get("hypothesis", "rack"),
                             float(g.get("confidence", 0.5)))
            box_pix = group.pixel_box(W, H)
            if not (box_pix[2] - box_pix[0] > 4
                    and box_pix[3] - box_pix[1] > 4):
                continue                  # degenerate/absent box
            masks, scores = sam.predict(side["image"], box_pix)
            best = None
            for mi, (mask, ms) in enumerate(zip(masks, scores)):
                # NO z-buffer: from along the row, every piece overlaps
                # in projection -- all of them must contribute points
                pts3 = _mask_to_points(scene, box, mask, side["cam"],
                                      z_buffer=False, exclude=doors)
                if out_dir:
                    _save_sam_debug(side, box_pix, mask, pts3, box,
                                    None, out_dir,
                                    f"{box.box_id}_{side['name']}_g{gi}_m{mi}")
                if len(pts3) < 20:
                    continue
                if best is None or ms > best[1]:
                    best = (pts3, float(ms))
            if best is not None:
                pool.append(best[0])
                pool_ms = max(pool_ms, best[1])
        if pool:
            instances = _build_split_pieces(spans, box)
            recs = _apply_depth_from_side(
                instances, np.vstack(pool), box)
            dva["instances"] = recs
            depth_ok = any(r.get("accepted") for r in recs)
            for inst, rec in zip(instances, recs):
                if not rec.get("accepted"):
                    continue    # thickness rejected: seed depth stands
                inst["view"] = inst["view"] + f'+{side["name"]}(depth)'
                inst["mask_score"] = 0.5 * (inst["mask_score"] + pool_ms)
                inst["score"] = min(inst["score"] + 0.08, 1.0)
        else:
            instances = _build_split_pieces(spans, box)
        if out_dir:
            try:
                with open(os.path.join(
                        out_dir, f"sam_boxes_{box.box_id}_{side['name']}.json"),
                          "w", encoding="utf-8") as f:
                    json.dump(dva, f, ensure_ascii=False, indent=2,
                              default=json_default)
            except OSError:
                pass
        audit["views"].append(dva)
    else:
        instances = _build_split_pieces(spans, box)

    # ---- pass 3: per-piece height + geometry thickness fallback ----
    # Heights are NEVER seed-inherited past this point when the piece
    # has its own points, and a piece the side view left uncorrected
    # gets its thickness from the raw cloud instead of the seed's
    # union-footprint depth (staggered rows, user report).
    _apply_height_and_geom_depth(instances, box, scene)

    if not front_ok and not depth_ok:
        # nothing was measured: keep the seed untouched
        audit["reason"] = "no front span and no side depth measurement"
        return [], audit

    instances.sort(key=lambda e: e["score"], reverse=True)
    audit.update({"accepted": True,
                  "instances": [
                      {"score": round(e["score"], 4), "view": e["view"],
                       "points": len(e["pts"]) if e["pts"] is not None else 0,
                       "sam_score": round(e["mask_score"], 4),
                       "label": e["label"], "box": e["fitted"].to_dict()}
                      for e in instances]})
    return instances, audit
