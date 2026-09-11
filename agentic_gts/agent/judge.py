"""VLM adjudicator for the agent loop.

The VLM is a *discriminator*, not a generator of geometry. It is asked
yes/no and multiple-choice questions about rendered evidence, and returns a
discrete verdict. Precise coordinates always come from the geometry tools.

Two backends:
  - "qwen"   : Qwen3-VL-8B served via an OpenAI-compatible endpoint.
  - "local"  : in-process HuggingFace transformers model. Loaded ONCE on the
               first adjudication and kept in memory; the agent loop then
               only pays per-call inference. Use when you don't want a
               separate server process (costs ~1-2 min model load at startup
               and the GPU memory is held for the whole pipeline run).
  - "mock"   : deterministic rule-based fallback (no network), so the whole
               pipeline runs without any model. This is also the reliability
               floor / baseline.
"""
from __future__ import annotations

import base64
import io
import json
import os
from dataclasses import dataclass

import numpy as np
import requests


@dataclass
class Verdict:
    action: str          # one of the discrete ActionType strings
    params: dict = None  # discrete args (n, width_unit, ...)
    confidence: float = 0.5
    detail: str = ""
    raw: str = ""
    png_path: str = None  # where the evidence image was persisted (if enabled)


# ---------- image rendering helpers ----------

def render_topdown_image(stage_points: np.ndarray, boxes, extent: float = 1.0,
                         size: int = 768, gs_ply: str | None = None,
                         overlay: str = "wire3d",
                         quality_out: dict | None = None,
                         slots: tuple[str, ...] | None = None,
                         gs_cams: str | None = None) -> np.ndarray:
    """Render the local evidence image the VLM adjudicates on.

    If the scene comes from a 3DGS model (gs_ply set and a CUDA rasterizer
    is available), this is a TRUE Gaussian-splat composite of up to THREE
    views (front / side / oblique-top), each with the full 3D wireframe
    overlaid and unrelated gaussians hidden -- the fine-detail counterpart
    to the god-view's coarse positioning. Otherwise it falls back to the
    2D scatter density view.

    extent controls the surrounding context: the camera backs off until
    the box union + extent fits, and gaussians within ISOLATE_MARGIN of
    the boxes stay visible (neighbouring racks, the adjacent aisle) --
    judging overhang/misfit needs the row rhythm, not the box alone.

    Quality-aware view selection: 3DGS render quality is anisotropic (a
    view that extrapolates away from the training cameras blurs / grows
    floaters), so each of the three slots is rendered at SEVERAL nearby
    azimuths, scored no-reference (sharpness / coverage / speckle, see
    gs_render.view_quality), and the best candidate wins. Role coverage
    is preserved by construction: candidates only jitter WITHIN a slot's
    role (front / side / oblique elevations stay distinct). If
    `quality_out` is given it is filled with per-slot scores (only the
    slots actually rendered) -- callers gate VLM verdict confidence on it
    (a bad render must not produce a confident delete).

    slots: optional subset to render, e.g. ("oblique",) or ("front",
    "side"). The two-phase refine asks ONE question per render -- yaw
    from the near-top-down view, extents from the horizontal views --
    and each phase's prompt describes exactly what its image shows.
    None renders all three.
    """
    # ---- 3DGS true render (preferred when available) ----
    if gs_ply and boxes:
        try:
            from agentic_gts.tools.gs_io import (read_gaussian_ply,
                                                 read_colmap_views)
            from agentic_gts.output.gs_render import (make_local_cam,
                                                      render_slot_candidates)
            gs = read_gaussian_ply(gs_ply)
            # training poses (COLMAP): when available, the candidate view
            # scores blend a pose-based trust (distance to the trained
            # ray distribution) into the image quality -- see
            # gs_render.train_view_trust. None keeps the pure image score.
            train_views = (read_colmap_views(gs_cams)
                           if gs_cams else None)
            # THREE view slots (tiled into one image), each with azimuth
            # candidates scored by render quality: front shows door/panel
            # detail, side shows the row context and depth, oblique (~70
            # deg tilt, near-top-down) shows the top face with minimal
            # perspective foreshortening so the VLM can measure the
            # row-direction THICKNESS of a sandwiched rack -- at 55 deg
            # the row depth compresses and adjacent racks fuse. front and
            # side ALSO carry opposite-side candidates (azim +180): a
            # fragment box hugging the BACK of a device has its 'front'
            # pointing INTO the device body, so every same-side candidate
            # is fully occluded by it -- the eligible filter (visibility
            # >= 0.25) then flips the slot to the visible outer surface.
            # A fixed single azimuth per slot gambles on that exact
            # direction being well-trained; picking the best of ~3-6
            # costs milliseconds per extra rasterization.
            # STEEP FALLBACK TIER (front/side): two facing full-height
            # rows with a ~0.5 m aisle leave NO horizontal sightline to
            # either inner face -- the horizontal camera survives only by
            # being pulled far back and lifted, which renders blurry
            # (extrapolated view). When the primary tier comes out weak,
            # the slot re-renders from ~58 deg over the aisle: near
            # camera, top face + aisle context, sharp. Normal-width
            # aisles never trigger it, so door/panel detail is kept
            # wherever it is actually reachable.
            # Mild ceiling cut: remove everything above the boxes' top so
            # cable trays / ceiling clutter near the rack do not pile up at
            # the top of the image. The cut dips ~8cm INTO the rack top
            # (user-approved 5-10cm sacrifice) -- flush overhead structure
            # (trays, ducts) otherwise survives a cut at exactly the top.
            top = max(b.center[2] + b.size[2] / 2.0 for b in boxes)
            cut_z = top - 0.08
            # context isolation margin: keeps the 1-2 neighbouring racks
            # and the adjacent aisle visible (row rhythm is evidence for
            # overhang / one-or-many judgements), drops the rest of the
            # room so it cannot occlude what is being adjudicated
            iso_margin = extent + 1.0
            all_slots = {
                "front": (((18.0, 0.0), (18.0, 12.0), (18.0, -12.0),
                           (18.0, 180.0), (18.0, 168.0), (18.0, 192.0)),
                          ((58.0, 0.0), (58.0, 12.0), (58.0, -12.0),
                           (58.0, 180.0))),
                "side": (((18.0, 90.0), (18.0, 76.0), (18.0, 104.0),
                          (18.0, 270.0), (18.0, 256.0), (18.0, 284.0)),
                         ((58.0, 90.0), (58.0, 104.0), (58.0, 270.0))),
                "oblique": (((70.0, 35.0), (72.0, 48.0), (66.0, 22.0)), ()),
            }
            # optional subset (two-phase refine renders one question's
            # evidence per image, so each prompt describes exactly the
            # views it can see)
            sel = {k: v for k, v in all_slots.items()
                   if slots is None or k in slots}
            views, quality, names = [], {}, []
            for name, (cands, fb) in sel.items():
                img, q, chosen = render_slot_candidates(
                    gs, boxes,
                    lambda e, a: make_local_cam(boxes, extent=extent * 2,
                                                elev_deg=e, azim_deg=a),
                    cands, cut_z=cut_z, overlay=overlay,
                    iso_margin=iso_margin, fallback_candidates=fb,
                    train_views=train_views)
                if img is None:
                    continue
                views.append(img)
                names.append(name)
                quality[name] = dict(q, view=[chosen[0], chosen[1]])
            if views:
                if quality_out is not None:
                    quality_out.update(quality)
                return _tile_views(views, labels=names)
        except Exception as e:
            print(f"[gs][local] true render failed ({type(e).__name__}: {e}) "
                  f"-> scatter fallback")
    if quality_out is not None:
        quality_out["mode"] = "scatter_fallback"

    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    # extract region around the candidate
    if boxes:
        all_c = np.vstack([np.asarray(b.center[:2]) for b in boxes])
        c = all_c.mean(axis=0)
    else:
        c = stage_points.mean(axis=0) if len(stage_points) else np.zeros(2)
    lo = c - extent
    hi = c + extent
    m = ((stage_points[:, 0] >= lo[0]) & (stage_points[:, 0] <= hi[0]) &
         (stage_points[:, 1] >= lo[1]) & (stage_points[:, 1] <= hi[1]))
    # lift the ceiling locally too: cut at the candidate's own top, so
    # overhead structure never buries the box being adjudicated
    cut = _render_cut_z(stage_points, boxes)
    if np.isfinite(cut):
        m &= stage_points[:, 2] < cut
    pts = stage_points[m][:, :2]

    fig, ax = plt.subplots(figsize=(4, 4), dpi=size // 4)  # 768px @ size=768
    if len(pts):
        ax.scatter(pts[:, 0], pts[:, 1], s=0.5, alpha=0.6, c="steelblue")
    ax.axis("equal")
    ax.set_xlim(lo[0], hi[0])
    ax.set_ylim(lo[1], hi[1])
    for b in boxes:
        cs = b.corners_2d()
        poly = plt.Polygon(cs, fill=False, edgecolor="red", linewidth=1.5)
        ax.add_patch(poly)
        if overlay == "wire3d_axes":
            # keep the prompt-image contract in the scatter fallback too:
            # green = local +x (length), blue = local +y (depth), drawn
            # from the footprint centre -- the same convention as the GS
            # render's top-face arrows
            c = np.asarray(b.center[:2], dtype=float)
            rot = b.rotation[:2, :2]
            # arrow length: proportional to the box, but always inside the
            # view (a deep box would otherwise push its +y arrow past the
            # crop and get clipped away entirely)
            ax_l = min(max(b.size[0], 0.2) * 0.55, extent * 0.8)
            ax_w = min(max(b.size[1], 0.2) * 0.55, extent * 0.8)
            ax.annotate("", xy=c + rot[:, 0] * ax_l, xytext=c,
                        arrowprops=dict(arrowstyle="-|>", color="lime",
                                        lw=2.5))
            ax.annotate("", xy=c + rot[:, 1] * ax_w, xytext=c,
                        arrowprops=dict(arrowstyle="-|>", color="dodgerblue",
                                        lw=2.5))
    ax.set_xticks([]); ax.set_yticks([])
    buf = io.BytesIO()
    fig.savefig(buf, format="png", bbox_inches="tight")
    plt.close(fig)
    buf.seek(0)
    return np.array(plt.imread(buf))  # HxWx4


# Description of the local evidence image shared by all per-box prompts.
# MUST match what render_topdown_image actually produces (see _tile_views):
# a three-view composite, NOT a bird's-eye view.
_LOCAL_VIEW_DESC = (
    "The image is a composite of THREE views of the same candidate region, "
    "tiled side by side, each labeled above the panel: 'front' (the rack's "
    "front face: doors, panels, LEDs), 'side' (view along the row: depth and "
    "neighbouring racks), and 'oblique' (near-top-down view: the top face "
    "with little perspective foreshortening -- use it to measure the box's "
    "row-direction thickness against its neighbours). Red wireframes mark "
    "the candidate box(es); each wireframe is "
    "the full 3D box, not just its top. Each panel's exact viewing angle is "
    "auto-selected for the clearest render, so panels may be rotated "
    "slightly within their role, a front/side panel may be rendered "
    "from the OPPOSITE side of the box when the primary side is occluded "
    "by the device body, and in a NARROW aisle between two facing rows "
    "the front panel is rendered from a steep angle above the aisle "
    "(top face + aisle context instead of door detail) -- judge the "
    "wireframe against whichever device surface is visible. "
    "(Fallback rendering without a GPU rasterizer: a SINGLE top-down view "
    "of the same region, with the axes arrows drawn on the footprint.)"
)


def _tile_views(views: list, labels=("front", "side", "oblique")) -> np.ndarray:
    """Tile multiple single-view renders into ONE composite image.

    One image per VLM call keeps the (OpenAI-compatible / transformers)
    API payload unchanged -- a single image input -- while showing the box
    from several viewpoints. Views are laid out horizontally with a small
    label strip above each so the VLM (and auditors reading the saved
    evidence) can tell them apart.
    """
    from PIL import Image, ImageDraw
    tiles = []
    label_h = max(18, views[0].shape[0] // 25)   # scale with tile resolution
    for i, v in enumerate(views):
        arr = (np.clip(v, 0, 1) * 255).astype(np.uint8)
        if arr.shape[2] == 4:
            arr = arr[:, :, :3]
        im = Image.fromarray(arr)
        strip = Image.new("RGB", (im.width, label_h), (0, 0, 0))
        d = ImageDraw.Draw(strip)
        d.text((10, label_h // 6), labels[i % len(labels)],
               fill=(255, 255, 255))
        tile = Image.new("RGB", (im.width, im.height + label_h), (0, 0, 0))
        tile.paste(strip, (0, 0))
        tile.paste(im, (0, label_h))
        tiles.append(tile)
    gap = max(4, label_h // 4)
    W = sum(t.width for t in tiles) + gap * (len(tiles) - 1)
    H = max(t.height for t in tiles)
    out = Image.new("RGB", (W, H), (0, 0, 0))
    x = 0
    for t in tiles:
        out.paste(t, (x, 0))
        x += t.width + gap
    return np.asarray(out).astype(np.float32) / 255.0


def _auto_ceiling_z(z: np.ndarray, gap: float = 0.3, frac: float = 0.05) -> float:
    """Find the z where a separated ceiling slab begins; inf if none.

    A top-down render must exclude the ceiling: it is the highest, fully
    covering layer and would visually bury every device below it. A ceiling
    exists iff the topmost dense z-segment is separated from the structure
    below by a near-empty vertical gap (racks ~2.4m, ceiling 3-5m). Tall
    structures that reach the ceiling leave no gap and are kept whole.
    """
    if len(z) < 200:
        return float("inf")
    edges = np.arange(z.min(), z.max() + 0.1, 0.1)
    if len(edges) < 5:
        return float("inf")
    hist, _ = np.histogram(z, bins=edges)
    pos = hist[hist > 0]
    if len(pos) == 0:
        return float("inf")
    # median-based (NOT max-based): a very dense ceiling must not raise the
    # bar so high that the sparser device layer reads as an empty gap
    thr = max(20.0, 0.3 * float(np.median(pos)))
    occ = hist >= thr
    segs = []
    s = None
    for i, v in enumerate(occ):
        if v and s is None:
            s = i
        if not v and s is not None:
            segs.append((s, i))
            s = None
    if s is not None:
        segs.append((s, len(occ)))
    if len(segs) >= 2:
        top_start, below_end = segs[-1][0], segs[-2][1]
        if (top_start - below_end) * 0.1 >= gap:
            return float(edges[below_end])
    return float("inf")


def _render_cut_z(points: np.ndarray, boxes, margin: float = 0.3,
                  cut_above_rack: bool = True) -> float:
    """Ceiling/overhead cut for top-down renders.

    Trusted box heights win: anything above the tallest device top is
    overhead structure (cable trays, pipes, ceiling) by definition. The cut
    sits at the highest box top + `margin`; overhead structure above it is
    removed before rasterization so it cannot occlude the racks in a
    top-down view.

    `margin` may be NEGATIVE to cut slightly INTO the rack tops. That is
    deliberate: cable trays / ducts often sit flush against the rack top and
    survive a cut at `top + small`, so dipping a little below the rack top
    removes them. Losing a sliver of the rack's top face is acceptable for a
    global layout view. Falls back to the z-histogram gap when no boxes are
    given.
    """
    if boxes:
        top = max(b.center[2] + b.size[2] / 2.0 for b in boxes)
        if cut_above_rack:
            return top + margin      # margin < 0 -> cut into the rack tops
        return top
    return _auto_ceiling_z(points[:, 2])


def _render_cut_z_low(boxes, lift: float = 0.35) -> float:
    """Floor cut for the top-down view: remove ground-level gaussians.

    3DGS floor splats are wide and have thickness -- their centres sit near
    z=0 but the gaussian body extends well above, so cutting at 'rack bottom
    - small' only trims a sliver. The floor texture then still occludes the
    rack footprints. To remove it, lift the lower bound ABOVE the rack bottom
    by `lift` (default 0.35m): ground splats are dropped while the racks keep
    everything above the bottom quarter. This is a discovery view, so losing
    the rack's lowest sliver is acceptable -- the top/side that identifies a
    rack remains.
    """
    if boxes:
        return min(b.center[2] - b.size[2] / 2.0 for b in boxes) + lift
    return float("-inf")


def render_godview_png(points: np.ndarray, boxes, max_points: int = 250_000,
                       dpi: int = 130, gs_ply: str | None = None) -> bytes:
    """Full-scene bird's-eye render for the god-view pass.

    With a 3DGS input (gs_ply set) this is a TRUE Gaussian-splat render from
    an oblique virtual camera + numbered box wireframes. Falls back to the
    height-colored scatter plot when no CUDA rasterizer is available.
    Returns PNG bytes.
    """
    # ---- 3DGS true render (preferred when available) ----
    if gs_ply:
        try:
            from agentic_gts.tools.gs_io import read_gaussian_ply
            from agentic_gts.output.gs_render import (make_godview_cam,
                                                      render_gs_view, png_bytes)
            gs = read_gaussian_ply(gs_ply)
            # Cut slightly INTO the rack tops (negative margin): cable trays
            # often sit flush against the rack top and survive a cut at
            # Cut INTO the rack tops (negative margin): overhead structure
            # (cable trays, ducts, lamps) sits above the rack top and must
            # not occlude the top-down discovery view. Dip ~0.45m below the
            # highest rack top so lamps and tray residue are removed too.
            # Losing the rack's top quarter is acceptable here -- this is a
            # find-gaps view, the top/side identity of a rack is unaffected.
            cut = _render_cut_z(points, boxes, margin=-0.45)
            cut_low = _render_cut_z_low(boxes)
            # True top-down camera. Height is auto-derived from the room
            # footprint (a god-view overlooks the whole layout), so it sits
            # well above every rack. Overhead structure is removed by the
            # ceiling Z-cut, so a high eye cannot re-introduce the ceiling.
            # Ground texture / reflections sit below cut_low and must not
            # figure into the render (they occlude the rack footprints).
            pts_for_cam = points[points[:, 2] < cut] if np.isfinite(cut) else points
            if np.isfinite(cut_low):
                pts_for_cam = pts_for_cam[pts_for_cam[:, 2] > cut_low]
            if len(pts_for_cam) < 100:
                pts_for_cam = points
            cam = make_godview_cam(pts_for_cam, boxes, nadir=True)
            # wire3d (same as the local view): the VLM sees each
            # candidate's FULL claimed volume -- both rings + vertical
            # edges -- not just the floor footprint. Under perspective the
            # bottom ring leans slightly outward for off-centre racks,
            # which is the intended 3D depth cue.
            img = render_gs_view(gs, boxes, cam, cut_z=cut, cut_z_low=cut_low,
                                 overlay="wire3d")
            if img is not None:
                print(f"[gs][godview] true 3DGS render "
                      f"({len(gs)} gaussians, top-down view)")
                return png_bytes(img)
        except Exception as e:
            print(f"[gs][godview] true render failed ({type(e).__name__}: {e}) "
                  f"-> scatter fallback")

    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.patches import Polygon as MplPolygon

    pts = points
    # lift the ceiling: everything above the tallest trusted box top is
    # ceiling/overhead structure and would bury the layout below it
    cut = _render_cut_z(pts, boxes)
    if np.isfinite(cut):
        pts = pts[pts[:, 2] < cut]
    if len(pts) > max_points:
        sel = np.random.default_rng(0).choice(len(pts), max_points, replace=False)
        pts = pts[sel]

    fig, ax = plt.subplots(figsize=(11, 9), dpi=dpi)
    sc = ax.scatter(pts[:, 0], pts[:, 1], c=pts[:, 2], s=0.3, cmap="viridis",
                    alpha=0.45, linewidths=0, rasterized=True)
    fig.colorbar(sc, ax=ax, label="height z (m)", shrink=0.7)
    for i, b in enumerate(boxes):
        color = "#d93025" if b.confidence.value != "low" else "#f2a900"
        ax.add_patch(MplPolygon(b.corners_2d(), closed=True, fill=False,
                                edgecolor=color, linewidth=1.4))
        ax.text(b.center[0], b.center[1], str(i), fontsize=7, ha="center",
                va="center", color=color, weight="bold")
    ax.set_aspect("equal")
    ax.set_title(f"top-down view | {len(boxes)} candidate boxes (numbered)", fontsize=10)
    ax.set_xlabel("x (m)"); ax.set_ylabel("y (m)")
    buf = io.BytesIO()
    fig.savefig(buf, format="png", bbox_inches="tight")
    plt.close(fig)
    return buf.getvalue()


def _extract_json(text: str):
    """Best-effort JSON object extraction from a VLM reply.

    The prompts ask for reasoning sentences FIRST and the JSON on the LAST
    line, so multiple {...} spans can appear -- prefer the last
    well-formed one. Balanced-brace scanning keeps NESTED replies intact:
    a regex like {[^{}]*} only ever matches the innermost objects (e.g.
    one region dict instead of the {"regions": [...]} wrapper around
    them), silently losing the answer key.
    """
    top, inner = [], []          # top-level spans vs nested ones
    stack = []
    for i, ch in enumerate(text):
        if ch == "{":
            stack.append(i)
        elif ch == "}" and stack:
            j = stack.pop()
            (top if not stack else inner).append(text[j:i + 1])
    for cands in (top, inner):
        for c in reversed(cands):
            try:
                return json.loads(c)
            except json.JSONDecodeError:
                continue
    return None


def _png_to_b64(png_bytes: bytes) -> str:
    return base64.b64encode(png_bytes).decode("ascii")


def _render_split_views(scene, box, extent: float = 1.2):
    """Two-panel evidence image for the row-split question: the row box's
    FRONT face (azim ~0) and BACK face (azim ~180), each best-of nearby
    azimuths with the same quality/visibility selection as the local
    view. Both long faces are needed because a joined row's cabinet
    boundaries are door seams -- often clearer on one face than the
    other. Falls back to the single top-down view where the row's
    internal gaps are still visible.

    Returns (image, quality_dict)."""
    gs_ply = scene.meta.get("gs_ply")
    if gs_ply:
        try:
            from agentic_gts.tools.gs_io import read_gaussian_ply
            from agentic_gts.output.gs_render import (make_local_cam,
                                                      render_slot_candidates)
            gs = read_gaussian_ply(gs_ply)
            cut_z = box.center[2] + box.size[2] / 2.0 - 0.08
            iso = extent + 1.0
            faces = {
                "front": ((18.0, 0.0), (18.0, 12.0), (18.0, -12.0)),
                "back": ((18.0, 180.0), (18.0, 168.0), (18.0, 192.0)),
            }
            views, quality = [], {}
            for name, cands in faces.items():
                img, q, chosen = render_slot_candidates(
                    gs, [box],
                    lambda e, a: make_local_cam([box], extent=extent * 2,
                                                elev_deg=e, azim_deg=a),
                    cands, cut_z=cut_z, overlay="wire3d", iso_margin=iso)
                if img is None:
                    continue
                views.append(img)
                quality[name] = dict(q, view=[chosen[0], chosen[1]])
            if views:
                return _tile_views(views, labels=list(quality)), quality
        except Exception as e:
            print(f"[gs][split] true render failed ({type(e).__name__}: {e}) "
                  f"-> top-down fallback")
    img = render_topdown_image(scene.points, [box], extent=extent,
                               gs_ply=gs_ply)
    return img, {}


class VLMJudge:
    def __init__(self, backend: str = "mock",
                 model: str | None = None,
                 api_base: str | None = None, api_key: str | None = None,
                 timeout: int = 60,
                 thinking_model: str | None = None,
                 thinking_api_base: str | None = None,
                 thinking_timeout: int = 300):
        self.backend = backend
        self.model = (model or os.environ.get("VLM_MODEL") or
                      "Qwen/Qwen3-VL-8B-Instruct")
        self.api_base = (api_base or os.environ.get("VLM_API_BASE") or
                         "http://127.0.0.1:8000/v1")
        self.api_key = api_key or os.environ.get("VLM_API_KEY", "EMPTY")
        self.timeout = timeout
        # ---- escalation tier (optional thinking checkpoint) ----
        # When set, verdicts whose evidence render scored below the
        # quality floor are re-asked on the thinking model (and the
        # god-view audit runs on it directly): multi-step visual
        # reasoning is exactly where thinking checkpoints gain, and the
        # 1.5-5x latency is paid only on the (few) hard cases.
        self.thinking_model = (thinking_model or
                               os.environ.get("VLM_THINKING_MODEL"))
        self.thinking_api_base = (thinking_api_base or
                                  os.environ.get("VLM_THINKING_API_BASE") or
                                  self.api_base)
        self.thinking_timeout = thinking_timeout
        self._local_model = None   # lazy: (processor, model), loaded once
        self._thinking_local_model = None  # lazy second slot, thinking only
        self.record_path = None    # if set, append JSONL records of adjudications
        self.evidence_dir = None   # if set, persist adjudication images here
        self._render_cache = {}    # render cache: box-geometry key -> image
        self._render_quality = {}  # same key -> per-slot view quality dict

    @staticmethod
    def _render_cache_key(boxes) -> tuple:
        """Cache key from the boxes' geometry (not identity): a retry after
        a rollback restores identical geometry and must reuse the render."""
        parts = []
        for b in boxes:
            c = np.round(np.asarray(b.center, dtype=float), 2)
            s = np.round(np.asarray(b.size, dtype=float), 2)
            parts.append((b.box_id, tuple(c), tuple(s), round(float(b.yaw), 3)))
        return tuple(parts)

    def _render_cached(self, scene, boxes):
        """render_topdown_image with caching: the agent loop re-decides the
        same issue up to max_retries times, and after a rollback the box
        geometry is identical -- the three-view composite (3 rasterizations)
        is then needlessly recomputed. Per-slot view quality is cached
        alongside so verdicts can be confidence-gated on render quality."""
        key = self._render_cache_key(boxes)
        if key not in self._render_cache:
            q = {}
            self._render_cache[key] = render_topdown_image(
                scene.points, boxes, gs_ply=scene.meta.get("gs_ply"),
                quality_out=q, gs_cams=scene.meta.get("gs_cams"))
            self._render_quality[key] = q
        return self._render_cache[key]

    @staticmethod
    def _quality_floor(quality: dict | None) -> float:
        """Worst per-slot view score (1.0 when unknown, e.g. scatter
        fallback where 'quality' is not a trained-view property). Callers
        cap the VLM verdict confidence when this is low: a blurry /
        floater-ridden / OCCLUDED evidence image must not produce a
        confident delete. A slot whose box is mostly hidden behind a wall
        or a flush neighbour (visibility < 0.3) counts as untrustworthy
        even when the image itself is sharp -- a sharp wall is still a
        wall. Likewise a camera still embedded in structure after the
        pullback (clearance < 0) renders a blurry wall of near splats and
        is untrustworthy however sharp the rest of the frame looks."""
        if not quality:
            return 1.0
        eff = []
        for v in quality.values():
            if not isinstance(v, dict):
                continue
            s = float(v.get("score", 1.0))
            vis = v.get("visibility")
            if vis is not None and float(vis) < 0.3:
                s = min(s, 0.3)
            clr = v.get("clearance")
            if clr is not None and float(clr) < 0.0:
                s = min(s, 0.3)
            eff.append(s)
        return min(eff) if eff else 1.0

    def _gate_quality(self, v: Verdict, boxes, quality: dict | None = None) -> dict:
        """Cap a verdict's confidence when its evidence render scored low
        (in-place on the Verdict). A blurry / floater-ridden image must not
        yield a confident delete -- the FP deletion threshold (0.6) then
        blocks the deletion, degrading it to the safer shrink path.
        Returns the quality dict so callers can pass it to _record."""
        q = quality if quality is not None else \
            self._render_quality.get(self._render_cache_key(boxes), {})
        qfloor = self._quality_floor(q)
        if qfloor < 0.35:
            v.confidence = min(v.confidence, 0.5)
            v.detail = f"{v.detail or ''} [low render quality {qfloor:.2f}]"
        return q

    def _should_escalate(self, quality: dict | None) -> bool:
        """Should this adjudication be re-asked on the thinking model?

        True when a thinking checkpoint is configured AND the evidence
        render scored below the quality floor (blurry extrapolated view,
        occluded box, or a camera still embedded in structure). Those are
        exactly the images where multi-step visual reasoning beats a
        single forward pass; every other verdict stays on the fast tier.
        No thinking model configured -> never escalate (the plain
        confidence cap from _gate_quality remains the only penalty)."""
        return bool(self.thinking_model) and self._quality_floor(quality) < 0.35

    def _api_target(self, thinking: bool):
        """(api_base, api_key, model, timeout) for the tier in question."""
        if thinking:
            return (self.thinking_api_base, self.api_key,
                    self.thinking_model, self.thinking_timeout)
        return (self.api_base, self.api_key, self.model, self.timeout)

    # built via concatenation so the literal tags survive any tooling that
    # strips angle-bracket markup from source edits
    _THINK_O = "<" + "think>"
    _THINK_C = "</" + "think>"

    @classmethod
    def _strip_think(cls, text: str) -> str:
        """Remove inline chain-of-thought blocks from a thinking model's
        reply. Served without a reasoning parser, Qwen3-VL Thinking emits
        an explicit think block (or the newer channel syntax) before the
        final answer. vLLM with a reasoning parser puts the chain in a
        separate field and the content arrives clean -- stripping is a
        no-op then. A truncated chain (max_tokens hit inside the block)
        leaves no closing tag: everything from the opener on is dropped.
        """
        if not text:
            return text
        import re
        text = re.sub(re.escape(cls._THINK_O) + r".*?" + re.escape(cls._THINK_C),
                      "", text, flags=re.DOTALL)
        text = re.sub(r"<\|channel\|>analysis<\|message\|>.*?(<\|end\|>|$)",
                      "", text, flags=re.DOTALL)
        text = re.sub(r"<\|channel\|>\s*final\s*<\|message\|>", "", text)
        # unclosed think block: max_tokens truncated inside the chain --
        # nothing after the opener is trustworthy
        i = text.find(cls._THINK_O)
        if i >= 0:
            text = text[:i]
        return text.strip()

    def set_record(self, record_path: str) -> None:
        """Enable structured recording of every adjudication to a JSONL file.
        Also enables persisting every evidence image the VLM actually saw
        (saved next to the record file) so decisions can be audited."""
        self.record_path = record_path
        import os as _os
        d = _os.path.dirname(record_path)
        self.evidence_dir = d if d else "."

    def _save_evidence_png(self, img_arr: np.ndarray, name: str) -> str | None:
        """Persist the exact image the VLM adjudicates on (best-effort)."""
        if not self.evidence_dir:
            return None
        try:
            import os as _os
            import matplotlib
            matplotlib.use("Agg")
            import matplotlib.pyplot as plt
            _os.makedirs(self.evidence_dir, exist_ok=True)
            path = _os.path.join(self.evidence_dir, name)
            plt.imsave(path, img_arr)
            return path
        except Exception as e:
            print(f"[vlm][evidence] save failed ({type(e).__name__}: {e})")
            return None

    def _record(self, kind: str, prompt: str, answer: str,
                choice: str, confidence: float, detail: str,
                png_path: str | None = None,
                quality: dict | None = None,
                escalated: bool = False) -> None:
        """Append one adjudication record (image path + prompt + answer)."""
        if not self.record_path:
            return
        import os as _os
        rec = {"kind": kind, "prompt": prompt, "answer": answer,
               "choice": choice, "confidence": confidence,
               "detail": detail, "image": png_path}
        if escalated:
            rec["escalated"] = True
        if quality:
            rec["quality"] = quality
        try:
            _os.makedirs(_os.path.dirname(self.record_path), exist_ok=True)
            with open(self.record_path, "a", encoding="utf-8") as f:
                f.write(json.dumps(rec, ensure_ascii=False) + "\n")
        except Exception as e:
            print(f"[vlm][record] failed to write ({type(e).__name__}: {e})")

    # ---- backend-agnostic interface ----
    def adjudicate_box(self, scene, box, question: str,
                       options: list[str]) -> Verdict:
        """Ask the VLM a multiple-choice question about a candidate box.

        Hard-evidence escalation: when the render scored below the
        quality floor and a thinking model is configured, the question
        is re-asked there (multi-step visual reasoning) and its verdict
        REPLACES the fast one -- that is the whole point of the tier.
        The escalated verdict keeps normal confidence: the quality cap
        punished the FAST model's shallow read, not the image's
        usability for a careful reader."""
        if self.backend == "mock":
            return self._mock_adjudicate(box, question)
        escalated = False
        if self.backend == "local":
            v = self._local_adjudicate(scene, box, question, options)
        else:
            v = self._qwen_adjudicate(scene, box, question, options)
        quality = self._render_quality.get(self._render_cache_key([box])) or {}
        self._gate_quality(v, [box], quality=quality)
        if self._should_escalate(quality):
            if self.backend == "local":
                v2 = self._local_adjudicate(scene, box, question, options,
                                            thinking=True)
            else:
                v2 = self._qwen_adjudicate(scene, box, question, options,
                                           thinking=True)
            if v2 is not None and "mock" not in (v2.detail or ""):
                v2.png_path = v2.png_path or v.png_path
                v2.detail = f"[thinking-escalated] {v2.detail or ''}".strip()
                v = v2
                escalated = True
        self._record("box", question, v.raw or v.detail,
                     (v.params or {}).get("choice", ""), v.confidence, v.detail,
                     png_path=v.png_path,
                     quality=quality or None,
                     escalated=escalated)
        return v

    def adjudicate_pair(self, scene, a, b, question: str,
                        options: list[str]) -> Verdict:
        """Ask the VLM whether two boxes are faces of the SAME rack.

        Renders a local crop showing BOTH boxes, then routes to the same
        backend as adjudicate_box. Mock / failure degrades to 'keep' (do not
        merge without evidence).
        """
        if not (a and b):
            return Verdict(action="keep", confidence=0.5, detail="no boxes")
        boxes = [a, b]
        v = None
        png_path = None
        img_arr = None
        q = {}
        try:
            img_arr = render_topdown_image(scene.points, boxes,
                                           gs_ply=scene.meta.get("gs_ply"),
                                           quality_out=q)
            # persist the exact image the VLM reasons over: for a merge
            # decision this crop (both boxes + surrounding structure) is the
            # single most useful artifact when auditing a wrong merge/keep
            png_path = self._save_evidence_png(
                img_arr, f"pair_evidence_{a.box_id[:8]}_{b.box_id[:8]}.png")
            if self.backend == "local":
                v = self._local_pair_call(img_arr, question, options)
            elif self.backend == "qwen":
                v = self._qwen_pair_call(img_arr, question, options)
            else:
                v = Verdict(action="keep", confidence=0.5,
                            detail="mock: keep (no VLM)")
        except Exception as e:
            print(f"[vlm][pair] failed ({type(e).__name__}: {e}) -> keep")
            v = Verdict(action="keep", confidence=0.5, detail=f"{type(e).__name__}")
        v.png_path = png_path
        escalated = False
        if (self.backend in ("local", "qwen") and img_arr is not None
                and self._should_escalate(q)):
            # hard evidence -> re-ask on the thinking tier (see
            # adjudicate_box); a failed escalation keeps the fast verdict
            try:
                if self.backend == "local":
                    v2 = self._local_pair_call(img_arr, question, options,
                                               thinking=True)
                else:
                    v2 = self._qwen_pair_call(img_arr, question, options,
                                              thinking=True)
                if v2 is not None and "mock" not in (v2.detail or ""):
                    v2.png_path = png_path
                    v2.detail = (f"[thinking-escalated] {v2.detail or ''}").strip()
                    v = v2
                    escalated = True
            except Exception as e:
                print(f"[vlm][pair][thinking] failed "
                      f"({type(e).__name__}: {e}) -> keeping primary")
        q = self._gate_quality(v, boxes, quality=q)
        self._record("pair", question, v.raw or v.detail,
                     (v.params or {}).get("choice", ""), v.confidence, v.detail,
                     png_path=png_path, quality=q, escalated=escalated)
        return v

    def _local_pair_call(self, img_arr, question, options,
                         thinking: bool = False) -> Verdict:
        from PIL import Image
        import io as _io
        self._ensure_local_model(thinking=thinking)
        if thinking:
            processor, model = self._thinking_local_model
        else:
            processor, model = self._local_model
        buf = _io.BytesIO()
        import matplotlib.pyplot as plt
        plt.imsave(buf, img_arr, format="png")
        image = Image.open(buf).convert("RGB")
        prompt = (f"Decide the best answer from the bird's-eye image "
                  f"(two red wireframes = two candidate boxes).\n\n"
                  f"QUESTION: {question}\n"
                  f"OPTIONS:\n" + "\n".join(f"- {o}" for o in options) +
                  f"\n\nReply with the exact option text only.")
        messages = [{"role": "user", "content": [
            {"type": "image", "image": image}, {"type": "text", "text": prompt}]}]
        text = processor.apply_chat_template(messages, tokenize=False,
                                             add_generation_prompt=True)
        inputs = processor(text=[text], images=[image],
                           return_tensors="pt").to(model.device)
        import torch
        with torch.inference_mode():
            # the thinking tier needs room for its reasoning chain
            out = model.generate(**inputs,
                                 max_new_tokens=1024 if thinking else 64,
                                 do_sample=False)
        trimmed = [o[len(i):] for i, o in zip(inputs.input_ids, out)]
        answer = self._strip_think(processor.batch_decode(
            trimmed, skip_special_tokens=True)[0].strip())
        matched = self._match_option(answer, options)
        return Verdict(action="answer", params={"choice": matched},
                       confidence=0.8, detail=answer, raw=answer)

    def _qwen_pair_call(self, img_arr, question, options,
                        thinking: bool = False) -> Verdict:
        b64 = self._array_to_png_b64(img_arr)
        prompt = (f"Decide the best answer from the evidence image "
                  f"(two red wireframes = two candidate boxes).\n\n"
                  f"{_LOCAL_VIEW_DESC}\n\n"
                  f"QUESTION: {question}\n"
                  f"OPTIONS:\n" + "\n".join(f"- {o}" for o in options) +
                  f"\n\nReply with the exact option text only.")
        api_base, api_key, model_name, timeout = self._api_target(thinking)
        r = requests.post(
            api_base + "/chat/completions",
            headers={"Authorization": f"Bearer {api_key}"},
            json={"model": model_name, "messages": [{"role": "user", "content": [
                {"type": "text", "text": prompt},
                {"type": "image_url", "image_url": {"url": f"data:image/png;base64,{b64}"}}]}],
                # the thinking tier needs room for its reasoning chain
                "max_tokens": 1024 if thinking else 64, "temperature": 0.0},
            timeout=timeout,
        )
        r.raise_for_status()
        text = self._strip_think(
            r.json()["choices"][0]["message"]["content"].strip())
        matched = self._match_option(text, options)
        return Verdict(action="answer", params={"choice": matched},
                       confidence=0.8, detail=text, raw=text)

    # ---- god-view global audit ----
    _GODVIEW_PROMPT = (
        "You are auditing a data-center layout. The image is a bird's-eye "
        "view of the room: either a photorealistic 3D Gaussian-splatting "
        "render or a height-colored point-cloud scatter (colorbar = height in "
        "meters). Devices are tall rack structures (~2 m) that form parallel "
        "rows separated by aisles. The numbered red wireframes are candidate "
        "device boxes: each wireframe is the FULL 3D box (top and bottom "
        "faces plus vertical edges), so you can see the height and total "
        "volume each candidate claims, not just its floor footprint. Under "
        "perspective an off-centre box's bottom face leans slightly outward. "
        "(In the scatter fallback only the top-face rectangle is drawn.)\n\n"
        "Look at the GLOBAL spatial structure: identify boxes that are clearly "
        "NOT real devices, e.g. a wireframe floating in the middle of an "
        "aisle with no structure inside it, a box far away from every device "
        "row, or a box on the room boundary where only a wall exists.\n\n"
        "Be conservative: only flag a box when the evidence is clear. Do not "
        "flag boxes that sit inside a device row.\n\n"
        "Reply with ONLY a JSON object, no other text:\n"
        '{"suspicious": [{"index": <box number>, "reason": "<short reason>"}, ...]}\n'
        'If no box looks suspicious, reply exactly {"suspicious": []}'
    )

    def adjudicate_godview(self, scene, boxes) -> list[dict]:
        """One global VLM call over the whole scene. Returns the suspicious
        list ([{"index": i, "reason": str}]) with indices clamped to valid
        box indices. Mock backend / any failure -> empty list (pipeline
        continues with rule-detected issues only)."""
        if self.backend == "mock" or not boxes:
            return []
        png = render_godview_png(scene.points, boxes,
                                 gs_ply=scene.meta.get("gs_ply"))
        # god-view runs on the thinking tier outright when configured:
        # it fires once per repair-loop pass (low frequency) and its
        # false-positive nominations drive box DELETION (high stakes) --
        # exactly the latency/quality trade worth paying. A failure
        # there falls back to the fast model, then to skipping.
        use_thinking = bool(self.thinking_model)
        try:
            for thinking in ((True, False) if use_thinking else (False,)):
                try:
                    if self.backend == "local":
                        text = self._local_image_call(
                            png, self._GODVIEW_PROMPT,
                            max_new_tokens=2048 if thinking else 400,
                            thinking=thinking)
                    else:
                        text = self._qwen_image_call(
                            png, self._GODVIEW_PROMPT,
                            max_tokens=2048 if thinking else 400,
                            thinking=thinking)
                    break
                except Exception as e:
                    if not thinking:
                        raise
                    print(f"[vlm][godview][thinking] failed "
                          f"({type(e).__name__}: {e}) -> fast model")
        except Exception as e:
            print(f"[vlm][godview] failed ({type(e).__name__}: {e}) -> skipped")
            return []
        data = _extract_json(text)
        if not isinstance(data, dict):
            print(f"[vlm][godview] unparseable reply -> skipped: {text[:120]!r}")
            return []
        out = []
        seen: set[int] = set()
        for item in data.get("suspicious", []) or []:
            try:
                idx = int(item.get("index"))
            except (TypeError, ValueError):
                continue
            if 0 <= idx < len(boxes) and idx not in seen:
                seen.add(idx)
                out.append({"index": idx,
                           "reason": str(item.get("reason", ""))[:80]})
        return out

    # ---- global 2D grounding (rows as whole regions) ----
    _GROUND_PROMPT = (
        "You are looking at a TOP-DOWN view of a data-center room (ceiling "
        "removed). Rows of tall server racks / cabinets appear as solid "
        "bright bands; aisles are dark or empty; walls are thin lines at "
        "the room boundary.\n\n"
        "Task: output ONE axis-aligned rectangle per DEVICE STRUCTURE, "
        "covering its full footprint. A continuous row of joined cabinets "
        "counts as ONE rectangle spanning the WHOLE row (do NOT split it "
        "into individual cabinets). Structures separated by an aisle or a "
        "clear gap get separate rectangles. Do NOT box walls, pillars, "
        "columns, or floor clutter.\n\n"
        "Work step by step:\n"
        "1. List each device structure you see with one short sentence "
        "(row / single cabinet, roughly where).\n"
        "2. Then output ONE JSON object on the LAST line:\n"
        '{"regions": [{"x0": <int>, "y0": <int>, "x1": <int>, "y1": <int>}, ...]}\n'
        "Pixel coordinates, origin at the TOP-LEFT corner of the image, x "
        "rightward, y downward, x0 < x1, y0 < y1."
    )

    _GROUND_PROMPT_OBLIQUE = (
        "You are looking at an OBLIQUE top-down view of a data-center "
        "room (camera tilted ~58 deg, looking down the aisles, ceiling "
        "removed). Rows of server racks / cabinets appear as long "
        "HORIZONTAL bands; you see each row's front or back FACE plus "
        "its top. Perspective makes a band slightly trapezoidal "
        "(the nearer end looks a bit larger) -- that is expected.\n\n"
        "Task: output ONE axis-aligned rectangle per DEVICE STRUCTURE, "
        "covering the structure's full visible extent (the whole band, "
        "including the perspective-widened near end). A continuous row "
        "of joined cabinets counts as ONE rectangle spanning the WHOLE "
        "row. Structures separated by an aisle or a clear gap get "
        "separate rectangles. Do NOT box walls, pillars, columns, or "
        "floor clutter.\n\n"
        "Work step by step:\n"
        "1. List each device structure you see with one short sentence.\n"
        "2. Then output ONE JSON object on the LAST line:\n"
        '{"regions": [{"x0": <int>, "y0": <int>, "x1": <int>, "y1": <int>}, ...]}\n'
        "Pixel coordinates, origin at the TOP-LEFT corner of the image, x "
        "rightward, y downward, x0 < x1, y0 < y1."
    )

    def ground_regions(self, png: bytes, W: int, H: int,
                       png_path: str | None = None,
                       oblique: bool = False) -> list[tuple]:
        """2D grounding over a top-down view: outline EVERY device
        structure (a joined row = one region).

        oblique: the view is the tilted aisle-looking complement to the
        nadir view -- rows are horizontal bands with faces visible. The
        nadir centre shows only rack TOPS, which a ground-level 3DGS
        training set barely observed (blurry smear); the oblique views
        recover the faces.

        Returns pixel rects [(x0, y0, x1, y1)] or [] on mock / failure.
        Runs on the thinking tier when configured: one call per view,
        and these regions BECOME the pipeline's boxes (high stakes)."""
        prompt = self._GROUND_PROMPT_OBLIQUE if oblique \
            else self._GROUND_PROMPT
        if self.backend == "mock":
            return []
        use_thinking = bool(self.thinking_model)
        try:
            for thinking in ((True, False) if use_thinking else (False,)):
                try:
                    if self.backend == "local":
                        text = self._local_image_call(
                            png, prompt,
                            max_new_tokens=2048 if thinking else 900,
                            thinking=thinking)
                    else:
                        text = self._qwen_image_call(
                            png, prompt,
                            max_tokens=2048 if thinking else 900,
                            thinking=thinking)
                    break
                except Exception as e:
                    if not thinking:
                        raise
                    print(f"[vlm][ground][thinking] failed "
                          f"({type(e).__name__}: {e}) -> fast model")
        except Exception as e:
            print(f"[vlm][ground] failed ({type(e).__name__}: {e}) "
                  f"-> no grounding")
            return []
        data = _extract_json(text)
        if not isinstance(data, dict):
            print(f"[vlm][ground] unparseable reply -> no grounding: "
                  f"{text[:120]!r}")
            return []
        rects = []
        for item in data.get("regions", []) or []:
            try:
                x0, y0 = float(item["x0"]), float(item["y0"])
                x1, y1 = float(item["x1"]), float(item["y1"])
            except (KeyError, TypeError, ValueError):
                continue
            x0, x1 = min(x0, x1), max(x0, x1)
            y0, y1 = min(y0, y1), max(y0, y1)
            x0, y0 = max(0.0, x0), max(0.0, y0)
            x1, y1 = min(float(W), x1), min(float(H), y1)
            if x1 - x0 >= 8.0 and y1 - y0 >= 8.0:
                rects.append((x0, y0, x1, y1))
        self._record("ground", prompt, text or "",
                    f"{len(rects)} regions", 0.5, "", png_path=png_path)
        return rects

    # ---- per-row split (how many cabinets in one row box) ----
    _SPLIT_PROMPT = (
        "You are auditing ONE row structure in a data center. The image is "
        "a composite of up to TWO views of the SAME row, tiled side by "
        "side, each labeled above the panel: 'front' and 'back' (the two "
        "opposite long faces of the row; in a fallback render a single "
        "top-down view is shown instead). The red wireframe marks the row "
        "box -- it spans the WHOLE row by design, so do NOT flag its "
        "ends.\n\n"
        "The row may contain MULTIPLE separate cabinets joined side by "
        "side, or a single wide cabinet. Judge the cabinet units by door "
        "seams, panel boundaries, and the width rhythm; use BOTH faces "
        "(seams are often clearer on one side).\n\n"
        "Work step by step:\n"
        "1. Write ONE short sentence per view about how many distinct "
        "cabinet units you count and where the boundaries are.\n"
        "2. Then output ONE JSON object on the LAST line:\n"
        '{"count": <int>, "gaps": [<float>, ...]}\n'
        "- count: how many distinct cabinet units the row contains.\n"
        "- gaps: for count > 1, the internal boundary positions as "
        "fractions 0.0-1.0 along the row, measured in the FRONT view "
        "from its LEFT edge to its RIGHT edge (count-1 values, "
        "increasing). Empty list when count is 1."
    )

    @staticmethod
    def _parse_split_reply(text: str) -> dict | None:
        """Parse the split reply. None = keep whole (unparseable -> no
        split is the safe default)."""
        data = _extract_json(text)
        if not isinstance(data, dict):
            return None
        try:
            count = int(data.get("count", 1))
        except (TypeError, ValueError):
            return None
        count = max(1, min(count, 40))
        gaps = []
        for g in data.get("gaps", []) or []:
            try:
                g = float(g)
            except (TypeError, ValueError):
                continue
            if 0.02 < g < 0.98:
                gaps.append(min(max(g, 0.02), 0.98))
        gaps = sorted(set(round(g, 4) for g in gaps))[: max(count, 1)]
        return {"count": count, "gaps": gaps}

    def adjudicate_split(self, scene, box) -> Verdict:
        """How many cabinets does one whole-row box contain, and where
        are the internal boundaries? Renders the row's two long faces.
        Mock / any failure degrades to keep (no split without evidence
        -- an over-split row is far harder to repair downstream)."""
        if self.backend == "mock":
            return Verdict(action="keep", params={"count": 1, "gaps": []},
                           confidence=0.5, detail="mock: no split")
        try:
            img, quality = _render_split_views(scene, box)
        except Exception as e:
            print(f"[vlm][split] render failed ({type(e).__name__}: {e}) "
                  f"-> keep whole")
            return Verdict(action="keep", params={"count": 1, "gaps": []},
                           confidence=0.3, detail=f"render failed: {e}")
        from agentic_gts.output.gs_render import png_bytes as _pb
        png = _pb(img)
        png_path = self._save_evidence_png(
            img, f"split_evidence_{box.box_id[:8]}.png")
        use_thinking = bool(self.thinking_model)
        text = None
        try:
            for thinking in ((True, False) if use_thinking else (False,)):
                try:
                    if self.backend == "local":
                        text = self._local_image_call(
                            png, self._SPLIT_PROMPT,
                            max_new_tokens=2048 if thinking else 300,
                            thinking=thinking)
                    else:
                        text = self._qwen_image_call(
                            png, self._SPLIT_PROMPT,
                            max_tokens=2048 if thinking else 300,
                            thinking=thinking)
                    break
                except Exception as e:
                    if not thinking:
                        raise
                    print(f"[vlm][split][thinking] failed "
                          f"({type(e).__name__}: {e}) -> fast model")
        except Exception as e:
            print(f"[vlm][split] failed ({type(e).__name__}: {e}) -> keep")
            return Verdict(action="keep", params={"count": 1, "gaps": []},
                           confidence=0.3, detail=f"call failed: {e}")
        p = self._parse_split_reply(text)
        if p is None:
            print(f"[vlm][split] unparseable reply -> keep: {text[:120]!r}")
            return Verdict(action="keep", params={"count": 1, "gaps": []},
                           confidence=0.3, detail="unparseable")
        self._record("split", self._SPLIT_PROMPT, text or "",
                    f"count={p['count']} gaps={p['gaps']}", 0.6,
                    "", png_path=png_path, quality=quality or None)
        return Verdict(action="keep" if p["count"] <= 1 else "split",
                       params=p, confidence=0.6, detail="")

    # ---- fine-grained fit refinement (size + yaw around z) ----
    _FIT_PROMPT = (
        "You are auditing the 3D bounding box around ONE data-center rack.\n"
        f"{_LOCAL_VIEW_DESC}\n"
        "Additionally, two arrows are drawn on the box's TOP face: a GREEN "
        "arrow along the box's local x axis (its length direction) and a "
        "BLUE arrow along its local y axis (its depth direction).\n\n"
        "IMPORTANT: these candidate boxes come from an INITIAL rough "
        "detection and usually do NOT fit the device well. Your job is NOT "
        "to measure metres or degrees -- an image does not support that "
        "precision, and a geometric stage re-fits the box to the point "
        "cloud afterwards. Your job is ONLY to nominate WHICH end is off "
        "and WHICH direction the yaw rotates. Direction judgments are "
        "reliable from an image; magnitude judgments are not.\n\n"
        "The typical errors are: the wireframe stops short of the device "
        "edge (the device continues past a wireframe end), the wireframe "
        "overhangs past the device edge into the aisle or over the "
        "neighbouring device, or the GREEN arrow is not parallel to the "
        "device's long axis. Only rotation around the vertical (z) axis "
        "is relevant. The box HEIGHT and its row DEPTH are handled by "
        "other stages -- do NOT judge them.\n\n"
        "Work step by step:\n"
        "1. For each of the three views, write ONE short sentence stating "
        "whether the red wireframe matches the device outline, and if "
        "not, which end is wrong in which direction.\n"
        "2. Then output ONE JSON object on the LAST line:\n"
        '{"x_minus": "short"|"over"|"ok", '
        '"x_plus": "short"|"over"|"ok", '
        '"yaw_dir": "cw"|"ccw"|"ok"}\n'
        "- x_minus: the wireframe end OPPOSITE the GREEN arrow, relative "
        "to the device edge: 'short' = the device continues past the "
        "wireframe end, 'over' = the wireframe hangs past the device "
        "edge, 'ok' = they match.\n"
        "- x_plus: the wireframe end the GREEN arrow points at (same "
        "three categories).\n"
        "- yaw_dir: seen from ABOVE, which way the GREEN arrow must "
        "rotate to become parallel to the device's long axis: 'cw' = "
        "clockwise, 'ccw' = counterclockwise, 'ok' = already parallel. "
        "Use the oblique near-top-down panel for this.\n"
        "Use 'ok' only for a judgment that genuinely fits."
    )

    _YAW_PROMPT = (
        "You are auditing the 3D bounding box around ONE data-center rack.\n"
        "The image is ONE near-top-down oblique view of the candidate "
        "region: the rack's TOP face is visible with little perspective "
        "foreshortening, together with the tops of its neighbours along "
        "the row. The red wireframe marks the candidate box (the full 3D "
        "box). Two arrows are drawn on its top face: a GREEN arrow along "
        "the box's local x axis (its length direction) and a BLUE arrow "
        "along its local y axis (its depth direction).\n\n"
        "ONE question only: is the GREEN arrow parallel to the device's "
        "long axis (the row direction)? If not, decide which way the box "
        "must rotate around the vertical z-axis, SEEN FROM ABOVE, to "
        "make it parallel. Length and depth judgments are handled by "
        "other stages -- do NOT judge them here.\n\n"
        "Work step by step:\n"
        "1. Write ONE short sentence comparing the GREEN arrow's "
        "direction against the row's direction.\n"
        "2. Then output ONE JSON object on the LAST line:\n"
        '{"yaw_dir": "cw"|"ccw"|"ok"}\n'
        "- 'cw' = the green arrow must rotate clockwise (seen from "
        "above),\n"
        "- 'ccw' = counterclockwise,\n"
        "- 'ok' = already parallel. Use 'ok' only when it genuinely fits."
    )

    _EXTENT_PROMPT = (
        "You are auditing the 3D bounding box around ONE data-center rack.\n"
        "The image is a composite of TWO views of the same candidate "
        "region, tiled side by side, each labeled above the panel: "
        "'front' (the rack's front face: doors, panels, LEDs) and 'side' "
        "(view along the row: neighbouring racks give the row rhythm). "
        "Red wireframes mark the candidate box (the full 3D box). A "
        "GREEN arrow on its top face shows the box's local x axis (its "
        "length direction).\n\n"
        "The box's ORIENTATION has already been corrected by a previous "
        "stage -- assume the green arrow is parallel to the row. ONE "
        "question only: do the wireframe's ends along the GREEN axis "
        "match the device's edges? For each end: 'short' = the device "
        "continues past the wireframe end, 'over' = the wireframe hangs "
        "past the device edge (into the aisle or over the neighbour), "
        "'ok' = they match. Rotation and depth are other stages' "
        "concerns -- do NOT judge them.\n\n"
        "Work step by step:\n"
        "1. For each view, write ONE short sentence about whether the "
        "wireframe's two ends along the green axis match the device "
        "edges.\n"
        "2. Then output ONE JSON object on the LAST line:\n"
        '{"x_minus": "short"|"over"|"ok", '
        '"x_plus": "short"|"over"|"ok"}\n'
        "- x_minus: the wireframe end OPPOSITE the GREEN arrow,\n"
        "- x_plus: the end the GREEN arrow points at.\n"
        "Use 'ok' only for a judgment that genuinely fits."
    )

    def _adjudicate_views(self, scene, box, kind: str, prompt: str,
                         slots: tuple, parser) -> Verdict:
        """Shared per-box single-question adjudication on SELECTED views.

        Renders the given view slots (see render_topdown_image), asks ONE
        focused question, parses the categorical reply. Guards identical
        to the combined fit pass: mock / any failure -> keep; thinking
        escalation on hard evidence (keep_allok lets a careful 'it fits'
        override a shaky fast nomination); the quality cap applies only
        to the fast tier.
        """
        if self.backend == "mock":
            return Verdict(action="keep", confidence=0.5,
                           detail=f"mock: no {kind} refinement")
        png_path = None
        q = {}
        try:
            img = render_topdown_image(scene.points, [box],
                                       gs_ply=scene.meta.get("gs_ply"),
                                       overlay="wire3d_axes", quality_out=q,
                                       slots=slots,
                                       gs_cams=scene.meta.get("gs_cams"))
            png_path = self._save_evidence_png(
                img, f"{kind}_evidence_{box.box_id[:8]}.png")
            png = self._array_png_bytes(img)
            if self.backend == "local":
                text = self._local_image_call(png, prompt,
                                              max_new_tokens=256)
            else:
                text = self._qwen_image_call(png, prompt, max_tokens=256)
        except Exception as e:
            print(f"[vlm][{kind}] failed ({type(e).__name__}: {e}) -> keep")
            return Verdict(action="keep", confidence=0.5,
                           detail=type(e).__name__)
        params = parser(text)
        # hard evidence (blurry / occluded / steep-fallback view) -> re-ask
        # on the thinking tier. keep_allok distinguishes "careful reader
        # says it fits" from an unparseable reply, so the escalation can
        # OVERRIDE a shaky fast nomination with a considered all-ok.
        escalated = False
        if self._should_escalate(q):
            try:
                if self.backend == "local":
                    text2 = self._local_image_call(png, prompt,
                                                  max_new_tokens=2048,
                                                  thinking=True)
                else:
                    text2 = self._qwen_image_call(png, prompt,
                                                  max_tokens=2048,
                                                  thinking=True)
                p2 = parser(text2, keep_allok=True)
                if p2 is not None:
                    params = p2
                    text = text2
                    escalated = True
            except Exception as e:
                print(f"[vlm][{kind}][thinking] failed "
                      f"({type(e).__name__}: {e}) -> keeping primary")
        # an escalated all-ok is a real "it fits" verdict -> keep, not refine
        has_nom = bool(params) and any(v != "ok" for v in params.values())
        self._record(kind, prompt, text, "",
                     0.8 if has_nom else 0.5,
                     str(params) if params else "no change",
                     png_path=png_path, quality=q or None,
                     escalated=escalated)
        if not has_nom:
            tag = "[thinking-escalated] " if escalated else ""
            return Verdict(action="keep", confidence=0.5,
                           detail=tag + text[:80],
                           raw=text, png_path=png_path)
        tag = "[thinking-escalated] " if escalated else ""
        v = Verdict(action="refine", params=params, confidence=0.8,
                    detail=tag + text[:80], raw=text, png_path=png_path)
        if not escalated:
            # a low-quality render weakens the visual evidence the
            # corrections were derived from -- cap the confidence so the
            # loop's guards (which weigh confidence) treat the proposal
            # more sceptically. An ESCALATED verdict skips the cap: the
            # thinking tier already reasoned carefully over the hard
            # image, re-penalising it would just push it back into the
            # conservative path the escalation was meant to escape.
            self._gate_quality(v, [box], quality=q)
        return v

    def adjudicate_fit(self, scene, box) -> Verdict:
        """COMBINED single-call variant: all three questions (yaw + both
        x-ends) on the three-view composite. Kept for auditing and tests;
        the production refine loop now asks them SEQUENTIALLY (see
        adjudicate_yaw / adjudicate_extent) -- a skewed box contaminates
        every horizontal-view extent judgment, so orientation is
        corrected FIRST and the extent question re-rendered on the
        corrected geometry.
        """
        return self._adjudicate_views(scene, box, "fit", self._FIT_PROMPT,
                                      ("front", "side", "oblique"),
                                      self._parse_fit_reply)

    def adjudicate_yaw(self, scene, box) -> Verdict:
        """Phase 1 of the two-phase refine: ORIENTATION only, judged from
        the oblique near-top-down view -- the row direction reads best
        with little perspective foreshortening, and every horizontal-view
        extent judgment downstream is unreliable until the yaw is right.
        Categorical cw/ccw reply; the magnitude is swept geometrically
        downstream (geo.sweep_yaw).
        """
        return self._adjudicate_views(scene, box, "yaw", self._YAW_PROMPT,
                                      ("oblique",), self._parse_yaw_reply)

    def adjudicate_extent(self, scene, box) -> Verdict:
        """Phase 2: LENGTH ends only, judged from the front/side views of
        the ALREADY yaw-corrected box (the caller re-renders after phase
        1 adopted a correction). Depth is complete_row_depth's domain,
        rotation phase 1's -- neither is asked here.
        """
        return self._adjudicate_views(scene, box, "extent",
                                      self._EXTENT_PROMPT,
                                      ("front", "side"),
                                      self._parse_extent_reply)

    @staticmethod
    def _parse_fit_reply(text: str, keep_allok: bool = False) -> dict | None:
        """Parse the VLM's categorical fit nomination. None if no change.

        The VLM no longer outputs metres/degrees: pseudo-precision from
        an image. It nominates WHICH x-end is off in which sense
        (short/over) and WHICH way the yaw rotates (cw/ccw); the exact
        magnitude is a geometric quantity searched downstream (yaw sweep
        to the support peak, edge snap to the density profile). Unknown
        category values fall back to 'ok'; an all-ok reply is a keep.
        keep_allok=True returns the all-ok dict instead of None -- used
        by the thinking escalation, which must be able to OVERRIDE a
        shaky fast nomination with a considered all-ok (a None there
        would mean "could not parse" and keep the fast verdict).
        """
        data = _extract_json(text)
        if not isinstance(data, dict):
            return None

        def _cat(key: str, allowed: set) -> str:
            v = str(data.get(key, "ok")).strip().lower()
            return v if v in allowed else "ok"

        p = {
            "x_minus": _cat("x_minus", {"short", "over", "ok"}),
            "x_plus": _cat("x_plus", {"short", "over", "ok"}),
            "yaw_dir": _cat("yaw_dir", {"cw", "ccw", "ok"}),
        }
        if all(v == "ok" for v in p.values()):
            return p if keep_allok else None
        return p

    @staticmethod
    def _parse_yaw_reply(text: str, keep_allok: bool = False) -> dict | None:
        """Parse the phase-1 orientation nomination: which way the yaw
        rotates (cw/ccw), magnitude measured geometrically downstream.
        Same keep_allok contract as _parse_fit_reply."""
        data = _extract_json(text)
        if not isinstance(data, dict):
            return None
        v = str(data.get("yaw_dir", "ok")).strip().lower()
        if v not in ("cw", "ccw"):
            v = "ok"
        if v == "ok" and not keep_allok:
            return None
        return {"yaw_dir": v}

    @staticmethod
    def _parse_extent_reply(text: str, keep_allok: bool = False) -> dict | None:
        """Parse the phase-2 length-end nomination (short/over per x-end).
        Same keep_allok contract as _parse_fit_reply."""
        data = _extract_json(text)
        if not isinstance(data, dict):
            return None

        def _cat(key: str) -> str:
            v = str(data.get(key, "ok")).strip().lower()
            return v if v in ("short", "over", "ok") else "ok"

        p = {"x_minus": _cat("x_minus"), "x_plus": _cat("x_plus")}
        if all(v == "ok" for v in p.values()):
            return p if keep_allok else None
        return p

    @staticmethod
    def _array_png_bytes(arr: np.ndarray) -> bytes:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        buf = io.BytesIO()
        plt.imsave(buf, arr, format="png")
        return buf.getvalue()

    # ---- mock (rule) fallback ----
    def _mock_adjudicate(self, box, question: str) -> Verdict:
        q = question.lower()
        if "merge" in q or "split" in q or "rack" in q:
            if "完整" in question or "complete" in q:
                return Verdict(action="completed", confidence=0.6, detail="mock: assume ok")
            return Verdict(action="split", params={"n": 2}, confidence=0.5,
                           detail="mock: assume merged row")
        if "missing" in q or "存在" in question:
            return Verdict(action="keep", confidence=0.5, detail="mock: assume fine")
        return Verdict(action="keep", confidence=0.5, detail="mock default")

    # ---- shared image-call helpers (used by godview and box adjudication) ----
    def _local_image_call(self, png_bytes: bytes, prompt: str,
                          max_new_tokens: int = 64,
                          thinking: bool = False) -> str:
        """In-process transformers call with a PNG image + text prompt."""
        from PIL import Image
        self._ensure_local_model(thinking=thinking)
        if thinking:
            processor, model = self._thinking_local_model
        else:
            processor, model = self._local_model
        image = Image.open(io.BytesIO(png_bytes)).convert("RGB")
        messages = [{
            "role": "user",
            "content": [
                {"type": "image", "image": image},
                {"type": "text", "text": prompt},
            ],
        }]
        text = processor.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True)
        inputs = processor(text=[text], images=[image],
                           return_tensors="pt").to(model.device)
        import torch
        with torch.inference_mode():
            out = model.generate(**inputs, max_new_tokens=max_new_tokens,
                                  do_sample=False)
        trimmed = [o[len(i):] for i, o in zip(inputs.input_ids, out)]
        return self._strip_think(
            processor.batch_decode(trimmed, skip_special_tokens=True)[0].strip())

    def _qwen_image_call(self, png_bytes: bytes, prompt: str,
                         max_tokens: int = 64,
                         thinking: bool = False) -> str:
        """OpenAI-compatible chat call with a base64 PNG image."""
        b64 = base64.b64encode(png_bytes).decode("ascii")
        api_base, api_key, model, timeout = self._api_target(thinking)
        r = requests.post(
            api_base + "/chat/completions",
            headers={"Authorization": f"Bearer {api_key}"},
            json={
                "model": model,
                "messages": [{
                    "role": "user",
                    "content": [
                        {"type": "text", "text": prompt},
                        {"type": "image_url",
                         "image_url": {"url": f"data:image/png;base64,{b64}"}},
                    ],
                }],
                "max_tokens": max_tokens,
                "temperature": 0.0,
            },
            timeout=timeout,
        )
        r.raise_for_status()
        return self._strip_think(
            r.json()["choices"][0]["message"]["content"].strip())

    # ---- local in-process transformers model ----
    def _ensure_local_model(self, thinking: bool = False):
        """Load the model once; subsequent adjudications reuse it.
        thinking=True loads the ESCALATION checkpoint into a separate
        slot (lazily -- only if an escalation ever fires)."""
        if not thinking and self._local_model is not None:
            return
        if thinking and self._thinking_local_model is not None:
            return
        if thinking and not self.thinking_model:
            raise RuntimeError("no thinking model configured")
        model_path = self.thinking_model if thinking else self.model
        import torch
        import transformers
        from transformers import AutoProcessor
        try:  # Qwen3-VL needs a recent transformers; fall back to the
              # generic auto class for other VL families (Qwen2-VL, ...)
            from transformers import Qwen3VLForConditionalGeneration as ModelCls
        except ImportError:
            from transformers import AutoModelForImageTextToText as ModelCls
        print(f"[vlm][local] loading {model_path} (transformers "
              f"{transformers.__version__}) ... first call only")
        processor = AutoProcessor.from_pretrained(model_path)
        # Pick the best available attention backend. flash_attention_2 is the
        # fastest on CUDA but requires the flash-attn package; if it is not
        # importable, fall back to sdpa (the default efficient path in
        # transformers >= 2.0, still much better than eager). Never force a
        # backend that is not installed.
        attn = None
        if torch.cuda.is_available():
            try:
                import flash_attn  # noqa: F401
                attn = "flash_attention_2"
                print("[vlm][local] using flash_attention_2")
            except ImportError:
                pass
        if attn is None:
            # is_torch_sdpa_available can live at the package root or under
            # transformers.utils depending on version; probe both.
            _probe = None
            try:
                from transformers import is_torch_sdpa_available as _probe
            except ImportError:
                try:
                    from transformers.utils import is_torch_sdpa_available as _probe
                except ImportError:
                    _probe = None
            if _probe is not None:
                try:
                    if _probe():
                        attn = "sdpa"
                        print("[vlm][local] using sdpa attention")
                except Exception:
                    pass
        if not attn:
            print("[vlm][local] no fast attention backend -> eager attention")
        kwargs = {"torch_dtype": torch.bfloat16, "device_map": "auto"}
        if attn:
            kwargs["attn_implementation"] = attn
        model = ModelCls.from_pretrained(model_path, **kwargs)
        model.eval()
        if thinking:
            self._thinking_local_model = (processor, model)
        else:
            self._local_model = (processor, model)

    def _local_adjudicate(self, scene, box, question: str,
                          options: list[str],
                          thinking: bool = False) -> Verdict | None:
        try:
            import matplotlib
            matplotlib.use("Agg")
            import matplotlib.pyplot as plt
            from PIL import Image

            self._ensure_local_model(thinking=thinking)
            if thinking:
                processor, model = self._thinking_local_model
            else:
                processor, model = self._local_model

            img_arr = self._render_cached(scene, [box])
            png_path = self._save_evidence_png(
                img_arr, f"evidence_{box.box_id[:8]}.png")
            buf = io.BytesIO()
            plt.imsave(buf, img_arr, format="png")
            image = Image.open(buf).convert("RGB")

            prompt = (
                f"You are an auditor in a data-center layout tool. Decide the best "
                f"answer for this question by looking at the bird's-eye view "
                f"image (red wireframe = the candidate box; photorealistic "
                f"3DGS render or point-cloud scatter).\n\n"
                f"QUESTION: {question}\n"
                f"OPTIONS:\n" + "\n".join(f"- {o}" for o in options) +
                f"\n\nReply with the exact option text only."
            )
            messages = [{
                "role": "user",
                "content": [
                    {"type": "image", "image": image},
                    {"type": "text", "text": prompt},
                ],
            }]
            text = processor.apply_chat_template(
                messages, tokenize=False, add_generation_prompt=True)
            inputs = processor(text=[text], images=[image],
                                return_tensors="pt").to(model.device)
            import torch
            with torch.inference_mode():
                # the thinking tier needs room for its reasoning chain
                out = model.generate(**inputs,
                                     max_new_tokens=1024 if thinking else 64,
                                     do_sample=False)
            trimmed = [o[len(i):] for i, o in zip(inputs.input_ids, out)]
            answer = self._strip_think(processor.batch_decode(
                trimmed, skip_special_tokens=True)[0].strip())
            matched = self._match_option(answer, options)
            return Verdict(action="answer", params={"choice": matched},
                           confidence=0.8, detail=answer, raw=answer,
                           png_path=png_path)
        except Exception as e:
            if thinking:
                # escalation failure must not clobber the primary verdict
                print(f"[vlm][local][thinking] failed "
                      f"({type(e).__name__}: {e}) -> keeping primary")
                return None
            print(f"[vlm][local] inference failed ({type(e).__name__}: {e}) "
                  f"-> falling back to mock")
            return self._mock_adjudicate(box, question)

    # ---- Qwen (OpenAI-compatible chat completions with image) ----
    def _qwen_adjudicate(self, scene, box, question: str,
                         options: list[str],
                         thinking: bool = False) -> Verdict | None:
        img_arr = self._render_cached(scene, [box])
        png_path = self._save_evidence_png(
            img_arr, f"evidence_{box.box_id[:8]}.png")
        b64 = self._array_to_png_b64(img_arr)
        prompt = (
            f"You are an auditor in a data-center layout tool. Decide the best "
            f"answer for this question by looking at the evidence image "
            f"(red wireframe = the candidate box).\n\n"
            f"{_LOCAL_VIEW_DESC}\n\n"
            f"QUESTION: {question}\n"
            f"OPTIONS:\n" + "\n".join(f"- {o}" for o in options) +
            f"\n\nReply with the exact option text only."
        )
        api_base, api_key, model_name, timeout = self._api_target(thinking)
        try:
            r = requests.post(
                api_base + "/chat/completions",
                headers={"Authorization": f"Bearer {api_key}"},
                json={
                    "model": model_name,
                    "messages": [{
                        "role": "user",
                        "content": [
                            {"type": "text", "text": prompt},
                            {"type": "image_url",
                             "image_url": {"url": f"data:image/png;base64,{b64}"}},
                        ],
                    }],
                    # the thinking tier needs room for its reasoning chain
                    "max_tokens": 1024 if thinking else 64,
                    "temperature": 0.0,
                },
                timeout=timeout,
            )
            r.raise_for_status()
            text = self._strip_think(
                r.json()["choices"][0]["message"]["content"].strip())
            matched = self._match_option(text, options)
            return Verdict(action="answer", params={"choice": matched},
                           confidence=0.8, detail=text, raw=text,
                           png_path=png_path)
        except Exception as e:  # fallback to mock on any failure
            if thinking:
                # escalation failure must not clobber the primary verdict
                print(f"[vlm][qwen][thinking] failed "
                      f"({type(e).__name__}: {e}) -> keeping primary")
                return None
            return self._mock_adjudicate(box, question)

    @staticmethod
    def _array_to_png_b64(arr: np.ndarray) -> str:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        buf = io.BytesIO()
        plt.imsave(buf, arr, format="png")
        buf.seek(0)
        return base64.b64encode(buf.read()).decode("ascii")

    @staticmethod
    def _match_option(text: str, options: list[str]) -> str:
        low = text.lower().strip()
        for o in options:
            if o.lower() in low or low in o.lower():
                return o
        return options[-1] if options else ""
