#!/usr/bin/env python
"""Standalone probe: VLM -> SAM point prompt generation, in isolation.

Feeds ONE local-view image through exactly the production path:
  adjudicate_sam_points (the _SAM_POINT_PROMPT call)
    -> parse_point_groups (0-1000 grid -> pixels)
    -> _pull_points_inward (edge positives snapped to device interior)
    -> _augment_spread   (farthest-point samples when coverage is poor)
and writes a side-by-side debug PNG:
  LEFT:  the points the VLM returned (as parsed, pre-postprocess)
  RIGHT: the points SAM would actually receive (post-processed)

Usage examples:
  # an existing rendered view (e.g. from a pipeline run dir)
  python test_sam_points.py --image runs/xxx/mask_prompt_ab12_front.png

  # render one from a GS ply + a box (center/size in metres, yaw in deg)
  python test_sam_points.py --gs scene.ply \
      --center 3.0 0.5 1.0 --size 1.1 0.6 2.0 --yaw 0

  # no VLM server up? mock backend still shows the render + empty prompts
  python test_sam_points.py --gs scene.ply --center ... --vlm mock

VLM connection follows the pipeline conventions: --vlm qwen|local|mock,
--vlm-base, --vlm-model, or env VLM_API_BASE / VLM_MODEL.
"""
from __future__ import annotations

import argparse
import json
import math
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from agentic_gts.agent.judge import VLMJudge
from agentic_gts.agent.mask_refine import (
    _augment_spread, _calibrate_coord_scale, _pull_points_inward,
    parse_point_groups,
)
from agentic_gts.core.models import OrientedBox


def _load_image(path: str) -> np.ndarray:
    from PIL import Image
    img = np.asarray(Image.open(path).convert("RGB"), dtype=np.float32) / 255.0
    if img.ndim == 2:
        img = np.stack([img] * 3, axis=-1)
    return img


def _render_view(gs_ply: str, center, size, yaw_deg: float) -> np.ndarray:
    """Render a front box-only local view for the given box params."""
    from agentic_gts.agent.mask_refine import _box_only_mask, _subset_or_none
    from agentic_gts.core.models import Scene
    from agentic_gts.output.gs_render import make_local_cam, rasterize_gs
    from agentic_gts.tools.gs_io import read_gaussian_ply

    box = OrientedBox(center=center, size=size, yaw=math.radians(yaw_deg))
    gs = read_gaussian_ply(gs_ply)
    open_vec, corridor = (np.array([0.0, 1.0]), 1.5)
    try:
        from agentic_gts.agent.mask_refine import _open_side
        open_vec, corridor = _open_side(gs, box)
    except Exception as e:
        print(f"[warn] _open_side failed ({e}); defaulting to +y")
    yaw = float(box.yaw)
    cross = np.array([-math.sin(yaw), math.cos(yaw)])
    axis = np.array([math.cos(yaw), math.sin(yaw)])
    if box.size[0] >= box.size[1]:
        azim, face = 0.0, cross
    else:
        azim, face = 90.0, axis
    if float(open_vec @ face) < 0.0:
        azim += 180.0
    standoff = float(np.clip(0.8 * corridor, 0.6, 2.2))
    cam = make_local_cam([box], W=768, H=768, elev_deg=18.0,
                         azim_deg=azim, standoff=standoff)
    sub = _subset_or_none(gs, _box_only_mask(gs, box))
    img = rasterize_gs(sub, cam) if sub is not None else None
    if img is None:
        raise SystemExit("no CUDA rasterizer available -- use --image "
                         "with an already-rendered view instead")
    return img


def _draw_points(img: np.ndarray, coords, labels, title: str,
                 moved=None) -> np.ndarray:
    """Copy of the debug composite's panel 1 drawing (PIL, no mpl)."""
    from PIL import Image, ImageDraw
    u8 = (np.clip(img, 0, 1) * 255).astype(np.uint8)[..., :3]
    p = Image.fromarray(u8.copy())
    d = ImageDraw.Draw(p)
    H, W = u8.shape[:2]
    r = max(6, W // 128)
    for i, ((x, y), lab) in enumerate(zip(coords, labels)):
        color = (0, 255, 0) if lab > 0 else (255, 40, 40)
        if moved is not None and i < len(moved) and moved[i]:
            color = (255, 200, 0)        # postprocess moved this one
        d.ellipse((x - r, y - r, x + r, y + r),
                  fill=color, outline=(255, 255, 255), width=2)
    strip = Image.new("RGB", (W, 26), (0, 0, 0))
    ImageDraw.Draw(strip).text((6, 6), title, fill=(255, 255, 255))
    return np.asarray(p), np.asarray(strip)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--image", help="path to an existing local-view PNG")
    ap.add_argument("--gs", help="3DGS ply to render a view from")
    ap.add_argument("--center", nargs=3, type=float, metavar=("X", "Y", "Z"))
    ap.add_argument("--size", nargs=3, type=float, metavar=("L", "W", "H"))
    ap.add_argument("--yaw", type=float, default=0.0, help="degrees")
    ap.add_argument("--vlm", default="mock", choices=["mock", "qwen", "local"])
    ap.add_argument("--vlm-base", default=None)
    ap.add_argument("--vlm-model", default=None)
    ap.add_argument("--view-name", default="front")
    ap.add_argument("--out", default="sam_points_probe.png")
    args = ap.parse_args()

    if args.image:
        img = _load_image(args.image)
        box = None
    elif args.gs and args.center and args.size:
        img = _render_view(args.gs, args.center, args.size, args.yaw)
        box = OrientedBox(center=args.center, size=args.size,
                          yaw=math.radians(args.yaw))
    else:
        ap.error("need --image, or --gs with --center/--size")

    judge = VLMJudge(backend=args.vlm, api_base=args.vlm_base,
                     model=args.vlm_model)
    H, W = img.shape[:2]
    probe_box = box or OrientedBox(center=(0, 0, 0), size=(1, 1, 1), yaw=0.0)
    verdict = judge.adjudicate_sam_points(img, probe_box, args.view_name,
                                          png_path=args.out)
    groups = verdict.params.get("groups", []) if verdict.params else []
    print(f"\n[VLM] backend={judge.backend} view={args.view_name} "
          f"conf={verdict.confidence:.2f}")
    print(f"[VLM] raw reply:\n{verdict.raw or verdict.detail}\n")

    parsed = parse_point_groups(verdict.raw or "")
    if not parsed:
        print("[parse] no groups parsed from the reply "
              "(mock backend returns none)")

    def _fg_frac(pts):
        """Fraction of points on the (bright) device -- box-only local
        views render the device on black, so fg = the device."""
        m = img[..., :3].mean(axis=2) > 0.08
        x = np.clip(np.rint(pts[:, 0]), 0, W - 1).astype(int)
        y = np.clip(np.rint(pts[:, 1]), 0, H - 1).astype(int)
        return float(m[y, x].mean())

    def _read(raw_pts, pixel, flip):
        p = np.asarray(raw_pts, dtype=float).copy()
        if flip:
            p[:, 1] = 1000.0 - p[:, 1]
        if pixel:
            return np.clip(p, 0, [W - 1, H - 1])
        return p / 1000.0 * np.array([W - 1, H - 1])

    for gi, g in enumerate(parsed):
        coords, labels = g.pixel_prompts(W, H)
        if not len(coords):
            continue
        # ---- coordinate-convention diagnosis -------------------------
        # score all four readings {grid, pixels} x {normal, Y-flip}
        raw = coords / np.array([W - 1, H - 1]) * 1000.0
        all_pts = g.positive_norm + g.negative_norm
        xs = [p[0] for p in all_pts]
        ys = [p[1] for p in all_pts]
        certain_grid = max(max(xs), max(ys)) > max(W, H) + 2
        pos = coords[labels > 0]
        scores = {}
        for name, pix, fl in (("grid", False, False),
                              ("pixels", True, False),
                              ("grid+Yflip", False, True),
                              ("pixels+Yflip", True, True)):
            scores[name] = _fg_frac(_read(pos, pix, fl))
        best = max(scores, key=scores.get)
        print(f"[group {gi}] hypothesis={g.hypothesis} "
              f"conf={g.confidence:.2f} "
              f"pos={int((labels > 0).sum())} "
              f"neg={int((labels < 1).sum())}")
        print(f"[group {gi}] raw x range [{min(xs):.0f}, {max(xs):.0f}] "
              f"y range [{min(ys):.0f}, {max(ys):.0f}] "
              f"(image {W}x{H})")
        print(f"[group {gi}] on-device fractions: "
              + " | ".join(f"{k} {v:.2f}" for k, v in scores.items()))
        if certain_grid:
            print(f"[group {gi}] convention: pixel readings ruled out "
                  f"(values beyond pixel range); grid vs grid+Yflip "
                  f"tested above")
            del scores["pixels"], scores["pixels+Yflip"]
            best = max(scores, key=scores.get)
        if scores[best] - scores["grid"] >= 0.25:
            print(f"[group {gi}] convention: looks like "
                  f"{best.upper()} (clear win over grid)")
        else:
            print(f"[group {gi}] convention: no clear winner -- keeping "
                  f"the 0-1000 grid (if points still look off in the "
                  "panels below, the model is just mis-grounding)")
        # the production postprocess chain, verbatim
        coords1 = _calibrate_coord_scale(img, coords, labels)
        coords2, labels2 = _pull_points_inward(img, coords1, labels)
        coords2, labels2 = _augment_spread(img, coords2, labels2)
        moved = [not np.allclose(a, b)
                 for a, b in zip(coords, coords2[:len(coords)])]
        n_pos1 = int((labels2 > 0).sum())
        print(f"[group {gi}] after postprocess pos={n_pos1} "
              f"({sum(moved)} moved by postproc)\n")
        # five panels: four readings + what SAM actually receives
        panels = []
        for title, cc, ll, mv in (
                ("1. as 0-1000 grid (assumed)", coords, labels, None),
                ("2. as ABSOLUTE PIXELS",
                 _read(raw, True, False), labels, None),
                ("3. grid + Y-FLIPPED",
                 _read(raw, False, True), labels, None),
                ("4. pixels + Y-FLIPPED",
                 _read(raw, True, True), labels, None),
                ("5. to SAM (calibrated+postproc)",
                 coords2, labels2, moved)):
            panel, strip = _draw_points(img, cc, ll, title, moved=mv)
            panels.append((panel, strip))
        from PIL import Image
        gap, strip_h = 8, panels[0][1].shape[0]
        pw = panels[0][0].shape[1]
        comp = Image.new("RGB", ((pw + gap) * len(panels) - gap,
                                 H + strip_h), (10, 10, 10))
        for i, (panel, strip) in enumerate(panels):
            x0 = i * (pw + gap)
            comp.paste(Image.fromarray(strip), (x0, 0))
            comp.paste(Image.fromarray(panel), (x0, strip_h))
        out_path = (args.out if len(parsed) == 1
                    else args.out.replace(".png", f"_g{gi}.png"))
        comp.save(out_path)
        print(f"[saved] {out_path}")
    # also dump the parsed groups as JSON for easy inspection
    jpath = args.out.replace(".png", ".json")
    with open(jpath, "w", encoding="utf-8") as f:
        json.dump([{"positive": g.positive_norm,
                    "negative": g.negative_norm,
                    "hypothesis": g.hypothesis,
                    "confidence": g.confidence} for g in parsed],
                  f, ensure_ascii=False, indent=2)
    print(f"[saved] {jpath}")


if __name__ == "__main__":
    main()
