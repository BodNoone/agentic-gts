#!/usr/bin/env python
"""Standalone probe: VLM -> SAM BOX prompt generation, in isolation.

Feeds ONE local-view image through exactly the production path:
  adjudicate_sam_boxes (the _SAM_BOX_PROMPT call)
    -> parse_box_groups (bbox_2d on the 0-1000 grid -> pixel box)
and draws the VLM's box on the image, plus diagnostics: the raw bbox
values and the fraction of the box area that lands on the (bright)
device -- box-only local views render the device on a dark background,
so a well-grounded box covers mostly bright pixels.

Usage examples:
  # an existing rendered view (e.g. from a pipeline run dir); use the
  # CLEAN render (mask_<id>_front.png), not the wireframe overlay one
  python test_sam_points.py --image runs/xxx/mask_ab12_front.png

  # render one from a GS ply + a box (center/size in metres, yaw in deg)
  python test_sam_points.py --gs scene.ply \
      --center 3.0 0.5 1.0 --size 1.1 0.6 2.0 --yaw 0

  # no VLM server up? mock backend still shows the render + empty boxes
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
from agentic_gts.agent.mask_refine import parse_box_groups
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
    from agentic_gts.output.gs_render import make_local_cam, rasterize_gs
    from agentic_gts.tools.gs_io import read_gaussian_ply

    box = OrientedBox(center=center, size=size, yaw=math.radians(yaw_deg))
    gs = read_gaussian_ply(gs_ply)
    open_vec, corridor = (np.array([0.0, 1.0]), 1.5)
    try:
        from agentic_gts.agent.mask_refine import _open_side
        open_vec, corridor, _closed = _open_side(gs, box)
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


def _draw_box(img: np.ndarray, box_pix, title: str) -> np.ndarray:
    """Draw the pixel box (green rectangle) on a copy of the image."""
    from PIL import Image, ImageDraw
    u8 = (np.clip(img, 0, 1) * 255).astype(np.uint8)[..., :3]
    p = Image.fromarray(u8.copy())
    d = ImageDraw.Draw(p)
    x1, y1, x2, y2 = (float(v) for v in box_pix[:4])
    d.rectangle((x1, y1, x2, y2), outline=(0, 255, 0), width=3)
    strip = Image.new("RGB", (p.width, 26), (0, 0, 0))
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
    ap.add_argument("--out", default="sam_boxes_probe.png")
    args = ap.parse_args()

    if args.image:
        img = _load_image(args.image)
    elif args.gs and args.center and args.size:
        img = _render_view(args.gs, args.center, args.size, args.yaw)
    else:
        ap.error("need --image, or --gs with --center/--size")

    judge = VLMJudge(backend=args.vlm, api_base=args.vlm_base,
                     model=args.vlm_model)
    H, W = img.shape[:2]
    probe_box = OrientedBox(center=(0, 0, 0), size=(1, 1, 1), yaw=0.0)
    verdict = judge.adjudicate_sam_boxes(img, probe_box, args.view_name,
                                         png_path=args.out)
    print(f"\n[VLM] backend={judge.backend} view={args.view_name} "
          f"conf={verdict.confidence:.2f}")
    print(f"[VLM] raw reply:\n{verdict.raw or verdict.detail}\n")

    parsed = parse_box_groups(verdict.raw or "")
    if not parsed:
        print("[parse] no boxes parsed from the reply "
              "(mock backend returns none)")
    # device foreground mask: box-only views render the device on black
    fg = img[..., :3].mean(axis=2) > 0.08

    from PIL import Image
    for gi, g in enumerate(parsed):
        box_pix = g.pixel_box(W, H)
        x1, y1, x2, y2 = box_pix
        # fraction of the box area landing on the bright device
        xs = slice(int(max(0, x1)), int(min(W, x2 + 1)))
        ys = slice(int(max(0, y1)), int(min(H, y2 + 1)))
        area = (xs.stop - xs.start) * (ys.stop - ys.start)
        on_device = float(fg[ys, xs].mean()) if area else 0.0
        print(f"[group {gi}] hypothesis={g.hypothesis} "
              f"conf={g.confidence:.2f} "
              f"bbox_2d={[round(v, 1) for v in g.bbox_norm]} -> "
              f"pixels [{x1:.0f},{y1:.0f},{x2:.0f},{y2:.0f}] "
              f"({100 * (x2 - x1) / 1000:.0f}%x{100 * (y2 - y1) / 1000:.0f}% "
              f"of the frame)")
        print(f"[group {gi}] box-on-device coverage: {on_device:.2f} "
              f"(a good grounding box is mostly bright; near 0 means "
              "the box landed on the dark background)")
        panel, strip = _draw_box(
            img, box_pix, f"{args.view_name} g{gi} "
            f"(coverage {on_device:.2f})")
        gap = 8
        comp = Image.new("RGB", (W, H + strip.shape[0]), (10, 10, 10))
        comp.paste(Image.fromarray(strip), (0, 0))
        comp.paste(Image.fromarray(panel), (0, strip.shape[0]))
        out_path = (args.out if len(parsed) == 1
                    else args.out.replace(".png", f"_g{gi}.png"))
        comp.save(out_path)
        print(f"[saved] {out_path}")
    # also dump the parsed boxes as JSON for easy inspection
    jpath = args.out.replace(".png", ".json")
    with open(jpath, "w", encoding="utf-8") as f:
        json.dump([{"bbox": g.bbox_norm,
                    "hypothesis": g.hypothesis,
                    "confidence": g.confidence} for g in parsed],
                  f, ensure_ascii=False, indent=2)
    print(f"[saved] {jpath}")


if __name__ == "__main__":
    main()
