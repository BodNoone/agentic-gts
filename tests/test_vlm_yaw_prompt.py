"""VLM capability probe: yaw-direction correction with a custom prompt.

Manually run (needs a live VLM server, e.g. vLLM serving
Qwen3-VL-8B-Instruct at http://127.0.0.1:8000/v1):

    python tests/test_vlm_yaw_prompt.py [--err-deg 12] [--true-yaw 0]
                                        [--rounds 2] [--out runs/vlm_yaw_test]

What it does:
  1. synthesizes a 5-cabinet row at a known yaw, and ONE candidate box
     whose yaw is off by --err-deg (the only planted error);
  2. renders the pipeline's local evidence image (red footprint +
     green +x / blue +y arrows; scatter fallback without a CUDA
     rasterizer);
  3. asks the VLM the user's CORE question on top of the pipeline's
     local-view background text;
  4. parses the replied rotation (signed degrees, ccw positive seen
     from above -- matches the right-handed yaw convention) and applies
     it to the box;
  5. reports the residual yaw error and the geometric support before
     vs after, and saves before/after renders for eyeballing.

This isolates ONE skill: can the VLM read an orientation error from the
evidence image and express the correction with the right SIGN.
"""
import argparse
import json
import math
import os
import re
import sys

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from agentic_gts.agent.judge import (VLMJudge, _LOCAL_VIEW_DESC,
                                     render_topdown_image)
from agentic_gts.core.models import OrientedBox, Scene
from agentic_gts.tools.geometry import support_fraction

PROMPT = (
    f"{_LOCAL_VIEW_DESC}\n\n"
    "Does the red box is a good 3d grounding result? If not, please "
    "answer me how many degrees should I rotate this box around the "
    "z-axis first?\n\n"
    "Conventions: the GREEN arrow on the footprint is the box's local x "
    "(length) axis, the BLUE arrow is its local y (depth) axis. Only "
    "rotation around the vertical z-axis matters. Express the correction "
    "as a signed number of degrees: positive = counterclockwise when "
    "seen from above, negative = clockwise. Answer with ONE short "
    "sentence of reasoning, then state the angle, e.g. 'the green axis "
    "leans clockwise relative to the cabinet row; rotate -12 degrees'. "
    "If the box already fits the device, answer exactly '0 degrees'."
)


def make_row_points(n_cabinets: int = 5, w: float = 0.6, d: float = 1.1,
                    h: float = 2.3, pitch: float = 0.65,
                    yaw_deg: float = 0.0, step: float = 0.05,
                    seed: int = 0) -> np.ndarray:
    """Surface points of one cabinet row (front/back faces, end caps,
    tops) in the world frame; the row is rotated by yaw_deg."""
    rng = np.random.default_rng(seed)
    parts = []          # each: Nx3 surface point block (row-local frame)
    for i in range(n_cabinets):
        x0 = (i - (n_cabinets - 1) / 2.0) * pitch
        x1 = x0 + w
        # front/back faces (+y / -y): the two dense observed surfaces
        for y in (d / 2, -d / 2):
            gx, gz = np.meshgrid(np.arange(x0, x1, step),
                                 np.arange(0.05, h, step))
            parts.append(np.column_stack(
                [gx.ravel(), np.full(gx.size, y), gz.ravel()]))
        # end caps
        for x in (x0, x1):
            gy, gz = np.meshgrid(np.arange(-d / 2, d / 2, step),
                                 np.arange(0.05, h, step))
            parts.append(np.column_stack(
                [np.full(gy.size, x), gy.ravel(), gz.ravel()]))
        # top face
        gx, gy = np.meshgrid(np.arange(x0, x1, step),
                             np.arange(-d / 2, d / 2, step))
        parts.append(np.column_stack(
            [gx.ravel(), gy.ravel(), np.full(gx.size, h)]))
    pts = np.vstack(parts)
    pts = pts + rng.normal(0, 0.003, pts.shape)   # surface jitter
    yaw = math.radians(yaw_deg)
    R = np.array([[math.cos(yaw), -math.sin(yaw)],
                  [math.sin(yaw), math.cos(yaw)]])
    pts[:, :2] = pts[:, :2] @ R.T
    return pts


def parse_rotation_deg(text: str) -> float:
    """Signed rotation from the VLM's reply. ccw (seen from above) is
    positive; clockwise negative; no number -> 0; clamped to +-45 to
    bound hallucinations."""
    t = text.lower()
    if not re.search(r"\d", t):
        return 0.0
    m = re.search(r"([+-]?\d+(?:\.\d+)?)\s*(?:degrees?|deg|°)", t)
    if m is None:
        m = re.search(r"([+-]?\d+(?:\.\d+)?)", t)
    if m is None:
        return 0.0
    deg = float(m.group(1))
    if re.search(r"counterclockwise|anti-?clockwise|\bccw\b", t):
        deg = abs(deg)
    elif re.search(r"(?<!counter)clockwise|(?<![a-z])cw(?![a-z])", t):
        deg = -abs(deg)
    return max(-45.0, min(45.0, deg))


def _save_img(arr, path):
    plt.imsave(path, arr)
    return path


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--err-deg", type=float, default=12.0,
                    help="planted yaw error of the candidate box (deg)")
    ap.add_argument("--true-yaw", type=float, default=0.0,
                    help="true row yaw in the world frame (deg)")
    ap.add_argument("--rounds", type=int, default=2,
                    help="render -> ask -> rotate iterations")
    ap.add_argument("--out", default="runs/vlm_yaw_test")
    ap.add_argument("--vlm-base", default=None)
    ap.add_argument("--vlm-model", default=None)
    args = ap.parse_args()
    os.makedirs(args.out, exist_ok=True)

    # live-server probe with a clear message instead of a stack trace
    import requests
    base = (args.vlm_base or os.environ.get("VLM_API_BASE")
            or "http://127.0.0.1:8000/v1")
    try:
        r = requests.get(base + "/models", timeout=5)
        r.raise_for_status()
        models = [m.get("id") for m in r.json().get("data", [])]
        print(f"[vlm] server up at {base}, models: {models}")
    except Exception as e:
        sys.exit(f"[abort] no VLM server at {base} ({type(e).__name__}: {e})"
                 f"\n        start it first, e.g. "
                 f"vllm serve Qwen/Qwen3-VL-8B-Instruct --port 8000")

    pts = make_row_points(yaw_deg=args.true_yaw)
    scene = Scene(points=pts)
    true_yaw = math.radians(args.true_yaw)

    # candidate: correct center/size, yaw off by err-deg, on the MIDDLE
    # cabinet (neighbours give the row rhythm the VLM judges against)
    mid = (5 - 1) / 2.0 * 0.65
    yaw = math.radians(args.true_yaw + args.err_deg)
    box = OrientedBox(center=(mid * math.cos(yaw), mid * math.sin(yaw),
                              1.15),
                      size=(0.6, 1.1, 2.3), yaw=yaw)

    judge = VLMJudge(backend="qwen", api_base=base,
                     model=args.vlm_model or os.environ.get("VLM_MODEL"))

    history = []
    for rnd in range(args.rounds):
        err_deg = math.degrees(box.yaw - true_yaw)
        sup = support_fraction(scene, box)
        img = render_topdown_image(pts, [box], extent=1.6,
                                   overlay="wire3d_axes")
        _save_img(img, os.path.join(args.out, f"round{rnd}_before.png"))
        png = VLMJudge._array_png_bytes(img)
        try:
            text = judge._qwen_image_call(png, PROMPT, max_tokens=512)
        except Exception as e:
            print(f"[abort] VLM call failed: {type(e).__name__}: {e}")
            return
        deg = parse_rotation_deg(text)
        print(f"[round {rnd}] yaw err={err_deg:+.1f}deg support={sup:.2f}")
        print(f"[round {rnd}] VLM: {text!r}")
        print(f"[round {rnd}] parsed rotation: {deg:+.1f}deg")
        box = OrientedBox(center=box.center, size=box.size,
                          yaw=box.yaw + math.radians(deg))
        new_err = math.degrees(box.yaw - true_yaw)
        img2 = render_topdown_image(pts, [box], extent=1.6,
                                    overlay="wire3d_axes")
        _save_img(img2, os.path.join(args.out, f"round{rnd}_after.png"))
        print(f"[round {rnd}] -> applied, residual err={new_err:+.1f}deg, "
              f"support={support_fraction(scene, box):.2f}")
        history.append({"round": rnd, "err_before": err_deg,
                        "reply": text, "parsed_deg": deg,
                        "err_after": new_err})
        if abs(new_err) < 1.5:
            print(f"[round {rnd}] converged")
            break

    with open(os.path.join(args.out, "history.json"), "w",
              encoding="utf-8") as f:
        json.dump(history, f, ensure_ascii=False, indent=2)
    first, last = history[0], history[-1]
    print(f"\n[result] initial err {first['err_before']:+.1f}deg -> "
          f"final err {last['err_after']:+.1f}deg "
          f"(true correction would have been "
          f"{-first['err_before']:+.1f}deg)")
    verdict = "PASS" if abs(last["err_after"]) < abs(first["err_before"]) * 0.4 \
        else ("OK" if abs(last["err_after"]) < abs(first["err_before"])
              else "FAIL")
    print(f"[result] {verdict}: evidence images + history in {args.out}")


if __name__ == "__main__":
    main()
