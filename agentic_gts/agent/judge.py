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


# ---------- image rendering helpers ----------

def render_topdown_image(stage_points: np.ndarray, boxes, extent: float = 0.5,
                         size: int = 320) -> np.ndarray:
    """Render a top-down 2D density image + box overlay for the VLM."""
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

    fig, ax = plt.subplots(figsize=(4, 4), dpi=size // 4)
    if len(pts):
        ax.scatter(pts[:, 0], pts[:, 1], s=0.5, alpha=0.6, c="steelblue")
    ax.axis("equal")
    ax.set_xlim(lo[0], hi[0])
    ax.set_ylim(lo[1], hi[1])
    for b in boxes:
        cs = b.corners_2d()
        poly = plt.Polygon(cs, fill=False, edgecolor="red", linewidth=1.5)
        ax.add_patch(poly)
    ax.set_xticks([]); ax.set_yticks([])
    buf = io.BytesIO()
    fig.savefig(buf, format="png", bbox_inches="tight")
    plt.close(fig)
    buf.seek(0)
    return np.array(plt.imread(buf))  # HxWx4


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


def _render_cut_z(points: np.ndarray, boxes, margin: float = 0.3) -> float:
    """Ceiling cut for top-down renders. Trusted box heights win: anything
    above the tallest device is ceiling/overhead structure by definition.
    Falls back to the z-histogram gap when no boxes are given."""
    if boxes:
        return max(b.center[2] + b.size[2] / 2.0 for b in boxes) + margin
    return _auto_ceiling_z(points[:, 2])


def render_godview_png(points: np.ndarray, boxes, max_points: int = 250_000,
                       dpi: int = 130) -> bytes:
    """Full-scene top-down render for the god-view pass: height-colored
    point cloud (dark = low, bright = high) + numbered box footprints.

    Numbering matches the index the VLM is asked about, so its answers map
    directly back to boxes. Returns PNG bytes.
    """
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
    """Best-effort JSON object extraction from a VLM reply."""
    import re
    m = re.search(r"\{.*\}", text, re.S)
    if not m:
        return None
    try:
        return json.loads(m.group(0))
    except json.JSONDecodeError:
        return None


def _png_to_b64(png_bytes: bytes) -> str:
    return base64.b64encode(png_bytes).decode("ascii")


class VLMJudge:
    def __init__(self, backend: str = "mock",
                 model: str | None = None,
                 api_base: str | None = None, api_key: str | None = None,
                 timeout: int = 60):
        self.backend = backend
        self.model = (model or os.environ.get("VLM_MODEL") or
                      "Qwen/Qwen3-VL-8B-Instruct")
        self.api_base = (api_base or os.environ.get("VLM_API_BASE") or
                         "http://127.0.0.1:8000/v1")
        self.api_key = api_key or os.environ.get("VLM_API_KEY", "EMPTY")
        self.timeout = timeout
        self._local_model = None   # lazy: (processor, model), loaded once

    # ---- backend-agnostic interface ----
    def adjudicate_box(self, scene, box, question: str,
                       options: list[str]) -> Verdict:
        """Ask the VLM a multiple-choice question about a candidate box."""
        if self.backend == "mock":
            return self._mock_adjudicate(box, question)
        if self.backend == "local":
            return self._local_adjudicate(scene, box, question, options)
        return self._qwen_adjudicate(scene, box, question, options)

    # ---- god-view global audit ----
    _GODVIEW_PROMPT = (
        "You are auditing a data-center layout. This is a TOP-DOWN view of the "
        "room: the colored scatter is the point cloud (colorbar on the right = "
        "height in meters; devices are tall ~2m structures, the floor is dark "
        "and low). The numbered rectangles are candidate device boxes.\n\n"
        "Look at the GLOBAL spatial structure: devices in machine rooms form "
        "parallel rows separated by aisles. Identify boxes that are clearly "
        "NOT real devices, e.g. a box floating in the middle of an aisle with "
        "no point structure, a box far away from every device row, or a box "
        "on the room boundary where only a wall exists.\n\n"
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
        png = render_godview_png(scene.points, boxes)
        try:
            if self.backend == "local":
                text = self._local_image_call(png, self._GODVIEW_PROMPT,
                                              max_new_tokens=400)
            else:
                text = self._qwen_image_call(png, self._GODVIEW_PROMPT,
                                              max_tokens=400)
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
                          max_new_tokens: int = 64) -> str:
        """In-process transformers call with a PNG image + text prompt."""
        from PIL import Image
        self._ensure_local_model()
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
        return processor.batch_decode(trimmed, skip_special_tokens=True)[0].strip()

    def _qwen_image_call(self, png_bytes: bytes, prompt: str,
                         max_tokens: int = 64) -> str:
        """OpenAI-compatible chat call with a base64 PNG image."""
        b64 = base64.b64encode(png_bytes).decode("ascii")
        r = requests.post(
            self.api_base + "/chat/completions",
            headers={"Authorization": f"Bearer {self.api_key}"},
            json={
                "model": self.model,
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
            timeout=self.timeout,
        )
        r.raise_for_status()
        return r.json()["choices"][0]["message"]["content"].strip()

    # ---- local in-process transformers model ----
    def _ensure_local_model(self):
        """Load the model once; subsequent adjudications reuse it."""
        if self._local_model is not None:
            return
        import torch
        import transformers
        from transformers import AutoProcessor
        try:  # Qwen3-VL needs a recent transformers; fall back to the
              # generic auto class for other VL families (Qwen2-VL, ...)
            from transformers import Qwen3VLForConditionalGeneration as ModelCls
        except ImportError:
            from transformers import AutoModelForImageTextToText as ModelCls
        print(f"[vlm][local] loading {self.model} (transformers "
              f"{transformers.__version__}) ... first call only")
        processor = AutoProcessor.from_pretrained(self.model)
        model = ModelCls.from_pretrained(
            self.model, torch_dtype=torch.bfloat16, device_map="auto")
        model.eval()
        self._local_model = (processor, model)

    def _local_adjudicate(self, scene, box, question: str,
                          options: list[str]) -> Verdict:
        try:
            import matplotlib
            matplotlib.use("Agg")
            import matplotlib.pyplot as plt
            from PIL import Image

            self._ensure_local_model()
            processor, model = self._local_model

            img_arr = render_topdown_image(scene.points, [box])
            buf = io.BytesIO()
            plt.imsave(buf, img_arr, format="png")
            image = Image.open(buf).convert("RGB")

            prompt = (
                f"You are an auditor in a data-center layout tool. Decide the best "
                f"answer for this question by looking at the top-down point cloud "
                f"image (red = a candidate box).\n\n"
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
                out = model.generate(**inputs, max_new_tokens=64,
                                      do_sample=False)
            trimmed = [o[len(i):] for i, o in zip(inputs.input_ids, out)]
            answer = processor.batch_decode(
                trimmed, skip_special_tokens=True)[0].strip()
            matched = self._match_option(answer, options)
            return Verdict(action="answer", params={"choice": matched},
                           confidence=0.8, detail=answer, raw=answer)
        except Exception as e:
            print(f"[vlm][local] inference failed ({type(e).__name__}: {e}) "
                  f"-> falling back to mock")
            return self._mock_adjudicate(box, question)

    # ---- Qwen (OpenAI-compatible chat completions with image) ----
    def _qwen_adjudicate(self, scene, box, question: str,
                         options: list[str]) -> Verdict:
        img_arr = render_topdown_image(scene.points, [box])  # HxWx4
        b64 = self._array_to_png_b64(img_arr)
        prompt = (
            f"You are an auditor in a data-center layout tool. Decide the best "
            f"answer for this question by looking at the top-down point cloud "
            f"image (red = a candidate box).\n\n"
            f"QUESTION: {question}\n"
            f"OPTIONS:\n" + "\n".join(f"- {o}" for o in options) +
            f"\n\nReply with the exact option text only."
        )
        try:
            r = requests.post(
                self.api_base + "/chat/completions",
                headers={"Authorization": f"Bearer {self.api_key}"},
                json={
                    "model": self.model,
                    "messages": [{
                        "role": "user",
                        "content": [
                            {"type": "text", "text": prompt},
                            {"type": "image_url",
                             "image_url": {"url": f"data:image/png;base64,{b64}"}},
                        ],
                    }],
                    "max_tokens": 64,
                    "temperature": 0.0,
                },
                timeout=self.timeout,
            )
            r.raise_for_status()
            text = r.json()["choices"][0]["message"]["content"].strip()
            matched = self._match_option(text, options)
            return Verdict(action="answer", params={"choice": matched},
                           confidence=0.8, detail=text, raw=text)
        except Exception as e:  # fallback to mock on any failure
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
