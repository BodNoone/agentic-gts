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
import re
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


def _extract_json_array(text: str):
    """Last balanced top-level JSON ARRAY in a reply, or None.

    The official Qwen3-VL grounding format is a bare array of
    {"bbox_2d": ...} items, which _extract_json (brace-scanner) cannot
    return -- hence this bracket-scanning twin.
    """
    depth, start = 0, -1
    spans = []
    in_str = False
    esc = False
    for i, ch in enumerate(text):
        if in_str:
            if esc:
                esc = False
            elif ch == "\\":
                esc = True
            elif ch == '"':
                in_str = False
            continue
        if ch == '"':
            in_str = True
        elif ch == "[":
            if depth == 0:
                start = i
            depth += 1
        elif ch == "]":
            depth -= 1
            if depth == 0 and start >= 0:
                spans.append(text[start:i + 1])
    for c in reversed(spans):
        try:
            arr = json.loads(c)
        except json.JSONDecodeError:
            continue
        if isinstance(arr, list):
            return arr
    return None


def _parse_ground_regions(text: str, W: int, H: int) -> list[tuple]:
    """Parse a grounding reply into pixel rects [(x0, y0, x1, y1, label)].

    Accepts the OFFICIAL Qwen3-VL grounding format (per the 2d_grounding
    cookbook): a JSON array of {"bbox_2d": [x1, y1, x2, y2], "label":
    ...} in RELATIVE 0-1000 coordinates -- the model's trained output
    distribution, which is why the prompt asks for it verbatim. The
    legacy {"regions": [{"x0", "y0", "x1", "y1"}]} dict with absolute
    pixels is still honoured (a reply that ignores the format and
    happens to use small pixel values is ambiguous; relative-first is
    the correct default since that is what was asked for). The label
    is kept (default "device") for the official-style audit plot.

    SALVAGE: real replies on row-heavy rooms arrive TRUNCATED (30+
    regions overflow the generation budget) and malformed (objects
    wrapped in parentheses instead of a JSON array, "bbox 2d" /
    "bbox _2d" key typos) -- the structural parse then finds nothing.
    The salvage scanner pulls every COMPLETE bbox object straight out
    of the raw text; a truncated tail yields no match. What survived
    is good grounding evidence: using it beats dropping everything
    (user report: an entire grounding run silently failed this way).
    """
    items = []
    data = _extract_json(text)
    if isinstance(data, list):
        items = data
    elif isinstance(data, dict):
        items = data.get("regions") or data.get("boxes") or []
    if not items:
        items = _extract_json_array(text) or []
    rects = []
    for item in items if isinstance(items, list) else []:
        if not isinstance(item, dict):
            continue
        label = str(item.get("label", "device"))[:20] or "device"
        bbox = item.get("bbox_2d")
        if bbox is not None:
            try:
                x0, y0, x1, y1 = (float(v) for v in bbox[:4])
            except (TypeError, ValueError):
                continue
            if max(x0, y0, x1, y1) > 1000.0:
                pass          # absolute pixels despite the format spec
            else:             # official relative 0-1000 grid -> pixels
                x0, x1 = x0 / 1000.0 * W, x1 / 1000.0 * W
                y0, y1 = y0 / 1000.0 * H, y1 / 1000.0 * H
        else:
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
            rects.append((x0, y0, x1, y1, label))
    if not rects:
        rects = _salvage_bboxes(text, W, H)
    return rects


_SALV_BBOX = re.compile(
    r'["\']?bbox[\s_]*2d["\']?\s*:\s*\[\s*(-?[\d.]+\s*,\s*-?[\d.]+\s*,'
    r'\s*-?[\d.]+\s*,\s*-?[\d.]+)\s*\]')
_SALV_LABEL = re.compile(r'["\']?label["\']?\s*:\s*["\']([^"\']{0,20})')


def _salvage_bboxes(text: str, W: int, H: int) -> list[tuple]:
    """Scan the RAW text for complete bbox objects (see the salvage
    note in _parse_ground_regions). Tolerates paren-wrapped objects,
    key typos ("bbox 2d", "bbox _2d"), and a truncated tail. A nearby
    label (within 80 chars after the bbox) is picked up when present."""
    rects = []
    for m in _SALV_BBOX.finditer(text or ""):
        try:
            x0, y0, x1, y1 = (float(v) for v in m.group(1).split(","))
        except ValueError:
            continue
        if max(abs(x0), abs(y0), abs(x1), abs(y1)) <= 1000.0:
            x0, x1 = x0 / 1000.0 * W, x1 / 1000.0 * W
            y0, y1 = y0 / 1000.0 * H, y1 / 1000.0 * H
        lm = _SALV_LABEL.search(text, m.end(), m.end() + 80)
        label = (lm.group(1).strip() if lm else "") or "device"
        x0, x1 = min(x0, x1), max(x0, x1)
        y0, y1 = min(y0, y1), max(y0, y1)
        x0, y0 = max(0.0, x0), max(0.0, y0)
        x1, y1 = min(float(W), x1), min(float(H), y1)
        if x1 - x0 >= 8.0 and y1 - y0 >= 8.0:
            rects.append((x0, y0, x1, y1, label))
    return rects


def _png_to_b64(png_bytes: bytes) -> str:
    return base64.b64encode(png_bytes).decode("ascii")


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

    # ---- global 2D grounding (rows as whole regions) ----
    # Prompt style follows the OFFICIAL 2d_grounding cookbook verbatim:
    # "Locate every instance that belongs to the following categories:
    # ... Report bbox coordinates in JSON format like this:
    # {\"bbox_2d\": [x1, y1, x2, y2], \"label\": ...}". Multi-target
    # grounding in relative 0-1000 coords is a TRAINED capability -- the
    # model needs the categories and the JSON template ONLY. Explaining
    # the coordinate system or dictating reply structure (as earlier
    # drafts did) is off-distribution instruction the model must
    # second-guess.
    _GROUND_PROMPT = (
        "This is a top-down view of a data-center room with the ceiling "
        "removed: rows of tall server racks appear as solid bright "
        "bands, aisles are dark, walls are thin lines at the room "
        "boundary.\n"
        "Locate every instance that belongs to the following categories: "
        '"cabinet, server rack, air conditioning". A continuous row of '
        "joined cabinets is ONE instance whose box covers the WHOLE row "
        "(do not split it into individual cabinets); structures separated "
        "by an aisle or a clear gap are separate instances. Do not "
        "include walls, pillars, columns, or floor clutter.\n"
        "Report bbox coordinates in JSON format like this: "
        '{"bbox_2d": [x1, y1, x2, y2], "label": "server rack"}'
    )

    def ground_regions(self, png: bytes, W: int, H: int,
                       png_path: str | None = None) -> list[tuple]:
        """2D grounding over the global top-down view: outline EVERY
        device structure (a joined row = one region).

        Returns pixel rects [(x0, y0, x1, y1)] or [] on mock / failure.
        Runs on the thinking tier when configured: one call per view,
        and these regions BECOME the pipeline's boxes (high stakes)."""
        prompt = self._GROUND_PROMPT
        if self.backend == "mock":
            return []
        use_thinking = bool(self.thinking_model)
        try:
            for thinking in ((True, False) if use_thinking else (False,)):
                try:
                    # generous budget: row-heavy rooms return 30+
                    # regions; the old 900/2048 caps TRUNCATED the
                    # reply mid-item and the whole grounding silently
                    # failed (user report)
                    if self.backend == "local":
                        text = self._local_image_call(
                            png, prompt, max_new_tokens=6000,
                            thinking=thinking)
                    else:
                        text = self._qwen_image_call(
                            png, prompt, max_tokens=6000,
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
        rects = _parse_ground_regions(text or "", W, H)
        if not rects:
            print(f"[vlm][ground] unparseable reply -> no grounding: "
                  f"{(text or '')[:200]!r}")
            # persist the FULL reply next to the evidence png: 200 chars
            # on the console is not enough to debug why the model's
            # grounding output does not parse (user needs the raw text)
            if png_path:
                try:
                    import os as _os
                    rp = _os.path.splitext(png_path)[0] + "_reply.txt"
                    with open(rp, "w", encoding="utf-8") as rf:
                        rf.write(text or "")
                    print(f"[vlm][ground] full raw reply -> {rp}")
                except Exception as _e:
                    print(f"[vlm][ground] reply dump failed "
                          f"({type(_e).__name__})")
        self._record("ground", prompt, text or "",
                    f"{len(rects)} regions", 0.5, "", png_path=png_path)
        return rects

    # Prompt style follows the OFFICIAL 2d_grounding cookbook verbatim
    # (same lesson as _GROUND_PROMPT above): categories + the JSON
    # template ONLY. Explaining the coordinate system or dictating a
    # custom reply structure (the earlier candidate_groups draft) is
    # off-distribution instruction the model must second-guess. The
    # official {"bbox_2d": ..., "label": ...} array is the trained
    # output; parse_box_groups accepts it natively (top-level array,
    # label -> hypothesis).
    # The splitting rules replace the removed split_stage: a joined row
    # whose cabinets differ in height or color must be grounded as
    # SEPARATE instances (the row split now comes from this grounding,
    # not a separate VLM pass); an OPEN door is its OWN positive class
    # (user finding: the model DETECTS "open cabinet door" reliably as
    # a detection task, but cannot EXCLUDE it via a negative
    # instruction) -- the door boxes feed pixel-level subtraction so
    # door points never enter the span/thickness pools.
    _SAM_BOX_PROMPT = (
        "This is a local {view_name} view of one target device in a "
        "data-center room, rendered clean on a dark background: the "
        "bright structure filling most of the frame IS the target.\n"
        "First judge whether the view is USABLE: begin your reply with "
        "the single line 'quality: good' when the device structure is "
        "visible and recognizable, or 'quality: poor' ONLY when you "
        "cannot see the device or cannot tell what it is (haze or "
        "blur fully hiding the structure, an empty or unrecognizable "
        "frame). Minor rendering imperfections are NOT poor.\n"
        "Locate every instance that belongs to the following categories: "
        '"server rack / IT cabinet, air-conditioning unit, '
        'open cabinet door, cable ladder, top cable".\n'
        "Instance rules: cabinets joined side by side in one row are "
        "DIFFERENT instances when they differ in height or in color -- "
        "give each its own box at its own boundary; truly identical "
        "joined cabinets may be covered by one box.\n"
        "A cabinet door standing open, swung out of the body, is its "
        "OWN instance labelled \"open cabinet door\" -- the box covers "
        "ONLY the door panel itself, NOT the cabinet body behind it.\n"
        "A cable ladder (vertical ladder rack / cable tray running up "
        "beside or behind the device) is its OWN instance labelled "
        "\"cable ladder\" -- the box covers ONLY the ladder itself, "
        "never any part of a rack or cabinet.\n"
        "A bundle of cables running across or connected to the TOP of "
        "the device (cable connections above the rack tops) is its OWN "
        "instance labelled \"top cable\" -- the box covers ONLY the "
        "cable bundle, never any part of a rack or cabinet.\n"
        "Each box must cover the whole visible instance it belongs to.\n"
        "Report bbox coordinates in JSON format like this: "
        '{"bbox_2d": [x1, y1, x2, y2], "label": "rack"}'
    )

    def adjudicate_sam_boxes(self, image: np.ndarray, box,
                             view_name: str,
                             png_path: str | None = None) -> Verdict:
        """Qwen3-VL box grounding for SAM's box prompt (native task)."""
        from agentic_gts.agent.mask_refine import (parse_box_groups,
                                                   reply_view_quality)
        # .replace, NOT .format: the prompt's JSON example carries
        # literal braces ({"candidate_groups": ...}) that str.format
        # parses as a replacement field named '"candidate_groups"'
        # (quotes included) -> KeyError on EVERY real-VLM call (mock
        # never formats, so the tests could not catch it)
        prompt = self._SAM_BOX_PROMPT.replace("{view_name}", view_name)
        if self.backend == "mock":
            return Verdict(action="keep", params={"groups": []},
                           confidence=0.0, detail="mock: no SAM boxes")
        png = self._array_png_bytes(image)
        if png_path is None:
            png_path = self._save_evidence_png(
                image, f"sam_boxes_{box.box_id}_{view_name}.png")
        try:
            # generous budget: a LONG joined row grounds dozens of
            # cabinets, each its own bbox_2d item (plus the door
            # instances). The old 800-token cap truncated the reply
            # mid-item on such rows (user report) -- parse_box_groups
            # salvage-recovers the COMPLETE boxes, but the tail
            # cabinets were still lost. Mirrors ground_regions' budget.
            if self.backend == "local":
                text = self._local_image_call(png, prompt,
                                              max_new_tokens=6000)
            else:
                text = self._qwen_image_call(png, prompt, max_tokens=6000)
        except Exception as e:
            self._record("sam_boxes", prompt, "", "", 0.0,
                         f"call failed: {e}", png_path=png_path)
            return Verdict(action="keep", params={"groups": []},
                           confidence=0.0, detail=f"call failed: {e}")
        parsed = parse_box_groups(self._strip_think(text))
        quality = reply_view_quality(self._strip_think(text))
        groups = [{"bbox": g.bbox_norm,
                   "hypothesis": g.hypothesis,
                   "confidence": g.confidence} for g in parsed]
        conf = max((g.confidence for g in parsed), default=0.0)
        self._record("sam_boxes", prompt, text,
                     f"{len(groups)} groups (view {quality})", conf,
                     "bbox_2d normalized 0-1000; converted once to "
                     "pixels for SAM's box prompt",
                     png_path=png_path)
        return Verdict(action="segment" if groups else "keep",
                       params={"groups": groups,
                               "view_quality": quality}, confidence=conf,
                       detail=f"{len(groups)} prompt groups "
                              f"(view {quality})", raw=text,
                       png_path=png_path)

    _RACK_CONFIRM_PROMPT = (
        "You are verifying ONE detected object in a data-center scene.\n"
        "The image shows the local neighborhood of one detected 3D box; "
        "the RED wireframe marks the box. Question: is the object the "
        "wireframe wraps really a piece of DC EQUIPMENT -- a SERVER RACK "
        "/ IT cabinet (or a joined row of them) OR an air-conditioning "
        "unit (CRAC / precision cooling)? A pillar, wall segment, cable "
        "tray, UPS unit, pipe, floor patch or any other clutter is NOT "
        "equipment even when the box fits it well. Judge the object, "
        "not the box fit.\n"
        "Output ONLY JSON on the last line:\n"
        '{"is_rack": true|false, "confidence": 0.0-1.0}'
    )

    def adjudicate_rack_confirm(self, image: np.ndarray, box,
                                png_path: str | None = None) -> Verdict:
        """Type-level guard: is the boxed object DC equipment (rack or AC)?

        The grounding guards only reject hallucinated EMPTY regions
        (no point support / floor patches); a real structure mislabelled
        equipment (pillar, UPS, wall) passes them all. One yes/no
        question on the local view. The caller NEVER deletes on a 'no'
        -- it marks LOW confidence and surfaces the box for human
        review (false-positive deletion is the dangerous direction).
        """
        if self.backend == "mock":
            return Verdict(action="keep", params=None, confidence=0.0,
                           detail="mock: no type signal")
        png = self._array_png_bytes(image)
        if png_path is None:
            png_path = self._save_evidence_png(
                image, f"rack_confirm_{box.box_id}.png")
        try:
            if self.backend == "local":
                text = self._local_image_call(png, self._RACK_CONFIRM_PROMPT,
                                              max_new_tokens=200)
            else:
                text = self._qwen_image_call(png, self._RACK_CONFIRM_PROMPT,
                                             max_tokens=200)
        except Exception as e:
            self._record("rack_confirm", self._RACK_CONFIRM_PROMPT, "", "",
                         0.0, f"call failed: {e}", png_path=png_path)
            return Verdict(action="keep", params=None, confidence=0.0,
                           detail=f"call failed: {e}")
        p = self._parse_rack_confirm(self._strip_think(text))
        self._record("rack_confirm", self._RACK_CONFIRM_PROMPT, text,
                     str(p), p["confidence"] if p else 0.0,
                     "no deletion on a 'no' -- LOW + human review",
                     png_path=png_path)
        if p is None:
            return Verdict(action="keep", params=None, confidence=0.0,
                           detail="unparseable", raw=text, png_path=png_path)
        return Verdict(action="keep", params=p, confidence=p["confidence"],
                       raw=text, png_path=png_path)

    @staticmethod
    def _parse_rack_confirm(text: str) -> dict | None:
        """Parse the rack yes/no JSON. Tolerates string booleans and
        missing confidence; None when no verdict can be extracted."""
        data = _extract_json(text)
        if not isinstance(data, dict) or "is_rack" not in data:
            return None
        v = data["is_rack"]
        if isinstance(v, bool):
            is_rack = v
        else:
            is_rack = str(v).strip().lower() in ("true", "yes", "1")
        try:
            conf = min(max(float(data.get("confidence", 0.5)), 0.0), 1.0)
        except (TypeError, ValueError):
            conf = 0.5
        return {"is_rack": is_rack, "confidence": conf}

    _SIDE_PICK_PROMPT = (
        "This image contains several square panels side by side; each "
        "panel has a big YELLOW letter (A, B, C...) in its top-left "
        "corner. Every panel renders the SAME device (a server rack / "
        "IT cabinet row or an air-conditioning unit) from a DIFFERENT "
        "camera position. Pick the panel where the device is shown "
        "most CLEARLY: crisp structure, panels and edges visible. A "
        "smooth grey veil, a blurred haze or a nearly empty frame is a "
        "BAD panel. Reply with ONLY the letter of the best panel."
    )

    def adjudicate_side_pick(self, image: np.ndarray, box,
                             n_panels: int,
                             png_path: str | None = None) -> Verdict:
        """Pick the clearest side-view candidate (user direction: the
        geometric rules for where the side camera stands keep
        misjudging which end is clear -- let the VLM look at the
        actual renders). One tiny call: panels labeled A.. in one
        image, reply one letter. params["pick"] is the panel index or
        None (unparseable -> the caller's rule order stands)."""
        import re
        if self.backend == "mock" or n_panels < 1:
            return Verdict(action="keep", params={"pick": None},
                           confidence=0.0,
                           detail="mock: no side-pick signal")
        png = self._array_png_bytes(image)
        if png_path is None:
            png_path = self._save_evidence_png(
                image, f"side_pick_{box.box_id}.png")
        try:
            if self.backend == "local":
                text = self._local_image_call(
                    png, self._SIDE_PICK_PROMPT, max_new_tokens=16)
            else:
                text = self._qwen_image_call(
                    png, self._SIDE_PICK_PROMPT, max_tokens=16)
        except Exception as e:
            self._record("side_pick", self._SIDE_PICK_PROMPT, "", "",
                         0.0, f"call failed: {e}", png_path=png_path)
            return Verdict(action="keep", params={"pick": None},
                           confidence=0.0, detail=f"call failed: {e}")
        body = self._strip_think(text)
        pick = None
        last = chr(ord("A") + n_panels - 1)
        m = re.match(r"\s*([A-%s])\b" % last, body.strip())
        if not m:
            m = re.search(r"\b([A-%s])\b" % last, body)
        if m:
            pick = ord(m.group(1)) - ord("A")
        self._record("side_pick", self._SIDE_PICK_PROMPT, body,
                    chr(ord("A") + pick) if pick is not None else "?",
                    1.0 if pick is not None else 0.0,
                    "panel letter -> side-view candidate",
                    png_path=png_path)
        return Verdict(action="keep", params={"pick": pick},
                       confidence=1.0 if pick is not None else 0.0,
                       detail=f"panel {chr(ord('A') + pick)}"
                       if pick is not None else "unparseable",
                       raw=text, png_path=png_path)

    @staticmethod
    def _array_png_bytes(arr: np.ndarray) -> bytes:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        buf = io.BytesIO()
        plt.imsave(buf, arr, format="png")
        return buf.getvalue()

    # ---- shared image-call helpers (grounding / split / sam / rack-confirm) ----
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
                # vLLM honours a per-request seed: even greedy decoding
                # varies run-to-run under continuous batching (kernel
                # order), and downstream geometry (yaw feedback votes,
                # cluster verdicts) is knife-edged on those flips
                # (user report: mesh yaw right on some runs, wrong on
                # others with identical input)
                "seed": 0,
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
