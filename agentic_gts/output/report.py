"""Per-box local-view + VLM-verdict HTML report.

Reads a run directory (boxes.json + vlm_records.jsonl + evidence PNGs),
renders a fresh local three-view composite for EVERY final box, attaches
the VLM verdicts each box received (matched via the evidence-image
filenames), and writes one self-contained HTML file (images inlined as
base64) -- open it in any browser, share it, no server needed.

Sections:
  - overview (box count, verdict counts, god-view before/after)
  - one card per final box: local view + its verdict timeline
  - verdicts about boxes that no longer exist (deleted candidates)
"""
from __future__ import annotations

import base64
import html as _html
import json
import os
import re

import numpy as np

_KIND_LABEL = {"fit": "姿态精修", "box": "box 判定", "pair": "配对判定"}
_EVIDENCE_RE = re.compile(r"(?:fit_|pair_)?evidence_([0-9a-f]{8})"
                          r"(?:_([0-9a-f]{8}))?\.png")


def _b64_file(path: str) -> str | None:
    try:
        with open(path, "rb") as f:
            return base64.b64encode(f.read()).decode("ascii")
    except OSError:
        return None


def _b64_array(arr: np.ndarray) -> str | None:
    """PNG base64 from an HxWx3 float image array."""
    try:
        import io
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        buf = io.BytesIO()
        plt.imsave(buf, np.clip(arr[..., :3], 0, 1), format="png")
        return base64.b64encode(buf.getvalue()).decode("ascii")
    except Exception:
        return None


def _load_records(run_dir: str) -> list[dict]:
    recs = []
    path = os.path.join(run_dir, "vlm_records.jsonl")
    if not os.path.exists(path):
        return recs
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                recs.append(json.loads(line))
            except json.JSONDecodeError:
                continue
    return recs


def _record_box_ids(rec: dict) -> list[str]:
    """Box-id prefixes a record is about, parsed from its image filename
    (evidence_{a}.png / fit_evidence_{a}.png / pair_evidence_{a}_{b}.png)."""
    img = rec.get("image") or ""
    if not img:
        return []
    m = _EVIDENCE_RE.search(os.path.basename(img))
    if not m:
        return []
    return [g for g in m.groups() if g]


def _confidence_chip(conf: float) -> str:
    conf = float(conf) if conf is not None else 0.5
    if conf >= 0.6:
        color, bg = "#1a7f37", "#e6f4ea"
    elif conf >= 0.4:
        color, bg = "#b26a00", "#fff4e0"
    else:
        color, bg = "#c62828", "#fdecea"
    return (f'<span style="color:{color};background:{bg};padding:1px 8px;'
            f'border-radius:9px;font-size:12px">置信度 {conf:.2f}</span>')


def _quality_chip(quality) -> str:
    """Render-quality badge for a verdict record (per-slot scores from the
    no-reference scorer). Shows the WORST slot -- that is the evidence the
    verdict should be least trusted on."""
    if not isinstance(quality, dict):
        return ""
    scores = [v.get("score") for v in quality.values() if isinstance(v, dict)]
    if not scores:
        return ""
    worst = min(scores)
    if worst >= 0.5:
        color = "#1a7f37"
    elif worst >= 0.35:
        color = "#b26a00"
    else:
        color = "#c62828"
    slots = " ".join(
        f"{k} {v.get('score', 0):.2f}"
        + (f" vis {float(v['visibility']):.2f}" if "visibility" in v else "")
        for k, v in quality.items() if isinstance(v, dict))
    return (f'<span title="{_html.escape(slots)}" '
            f'style="color:{color};font-size:12px">渲染质量 ≥ {worst:.2f}</span>')


def _verdict_block(rec: dict, run_dir: str, idx: int) -> str:
    kind = str(rec.get("kind", "?"))
    label = _KIND_LABEL.get(kind, kind)
    ids = _record_box_ids(rec)
    ids_html = (f'<span style="font-size:12px;color:#888">box '
                f'<code>{" / ".join(ids)}</code></span>' if ids else "")
    q = _html.escape(str(rec.get("prompt", ""))[:600])
    ans = _html.escape(str(rec.get("answer", "")) or str(rec.get("detail", "")))
    choice = _html.escape(str(rec.get("choice", "")) or "—")
    img = rec.get("image") or ""
    img_b64 = _b64_file(os.path.join(run_dir, os.path.basename(img))) if img else None
    img_html = (f'<img loading="lazy" src="data:image/png;base64,{img_b64}" '
                f'style="max-width:100%;border:1px solid #ddd;border-radius:4px">'
                if img_b64 else "")
    return f"""
    <div style="border:1px solid #e0e0e0;border-radius:6px;padding:8px;margin:6px 0;
                background:#fafafa">
      <div style="display:flex;gap:8px;align-items:center;flex-wrap:wrap">
        <span style="background:#37474f;color:#fff;padding:1px 8px;border-radius:4px;
                     font-size:12px">{_html.escape(label)}</span>
        <b>判定: {_html.escape(choice)}</b>
        {_confidence_chip(rec.get("confidence", 0.5))}
        {_quality_chip(rec.get("quality"))}
        {ids_html}
      </div>
      <div style="margin:4px 0;font-size:13px;color:#333">
        回答: <code style="word-break:break-all">{ans}</code>
      </div>
      <details><summary style="font-size:12px;color:#666;cursor:pointer">问题原文</summary>
        <pre style="white-space:pre-wrap;font-size:11px;color:#555">{q}</pre>
      </details>
      {img_html}
    </div>"""


def build_report(run_dir: str, out_path: str | None = None,
                 points: np.ndarray | None = None,
                 gs_ply: str | None = None) -> str:
    """Build the HTML report for a run directory. Returns the output path."""
    from agentic_gts.core.models import Scene
    from agentic_gts.agent.judge import render_topdown_image

    out_path = out_path or os.path.join(run_dir, "vlm_report.html")
    boxes_path = os.path.join(run_dir, "boxes.json")
    scene = Scene(points=points if points is not None else np.zeros((0, 3)))
    if os.path.exists(boxes_path):
        scene.load_boxes(boxes_path)
    records = _load_records(run_dir)

    final_ids = {b.box_id[:8] for b in scene.boxes}
    per_box: dict[str, list[dict]] = {b.box_id[:8]: [] for b in scene.boxes}
    orphans: list[dict] = []
    for rec in records:
        ids = _record_box_ids(rec)
        if ids and all(i in final_ids for i in ids):
            for i in ids:
                per_box[i].append(rec)
        else:
            orphans.append(rec)

    # ---------------- HTML assembly ----------------
    parts = ["""<!DOCTYPE html><html><head><meta charset="utf-8">
<title>agentic-gts VLM 判定报告</title></head>
<body style="font-family:system-ui,-apple-system,'Segoe UI',sans-serif;
             margin:0;background:#f5f5f5">
<div style="max-width:1200px;margin:0 auto;padding:16px">"""]

    kind_counts = {}
    for r in records:
        k = str(r.get("kind", "?"))
        kind_counts[k] = kind_counts.get(k, 0) + 1
    parts.append(
        f"<h2>agentic-gts VLM 判定报告</h2>"
        f"<p>最终 box 数: <b>{len(scene.boxes)}</b> · "
        f"判定记录: <b>{len(records)}</b> "
        f"({' · '.join(f'{_KIND_LABEL.get(k, k)} {v}' for k, v in kind_counts.items()) or '无'}) · "
        f"已删除候选的记录: <b>{len(orphans)}</b></p>")
    if not records:
        parts.append('<p style="color:#b26a00">本次运行没有 VLM 判定记录'
                     '（mock 后端或无 issue 触发）。</p>')

    # god views
    for name, cap in (("godview.png", "修复前 god-view"),
                      ("godview_final.png", "终审 god-view")):
        b64 = _b64_file(os.path.join(run_dir, name))
        if b64:
            parts.append(f"<h3>{_html.escape(cap)}</h3>"
                         f'<img loading="lazy" src="data:image/png;base64,{b64}" '
                         f'style="max-width:100%;border:1px solid #ccc;'
                         f'border-radius:6px">')

    # per-box cards
    parts.append("<h3>每个 box 的局部视角与判定记录</h3>")
    parts.append('<div style="display:grid;grid-template-columns:'
                  'repeat(auto-fill,minmax(560px,1fr));gap:14px">')
    for i, b in enumerate(scene.boxes):
        try:
            img = render_topdown_image(scene.points, [b], gs_ply=gs_ply)
            view_b64 = _b64_array(img)
        except Exception:
            view_b64 = None
        view_html = (f'<img loading="lazy" src="data:image/png;base64,{view_b64}" '
                     f'style="width:100%;border:1px solid #ccc;border-radius:6px">'
                     if view_b64 else
                     '<div style="color:#999;padding:12px">局部视角渲染失败</div>')
        verdicts = per_box.get(b.box_id[:8], [])
        v_html = ("".join(_verdict_block(r, run_dir, j)
                          for j, r in enumerate(verdicts))
                  or '<div style="color:#999;font-size:13px;padding:4px 0">'
                    '无判定记录（未触发任何 issue）</div>')
        yaw_deg = float(np.degrees(b.yaw))
        parts.append(f"""
        <div style="background:#fff;border:1px solid #ddd;border-radius:8px;
                    padding:12px;box-shadow:0 1px 2px rgba(0,0,0,.05)">
          <div style="display:flex;gap:8px;align-items:center;flex-wrap:wrap;
                      margin-bottom:6px">
            <b>#{i} <code>{_html.escape(b.box_id[:8])}</code></b>
            <span style="font-size:12px;color:#666">
              中心({b.center[0]:.2f}, {b.center[1]:.2f}, {b.center[2]:.2f}) ·
              尺寸({b.size[0]:.2f}×{b.size[1]:.2f}×{b.size[2]:.2f}) ·
              yaw {yaw_deg:.1f}° ·
              {_html.escape(b.device_type.value)} /
              {_html.escape(b.confidence.value)}
            </span>
          </div>
          {view_html}
          <div style="margin-top:6px">{v_html}</div>
        </div>""")
    parts.append("</div>")

    # orphan verdicts (deleted boxes)
    if orphans:
        parts.append("<h3>已删除 / 不在最终结果中的候选（判定记录）</h3>")
        parts.append('<div style="display:grid;grid-template-columns:'
                      'repeat(auto-fill,minmax(560px,1fr));gap:14px">')
        for j, rec in enumerate(orphans):
            parts.append(_verdict_block(rec, run_dir, j))
        parts.append("</div>")

    parts.append("</div></body></html>")
    html_doc = "".join(parts)
    with open(out_path, "w", encoding="utf-8") as f:
        f.write(html_doc)
    return out_path
