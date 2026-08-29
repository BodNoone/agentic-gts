"""2D layout rendering: SVG (vector, machine-friendly) and PNG (preview)."""
from __future__ import annotations

import io
import math

import numpy as np

from agentic_gts.core.models import OrientedBox


def _bounds(boxes: list[OrientedBox], margin: float = 0.4):
    cs = np.vstack([b.corners_2d() for b in boxes]) if boxes else np.zeros((1, 2))
    lo = cs.min(axis=0) - margin
    hi = cs.max(axis=0) + margin
    return lo, hi


def boxes_to_svg(boxes: list[OrientedBox], title: str = "layout",
                 margin: float = 0.4, scale: float = 220) -> str:
    lo, hi = _bounds(boxes, margin)
    w, h = hi - lo
    sw = max(int(w * scale), 50)
    sh = max(int(h * scale), 50)
    pad = 20

    def world_to_px(x: float, y: float) -> tuple[float, float]:
        px = pad + (x - lo[0]) * scale
        py = pad + (hi[1] - y) * scale
        return px, py

    parts = [f'<svg xmlns="http://www.w3.org/2000/svg" '
             f'width="{sw + 2*pad}" height="{sh + 2*pad}" viewBox="0 0 {sw + 2*pad} {sh + 2*pad}">',
             f'<rect x="0" y="0" width="{sw + 2*pad}" height="{sh + 2*pad}" fill="white"/>',
             f'<text x="{pad}" y="{pad-6}" font-size="12" font-family="sans-serif">{title}</text>']
    for b in boxes:
        cs = b.corners_2d()
        pts = [world_to_px(x, y) for x, y in cs]
        points_str = " ".join(f"{px:.1f},{py:.1f}" for px, py in pts)
        conf_color = {"high": "#1a7f37", "mid": "#b8860b", "low": "#c0392b"}
        color = conf_color.get(b.confidence.value, "#555")
        label = f"{b.device_type.value}#{b.box_id[:4]}"
        lx, ly = world_to_px(b.center[0], b.center[1])
        parts.append(f'<polygon points="{points_str}" fill="{color}" fill-opacity="0.15" '
                     f'stroke="{color}" stroke-width="1.5"/>')
        parts.append(f'<text x="{lx:.1f}" y="{ly:.1f}" font-size="9" text-anchor="middle" '
                     f'dominant-baseline="middle" fill="#333">{label}</text>')
    parts.append("</svg>")
    return "\n".join(parts)


def boxes_to_png(boxes: list[OrientedBox], title: str = "layout",
                 margin: float = 0.4, scale: float = 220) -> bytes:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.patches import Polygon

    lo, hi = _bounds(boxes, margin)
    fig, ax = plt.subplots(figsize=(8, 6))
    for b in boxes:
        cs = b.corners_2d()
        conf_color = {"high": "#1a7f37", "mid": "#b8860b", "low": "#c0392b"}
        color = conf_color.get(b.confidence.value, "#555")
        ax.add_patch(Polygon(cs, closed=True, fill=True, alpha=0.15,
                             edgecolor=color, linewidth=1.5))
        ax.text(b.center[0], b.center[1], b.device_type.value[:6], fontsize=7,
                ha="center", va="center")
    ax.set_aspect("equal")
    ax.set_xlim(lo[0], hi[0])
    ax.set_ylim(lo[1], hi[1])
    ax.set_title(title)
    ax.set_xticks([]); ax.set_yticks([])
    buf = io.BytesIO()
    fig.savefig(buf, format="png", bbox_inches="tight")
    plt.close(fig)
    buf.seek(0)
    return buf.read()
