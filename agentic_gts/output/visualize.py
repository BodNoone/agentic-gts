"""Visualization: point cloud + detected boxes together (2D overlay & 3D).

Three outputs:
  1. overlay_topdown  -- 2D top-down scatter of the point cloud with box
                         footprints drawn on top (PNG). Fast visual QA.
  2. view_3d          -- interactive Open3D window with the cloud and 3D
                         wireframe boxes (color-coded by confidence).
  3. export_ply       -- merged PLY: cloud + dense wireframe points so any
                         external viewer (CloudCompare/MeshLab) shows both.
"""
from __future__ import annotations

import io

import numpy as np

from agentic_gts.core.models import OrientedBox, Scene

_CONF_COLOR = {  # RGB in [0,1]
    "high": (0.10, 0.50, 0.22),
    "mid": (0.72, 0.53, 0.04),
    "low": (0.75, 0.22, 0.17),
}


# ------------------------------------------------------- yaw diagnosis render
def render_yaw_diagnosis(device_pts: np.ndarray, candidates: list,
                         chosen_yaw: float, path: str,
                         max_points: int = 80_000) -> None:
    """Save a top-down PNG of device-band points with candidate yaw arrows.

    Gray scatter = points in the device height band (after denoise/align).
    Thin blue arrows = scored candidate directions (length ~ alpha by score);
    the thick red double arrow = the chosen yaw. If the red arrow does not
    follow the device rows in the scatter, yaw estimation is being hijacked
    by another structure -- the PNG shows which one.
    """
    import math as _math
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    pts = device_pts
    if len(pts) > max_points:
        sel = np.random.default_rng(0).choice(len(pts), max_points, replace=False)
        pts = pts[sel]
    fig, ax = plt.subplots(figsize=(10, 10))
    ax.scatter(pts[:, 0], pts[:, 1], s=0.4, c="#999", alpha=0.25, linewidths=0)
    ax.set_aspect("equal")

    if len(pts):
        c0 = pts.mean(axis=0)
        span = float(np.ptp(pts, axis=0).max())
        L = 0.35 * span
        scores = [s for _, s in candidates] or [1.0]
        smax = max(scores)
        for deg, score in candidates:
            a = _math.radians(float(deg))
            d = np.array([_math.cos(a), _math.sin(a)])
            ax.annotate("", xy=c0 + L * d, xytext=c0 - L * d,
                        arrowprops=dict(arrowstyle="<->", color="#4a90d9",
                                        alpha=0.35 + 0.6 * score / smax, lw=1.2))
        d = np.array([_math.cos(chosen_yaw), _math.sin(chosen_yaw)])
        ax.annotate("", xy=c0 + L * d, xytext=c0 - L * d,
                    arrowprops=dict(arrowstyle="<->", color="#d93025", lw=3))
        ax.text(c0[0], c0[1] + 0.05 * span,
                f"chosen yaw = {_math.degrees(chosen_yaw):.1f} deg",
                color="#d93025", fontsize=13, ha="center", weight="bold")
    cand_txt = "  ".join(f"{d:.1f}deg:{s:.0f}" for d, s in candidates[:6])
    ax.set_title(f"yaw diagnosis | candidates: {cand_txt or 'none'}", fontsize=10)
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)


# ---------------------------------------------------------------- 2D overlay
def overlay_topdown(scene: Scene, gt_boxes: list[OrientedBox] | None = None,
                    max_points: int = 200_000, point_size: float = 0.4,
                    title: str = "points + boxes") -> bytes:
    """Top-down scatter of the cloud with box footprints overlaid (PNG bytes).

    Predicted boxes: solid, confidence-colored. GT boxes (optional): dashed
    blue -- so pred-vs-gt misalignment is visible at a glance.
    """
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.lines import Line2D
    from matplotlib.patches import Polygon

    pts = scene.points
    if len(pts) > max_points:
        sel = np.random.default_rng(0).choice(len(pts), max_points, replace=False)
        pts = pts[sel]

    fig, ax = plt.subplots(figsize=(11, 8), dpi=110)
    # height-colored scatter makes racks pop out from the floor
    z = pts[:, 2]
    sc = ax.scatter(pts[:, 0], pts[:, 1], c=z, s=point_size, cmap="viridis",
                    alpha=0.5, linewidths=0, rasterized=True)
    fig.colorbar(sc, ax=ax, label="height z (m)", shrink=0.75)

    for b in scene.boxes:
        color = _CONF_COLOR.get(b.confidence.value, (0.3, 0.3, 0.3))
        ax.add_patch(Polygon(b.corners_2d(), closed=True, fill=False,
                             edgecolor=color, linewidth=1.8))
        ax.text(b.center[0], b.center[1], b.box_id[:4], fontsize=6,
                ha="center", va="center", color=color)
    if gt_boxes:
        for g in gt_boxes:
            ax.add_patch(Polygon(g.corners_2d(), closed=True, fill=False,
                                 edgecolor="royalblue", linewidth=1.2,
                                 linestyle="--"))

    handles = [Line2D([0], [0], color=c, lw=2, label=f"pred ({k})")
               for k, c in _CONF_COLOR.items()]
    if gt_boxes:
        handles.append(Line2D([0], [0], color="royalblue", lw=2,
                              linestyle="--", label="ground truth"))
    ax.legend(handles=handles, loc="upper right", fontsize=8)
    ax.set_aspect("equal")
    ax.set_title(title)
    ax.set_xlabel("x (m)"); ax.set_ylabel("y (m)")
    buf = io.BytesIO()
    fig.savefig(buf, format="png", bbox_inches="tight")
    plt.close(fig)
    buf.seek(0)
    return buf.read()


# ---------------------------------------------------------------- 3D helpers
def _box_lineset(box: OrientedBox):
    """Open3D LineSet for a 3D wireframe of the oriented box."""
    import open3d as o3d
    l, w, h = np.asarray(box.size) / 2.0
    corners_local = np.array([
        [-l, -w, -h], [l, -w, -h], [l, w, -h], [-l, w, -h],
        [-l, -w, h], [l, -w, h], [l, w, h], [-l, w, h],
    ])
    corners = box.local_to_world(corners_local)
    lines = [[0, 1], [1, 2], [2, 3], [3, 0],
             [4, 5], [5, 6], [6, 7], [7, 4],
             [0, 4], [1, 5], [2, 6], [3, 7]]
    color = _CONF_COLOR.get(box.confidence.value, (0.3, 0.3, 0.3))
    ls = o3d.geometry.LineSet(
        points=o3d.utility.Vector3dVector(corners),
        lines=o3d.utility.Vector2iVector(lines))
    ls.colors = o3d.utility.Vector3dVector([color] * len(lines))
    return ls


def view_3d(scene: Scene, gt_boxes: list[OrientedBox] | None = None,
            max_points: int = 400_000) -> None:
    """Open an interactive Open3D window: cloud + wireframe boxes."""
    import open3d as o3d
    pts = scene.points
    if len(pts) > max_points:
        sel = np.random.default_rng(0).choice(len(pts), max_points, replace=False)
        pts = pts[sel]
    pcd = o3d.geometry.PointCloud(o3d.utility.Vector3dVector(pts))
    # height-based gray-to-blue coloring
    z = pts[:, 2]
    t = (z - z.min()) / max(float(np.ptp(z)), 1e-6)
    colors = np.stack([0.55 - 0.25 * t, 0.55 - 0.1 * t, 0.55 + 0.35 * t], axis=1)
    pcd.colors = o3d.utility.Vector3dVector(np.clip(colors, 0, 1))

    geoms = [pcd] + [_box_lineset(b) for b in scene.boxes]
    if gt_boxes:
        import copy as _copy
        for g in gt_boxes:
            g2 = _copy.deepcopy(g)
            ls = _box_lineset(g2)
            ls.paint_uniform_color((0.25, 0.41, 0.88))
            geoms.append(ls)
    o3d.visualization.draw_geometries(
        geoms, window_name="agentic-gts: points + boxes",
        width=1440, height=900)


def _wireframe_points(box: OrientedBox, step: float = 0.02) -> np.ndarray:
    """Densely sample the 12 wireframe edges as points (for PLY export)."""
    l, w, h = np.asarray(box.size) / 2.0
    c = np.array([
        [-l, -w, -h], [l, -w, -h], [l, w, -h], [-l, w, -h],
        [-l, -w, h], [l, -w, h], [l, w, h], [-l, w, h],
    ])
    edges = [(0, 1), (1, 2), (2, 3), (3, 0),
             (4, 5), (5, 6), (6, 7), (7, 4),
             (0, 4), (1, 5), (2, 6), (3, 7)]
    segs = []
    for i, j in edges:
        d = np.linalg.norm(c[j] - c[i])
        n = max(int(d / step), 2)
        t = np.linspace(0, 1, n)[:, None]
        segs.append(c[i] * (1 - t) + c[j] * t)
    return box.local_to_world(np.vstack(segs))


def export_ply(scene: Scene, path: str,
               gt_boxes: list[OrientedBox] | None = None,
               max_points: int = 1_000_000) -> None:
    """Write a merged PLY: cloud (gray, height-tinted) + box wireframes.

    Openable in CloudCompare / MeshLab / any PLY viewer for 3D inspection
    without needing this codebase.
    """
    import open3d as o3d
    pts = scene.points
    if len(pts) > max_points:
        sel = np.random.default_rng(0).choice(len(pts), max_points, replace=False)
        pts = pts[sel]
    z = pts[:, 2]
    t = (z - z.min()) / max(float(np.ptp(z)), 1e-6)
    cloud_colors = np.stack([0.6 - 0.2 * t, 0.6 - 0.05 * t, 0.6 + 0.3 * t], axis=1)

    all_pts = [pts]
    all_col = [np.clip(cloud_colors, 0, 1)]
    for b in scene.boxes:
        wp = _wireframe_points(b)
        color = np.asarray(_CONF_COLOR.get(b.confidence.value, (0.3, 0.3, 0.3)))
        all_pts.append(wp)
        all_col.append(np.tile(color, (len(wp), 1)))
    if gt_boxes:
        for g in gt_boxes:
            wp = _wireframe_points(g)
            all_pts.append(wp)
            all_col.append(np.tile((0.25, 0.41, 0.88), (len(wp), 1)))

    merged = o3d.geometry.PointCloud(
        o3d.utility.Vector3dVector(np.vstack(all_pts)))
    merged.colors = o3d.utility.Vector3dVector(np.vstack(all_col))
    o3d.io.write_point_cloud(path, merged)
