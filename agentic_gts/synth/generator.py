"""Synthetic machine-room generator.

Generates:
  1. Ground-truth layout (rows of racks + optional AC units)
  2. A point cloud sampled from device surfaces + floor + walls,
     with configurable noise / density variation / reflective dropout
  3. A corrupted "initial detection" box set that simulates the four
     real-world failure modes:
       - oversized boxes
       - missing detections
       - false positives
       - merged (row-adjacent racks fused into one box)

This gives the pipeline a fully self-contained test bed with
ground truth for edge-accuracy evaluation.
"""
from __future__ import annotations

import math
import random
from dataclasses import dataclass, field

import numpy as np

from agentic_gts.core.models import (
    BoxSource,
    Confidence,
    DeviceType,
    OrientedBox,
    Scene,
)


@dataclass
class SynthConfig:
    seed: int = 42
    n_rows: int = 3
    racks_per_row: tuple[int, int] = (6, 10)      # min, max
    rack_width: float = 0.6                        # along row
    rack_depth: float = 1.1
    rack_height: float = 2.0
    rack_gap: float = 0.005                        # tiny gap between adjacent racks
    row_spacing: float = 2.4                       # aisle pitch between row center lines
    room_margin: float = 1.5                       # free space around layout
    room_yaw_deg: float = 0.0                      # global rotation of the room layout
    points_per_m2: float = 900.0                   # surface sampling density
    noise_sigma: float = 0.008                     # gaussian noise on points (m)
    density_jitter: tuple[float, float] = (0.5, 1.5)   # per-device density multiplier range
    reflective_dropout_prob: float = 0.15          # prob a rack gets a face with big dropout
    reflective_dropout_frac: float = 0.6           # fraction of that face's points removed
    include_floor: bool = True
    include_walls: bool = True
    n_ac_units: int = 1
    # corruption of initial detections
    p_oversize: float = 0.2
    oversize_amount: tuple[float, float] = (0.05, 0.25)
    p_missing: float = 0.1
    p_false_positive_per_row: float = 0.4
    p_merge_pair: float = 0.15                     # prob to fuse a pair of adjacent racks
    center_jitter: float = 0.03


def _sample_box_surface(box: OrientedBox, pts_per_m2: float, rng: np.random.Generator,
                        top: bool = True) -> np.ndarray:
    """Sample points on the 4 side faces (+optional top) of an oriented box.

    Local coordinates use the box's CENTERED convention (local_to_world adds
    the box center), so faces live at +/- size/2 in their normal direction.
    """
    L, W, H = box.size
    faces = []
    # each face: (u_len, v_len, generator of local coords)
    specs = [
        ("front", L, H), ("back", L, H),
        ("left", W, H), ("right", W, H),
    ]
    if top:
        specs.append(("top", L, W))
    for name, ul, vl in specs:
        n = max(4, int(ul * vl * pts_per_m2))
        u = rng.uniform(-ul / 2, ul / 2, n)
        v = rng.uniform(-vl / 2, vl / 2, n)
        if name == "front":
            local = np.stack([u, np.full(n, -W / 2), v], axis=1)
        elif name == "back":
            local = np.stack([u, np.full(n, W / 2), v], axis=1)
        elif name == "left":
            local = np.stack([np.full(n, -L / 2), u, v], axis=1)
        elif name == "right":
            local = np.stack([np.full(n, L / 2), u, v], axis=1)
        else:  # top
            local = np.stack([u, v, np.full(n, H / 2)], axis=1)
        faces.append((name, local))
    return faces


def generate(config: SynthConfig | None = None) -> tuple[Scene, list[OrientedBox], list[OrientedBox]]:
    """Build a synthetic scene.

    Returns (scene, gt_boxes, corrupted_boxes).
    The scene's `boxes` field is set to the corrupted set (pipeline input).
    """
    cfg = config or SynthConfig()
    rng = np.random.default_rng(cfg.seed)
    pyrng = random.Random(cfg.seed)

    yaw = math.radians(cfg.room_yaw_deg)
    cos_y, sin_y = math.cos(yaw), math.sin(yaw)

    def room_xform(x: float, y: float) -> tuple[float, float]:
        return (x * cos_y - y * sin_y, x * sin_y + y * cos_y)

    gt_boxes: list[OrientedBox] = []
    all_pts: list[np.ndarray] = []

    # ---- build rows of racks ----
    pitch = cfg.rack_width + cfg.rack_gap
    for row in range(cfg.n_rows):
        n_racks = pyrng.randint(*cfg.racks_per_row)
        row_y = row * cfg.row_spacing
        x0 = 0.0
        for i in range(n_racks):
            cx = x0 + i * pitch + cfg.rack_width / 2
            wx, wy = room_xform(cx, row_y)
            box = OrientedBox(
                center=(wx, wy, cfg.rack_height / 2),
                size=(cfg.rack_width, cfg.rack_depth, cfg.rack_height),
                yaw=yaw,
                device_type=DeviceType.RACK,
                source=BoxSource.MANUAL,
                row_id=row,
            )
            gt_boxes.append(box)

    # ---- optional AC units at row ends ----
    for k in range(cfg.n_ac_units):
        row = k % cfg.n_rows
        n_in_row = sum(1 for b in gt_boxes if b.row_id == row)
        cx = n_in_row * pitch + 0.9
        wx, wy = room_xform(cx, row * cfg.row_spacing)
        gt_boxes.append(OrientedBox(
            center=(wx, wy, 1.0),
            size=(1.0, 0.9, 2.0),
            yaw=yaw,
            device_type=DeviceType.AC,
            source=BoxSource.MANUAL,
            row_id=row,
        ))

    # ---- sample device surfaces ----
    for box in gt_boxes:
        density = cfg.points_per_m2 * rng.uniform(*cfg.density_jitter)
        faces = _sample_box_surface(box, density, rng)
        drop_face = None
        if rng.random() < cfg.reflective_dropout_prob:
            drop_face = pyrng.choice(["front", "back", "left", "right"])
        for name, local in faces:
            if name == drop_face:
                keep = rng.random(len(local)) > cfg.reflective_dropout_frac
                local = local[keep]
            world = box.local_to_world(local)
            all_pts.append(world)

    # ---- extents ----
    dev_pts = np.vstack(all_pts)
    lo = dev_pts.min(axis=0) - cfg.room_margin
    hi = dev_pts.max(axis=0) + cfg.room_margin
    lo[2] = 0.0

    # ---- floor ----
    if cfg.include_floor:
        area = (hi[0] - lo[0]) * (hi[1] - lo[1])
        n = int(area * cfg.points_per_m2 * 0.25)
        fx = rng.uniform(lo[0], hi[0], n)
        fy = rng.uniform(lo[1], hi[1], n)
        all_pts.append(np.stack([fx, fy, np.zeros(n)], axis=1))

    # ---- walls (sparse) ----
    if cfg.include_walls:
        wall_h = 3.0
        for (x_fixed, along_y) in [(lo[0], True), (hi[0], True)]:
            span = hi[1] - lo[1]
            n = int(span * wall_h * cfg.points_per_m2 * 0.08)
            wy = rng.uniform(lo[1], hi[1], n)
            wz = rng.uniform(0, wall_h, n)
            all_pts.append(np.stack([np.full(n, x_fixed), wy, wz], axis=1))
        for y_fixed in [lo[1], hi[1]]:
            span = hi[0] - lo[0]
            n = int(span * wall_h * cfg.points_per_m2 * 0.08)
            wx = rng.uniform(lo[0], hi[0], n)
            wz = rng.uniform(0, wall_h, n)
            all_pts.append(np.stack([wx, np.full(n, y_fixed), wz], axis=1))

    points = np.vstack(all_pts)
    points += rng.normal(0, cfg.noise_sigma, points.shape)

    # ---- corrupt detections ----
    corrupted: list[OrientedBox] = []
    racks = [b for b in gt_boxes if b.device_type == DeviceType.RACK]
    others = [b for b in gt_boxes if b.device_type != DeviceType.RACK]

    merged_ids: set[str] = set()
    by_row: dict[int, list[OrientedBox]] = {}
    for b in racks:
        by_row.setdefault(b.row_id, []).append(b)
    for row, rbs in by_row.items():
        rbs_sorted = sorted(rbs, key=lambda b: (np.asarray(b.center[:2]) @ np.array([cos_y, sin_y])))
        i = 0
        while i < len(rbs_sorted) - 1:
            if pyrng.random() < cfg.p_merge_pair and rbs_sorted[i].box_id not in merged_ids:
                a, b2 = rbs_sorted[i], rbs_sorted[i + 1]
                merged_ids.update([a.box_id, b2.box_id])
                ctr = (np.asarray(a.center) + np.asarray(b2.center)) / 2
                corrupted.append(OrientedBox(
                    center=tuple(ctr),
                    size=(a.size[0] + b2.size[0] + cfg.rack_gap, a.size[1], a.size[2]),
                    yaw=a.yaw, device_type=DeviceType.RACK,
                    source=BoxSource.COARSE_SEG, confidence=Confidence.MID,
                    row_id=row, meta={"corruption": "merged"},
                ))
                i += 2
            else:
                i += 1

    for b in racks:
        if b.box_id in merged_ids:
            continue
        r = pyrng.random()
        if r < cfg.p_missing:
            continue  # dropped -> missing detection
        size = list(b.size)
        center = list(b.center)
        meta = {}
        if pyrng.random() < cfg.p_oversize:
            grow = rng.uniform(*cfg.oversize_amount)
            axis = pyrng.choice([0, 1])
            size[axis] += grow
            meta["corruption"] = "oversized"
        center[0] += rng.normal(0, cfg.center_jitter)
        center[1] += rng.normal(0, cfg.center_jitter)
        corrupted.append(OrientedBox(
            center=tuple(center), size=tuple(size), yaw=b.yaw,
            device_type=DeviceType.RACK, source=BoxSource.COARSE_SEG,
            confidence=Confidence.MID, row_id=b.row_id, meta=meta,
        ))

    for b in others:
        corrupted.append(OrientedBox(
            center=b.center, size=b.size, yaw=b.yaw,
            device_type=b.device_type, source=BoxSource.COARSE_SEG,
            confidence=Confidence.MID, row_id=b.row_id,
        ))

    # false positives in the aisles
    for row in range(cfg.n_rows - 1):
        if pyrng.random() < cfg.p_false_positive_per_row:
            fx = rng.uniform(0.5, 3.0)
            fy = row * cfg.row_spacing + cfg.row_spacing / 2
            wx, wy = room_xform(fx, fy)
            corrupted.append(OrientedBox(
                center=(wx, wy, 0.9),
                size=(0.55, 0.7, 1.8), yaw=yaw,
                device_type=DeviceType.RACK, source=BoxSource.COARSE_SEG,
                confidence=Confidence.LOW, meta={"corruption": "false_positive"},
            ))

    scene = Scene(points=points, boxes=corrupted, floor_z=0.0,
                  meta={"synthetic": True, "yaw": yaw})
    return scene, gt_boxes, corrupted
