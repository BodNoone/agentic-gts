"""Core data models for the agentic-gts pipeline.

All geometry uses a right-handed coordinate system:
  x, y: floor plane; z: up.
Units: meters.

BBox convention: axis-aligned in a *row-local* frame defined by `yaw`
(rotation around z). `center` is the box center, `size` = (L, W, H)
where L is along the row direction (local x after yaw rotation).
"""
from __future__ import annotations

import json
import math
import uuid
from dataclasses import dataclass, field, asdict
from enum import Enum
from typing import Optional

import numpy as np


class DeviceType(str, Enum):
    RACK = "rack"            # standard cabinet
    AC = "ac"                # CRAC unit / air conditioner
    UPS = "ups"
    UNKNOWN = "unknown"


class BoxSource(str, Enum):
    COARSE_SEG = "coarse_seg"        # stage A output
    RULE_FIX = "rule_fix"            # stage B modified
    AGENT_FIX = "agent_fix"          # stage C modified
    ROW_COMPLETION = "row_completion"  # gap filling
    MANUAL = "manual"


class Confidence(str, Enum):
    HIGH = "high"
    MID = "mid"
    LOW = "low"


@dataclass
class OrientedBox:
    """Oriented 3D bounding box (rotation only around z)."""
    center: tuple[float, float, float]
    size: tuple[float, float, float]     # (length_x_local, width_y_local, height_z)
    yaw: float = 0.0                     # radians, rotation around z
    box_id: str = field(default_factory=lambda: uuid.uuid4().hex[:8])
    device_type: DeviceType = DeviceType.UNKNOWN
    source: BoxSource = BoxSource.COARSE_SEG
    confidence: Confidence = Confidence.MID
    score: float = 0.5                   # scalar confidence score in [0,1]
    row_id: Optional[int] = None
    meta: dict = field(default_factory=dict)

    # ---------- geometry helpers ----------
    @property
    def rotation(self) -> np.ndarray:
        c, s = math.cos(self.yaw), math.sin(self.yaw)
        return np.array([[c, -s, 0.0], [s, c, 0.0], [0.0, 0.0, 1.0]])

    def world_to_local(self, pts: np.ndarray) -> np.ndarray:
        """Transform Nx3 world points into the box-local frame."""
        return (pts - np.asarray(self.center)) @ self.rotation  # R^T applied via right-mult

    def local_to_world(self, pts: np.ndarray) -> np.ndarray:
        return pts @ self.rotation.T + np.asarray(self.center)

    def contains(self, pts: np.ndarray, margin: float = 0.0) -> np.ndarray:
        """Boolean mask of points inside the box (with optional margin)."""
        local = self.world_to_local(pts)
        half = np.asarray(self.size) / 2.0 + margin
        return np.all(np.abs(local) <= half, axis=1)

    def corners_2d(self) -> np.ndarray:
        """4x2 footprint corners in world frame (x, y)."""
        l, w = self.size[0] / 2.0, self.size[1] / 2.0
        local = np.array([[-l, -w], [l, -w], [l, w], [-l, w]])
        c, s = math.cos(self.yaw), math.sin(self.yaw)
        rot = np.array([[c, -s], [s, c]])
        return local @ rot.T + np.asarray(self.center[:2])

    def footprint_edges(self) -> list[tuple[np.ndarray, np.ndarray]]:
        """Four (start, end) 2D edges of the footprint."""
        cs = self.corners_2d()
        return [(cs[i], cs[(i + 1) % 4]) for i in range(4)]

    def iou_2d(self, other: "OrientedBox", grid: float = 0.02) -> float:
        """Approximate 2D IoU by rasterization (robust for small yaw diffs)."""
        all_pts = np.vstack([self.corners_2d(), other.corners_2d()])
        lo = all_pts.min(axis=0) - grid
        hi = all_pts.max(axis=0) + grid
        xs = np.arange(lo[0], hi[0], grid)
        ys = np.arange(lo[1], hi[1], grid)
        if len(xs) == 0 or len(ys) == 0 or len(xs) * len(ys) > 4_000_000:
            return 0.0
        gx, gy = np.meshgrid(xs, ys)
        pts = np.stack([gx.ravel(), gy.ravel(), np.zeros(gx.size)], axis=1)
        za = np.zeros(3)
        a = OrientedBox(center=(self.center[0], self.center[1], 0), size=(self.size[0], self.size[1], 10), yaw=self.yaw)
        b = OrientedBox(center=(other.center[0], other.center[1], 0), size=(other.size[0], other.size[1], 10), yaw=other.yaw)
        ma = a.contains(pts)
        mb = b.contains(pts)
        inter = np.count_nonzero(ma & mb)
        union = np.count_nonzero(ma | mb)
        return float(inter) / union if union else 0.0

    def to_dict(self) -> dict:
        d = asdict(self)
        d["device_type"] = self.device_type.value
        d["source"] = self.source.value
        d["confidence"] = self.confidence.value
        return d

    @classmethod
    def from_dict(cls, d: dict) -> "OrientedBox":
        d = dict(d)
        d["device_type"] = DeviceType(d.get("device_type", "unknown"))
        d["source"] = BoxSource(d.get("source", "coarse_seg"))
        d["confidence"] = Confidence(d.get("confidence", "mid"))
        d["center"] = tuple(d["center"])
        d["size"] = tuple(d["size"])
        return cls(**d)


class IssueType(str, Enum):
    OVERSIZED = "oversized"          # box larger than point support
    UNDERSIZED = "undersized"
    MISSING = "missing"              # gap in row with point density but no box
    FALSE_POSITIVE = "false_positive"
    MERGED_ROW = "merged_row"        # multiple racks in one box
    OVERLAP = "overlap"              # two boxes overlapping
    MISALIGNED = "misaligned"        # not aligned with row direction
    LOW_SUPPORT = "low_support"


@dataclass
class Issue:
    issue_type: IssueType
    box_ids: list[str]
    region: tuple[float, float, float, float]   # xmin, ymin, xmax, ymax
    detail: str = ""
    severity: float = 0.5
    issue_id: str = field(default_factory=lambda: uuid.uuid4().hex[:8])

    def to_dict(self) -> dict:
        d = asdict(self)
        d["issue_type"] = self.issue_type.value
        return d


@dataclass
class Scene:
    """A scene = point cloud + working set of boxes + metadata."""
    points: np.ndarray                       # Nx3 float
    colors: Optional[np.ndarray] = None      # Nx3 float in [0,1]
    boxes: list[OrientedBox] = field(default_factory=list)
    floor_z: float = 0.0
    meta: dict = field(default_factory=dict)

    def get_box(self, box_id: str) -> Optional[OrientedBox]:
        for b in self.boxes:
            if b.box_id == box_id:
                return b
        return None

    def remove_box(self, box_id: str) -> bool:
        n = len(self.boxes)
        self.boxes = [b for b in self.boxes if b.box_id != box_id]
        return len(self.boxes) < n

    def points_in_region(self, region: tuple[float, float, float, float],
                         z_range: Optional[tuple[float, float]] = None) -> np.ndarray:
        xmin, ymin, xmax, ymax = region
        m = ((self.points[:, 0] >= xmin) & (self.points[:, 0] <= xmax)
             & (self.points[:, 1] >= ymin) & (self.points[:, 1] <= ymax))
        if z_range is not None:
            m &= (self.points[:, 2] >= z_range[0]) & (self.points[:, 2] <= z_range[1])
        return self.points[m]

    def save_boxes(self, path: str) -> None:
        with open(path, "w", encoding="utf-8") as f:
            json.dump([b.to_dict() for b in self.boxes], f, ensure_ascii=False, indent=2)

    def load_boxes(self, path: str) -> None:
        with open(path, "r", encoding="utf-8") as f:
            self.boxes = [OrientedBox.from_dict(d) for d in json.load(f)]


# Standard rack dimensions (meters). Optional priors -- the pipeline
# must work even when a machine room does not match these.
STANDARD_RACK_WIDTHS = [0.6, 0.8]
STANDARD_RACK_DEPTHS = [1.0, 1.1, 1.2]
STANDARD_RACK_HEIGHT_RANGE = (1.8, 2.3)
