"""Evaluation: edge-accuracy against ground truth.

Acceptance metric per user spec:
  A predicted box edge is CORRECT if its distance to the matched ground-truth
  edge is below a threshold (default 5 cm). We report:

    - edge_accuracy: fraction of predicted edges within threshold
    - detection recall / precision (box-level, via 2D IoU matching)
    - mean / p90 edge error over matched boxes
"""
from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

from agentic_gts.core.models import OrientedBox


@dataclass
class EvalResult:
    edge_threshold_m: float
    n_gt: int = 0
    n_pred: int = 0
    n_matched: int = 0
    edge_accuracy: float = 0.0
    mean_edge_error_m: float = 0.0
    p90_edge_error_m: float = 0.0
    recall: float = 0.0
    precision: float = 0.0
    per_box: list[dict] = field(default_factory=list)

    def to_dict(self) -> dict:
        return {
            "edge_threshold_m": self.edge_threshold_m,
            "n_gt": self.n_gt, "n_pred": self.n_pred, "n_matched": self.n_matched,
            "edge_accuracy": round(self.edge_accuracy, 4),
            "mean_edge_error_m": round(self.mean_edge_error_m, 4),
            "p90_edge_error_m": round(self.p90_edge_error_m, 4),
            "recall": round(self.recall, 4),
            "precision": round(self.precision, 4),
        }

    def summary(self) -> str:
        return (f"edge_acc={self.edge_accuracy:.1%} "
                f"(thr={self.edge_threshold_m*100:.0f}cm)  "
                f"mean_err={self.mean_edge_error_m*100:.1f}cm  "
                f"p90_err={self.p90_edge_error_m*100:.1f}cm  "
                f"recall={self.recall:.1%}  precision={self.precision:.1%}  "
                f"({self.n_matched}/{self.n_gt} matched, {self.n_pred} pred)")


def _edge_error(pred: OrientedBox, gt: OrientedBox) -> list[float]:
    """Per-edge distance between matched boxes.

    For each of the 4 footprint edges of the GT box, find the nearest parallel
    edge of the predicted box and measure perpendicular offset. Simplified via
    the local frame of the GT box: compare the +-x and +-y face positions.
    """
    # express pred corners in gt local frame
    corners = pred.corners_2d()
    gtc = np.asarray(gt.center[:2])
    r = gt.rotation[:2, :2]
    local = (corners - gtc) @ r
    gl, gw = gt.size[0] / 2, gt.size[1] / 2
    px_min, px_max = local[:, 0].min(), local[:, 0].max()
    py_min, py_max = local[:, 1].min(), local[:, 1].max()
    return [abs(px_min - (-gl)), abs(px_max - gl),
            abs(py_min - (-gw)), abs(py_max - gw)]


def evaluate(pred_boxes: list[OrientedBox], gt_boxes: list[OrientedBox],
             edge_threshold_m: float = 0.05, match_iou: float = 0.3) -> EvalResult:
    res = EvalResult(edge_threshold_m=edge_threshold_m,
                     n_gt=len(gt_boxes), n_pred=len(pred_boxes))
    if not gt_boxes or not pred_boxes:
        return res
    # greedy IoU matching
    iou = np.zeros((len(gt_boxes), len(pred_boxes)))
    for i, g in enumerate(gt_boxes):
        for j, p in enumerate(pred_boxes):
            if abs(g.center[0] - p.center[0]) > 3 or abs(g.center[1] - p.center[1]) > 3:
                continue
            iou[i, j] = g.iou_2d(p)
    matched_gt: set[int] = set()
    matched_pred: set[int] = set()
    pairs: list[tuple[int, int]] = []
    flat = [(iou[i, j], i, j) for i in range(iou.shape[0]) for j in range(iou.shape[1])
            if iou[i, j] >= match_iou]
    for s, i, j in sorted(flat, reverse=True):
        if i in matched_gt or j in matched_pred:
            continue
        matched_gt.add(i)
        matched_pred.add(j)
        pairs.append((i, j))

    edge_errors: list[float] = []
    n_edges_ok = 0
    n_edges = 0
    for i, j in pairs:
        errs = _edge_error(pred_boxes[j], gt_boxes[i])
        edge_errors.extend(errs)
        n_edges += len(errs)
        n_edges_ok += sum(1 for e in errs if e <= edge_threshold_m)
        res.per_box.append({
            "gt_id": gt_boxes[i].box_id, "pred_id": pred_boxes[j].box_id,
            "iou": round(float(iou[i, j]), 3),
            "edge_errors_cm": [round(e * 100, 1) for e in errs],
        })

    res.n_matched = len(pairs)
    res.recall = len(pairs) / len(gt_boxes)
    res.precision = len(pairs) / len(pred_boxes) if pred_boxes else 0.0
    if edge_errors:
        res.edge_accuracy = n_edges_ok / n_edges
        res.mean_edge_error_m = float(np.mean(edge_errors))
        res.p90_edge_error_m = float(np.percentile(edge_errors, 90))
    return res
