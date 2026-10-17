"""Local row directions must support fan layouts without VLM/GPU calls."""
import math

import numpy as np

from agentic_gts.core.models import OrientedBox
from agentic_gts.agent.ground import _box_frame_rect, _refine_layout_directions


def _surface(center, angle):
    x = np.linspace(-1.5, 1.5, 121)
    y = np.linspace(-.4, .4, 33)
    xy = np.vstack([np.column_stack([x, np.full(len(x), side)])
                    for side in (-.4, .4)] +
                   [np.column_stack([np.full(len(y), end), y])
                    for end in (-1.5, 1.5)])
    c, s = math.cos(angle), math.sin(angle)
    xy = xy @ np.array([[c, s], [-s, c]]) + center
    return np.column_stack([xy, np.full(len(xy), .65)])


def test_arbitrary_angle_coverage_uses_all_corners():
    box = OrientedBox(center=(2, 3, 1), size=(3, .8, 2), yaw=.2)
    rect = _box_frame_rect(box)
    corners = box.corners_2d()
    assert np.allclose(rect[:2], corners.min(axis=0))
    assert np.allclose(rect[2:], corners.max(axis=0))
    assert rect[2] - rect[0] > 2.9


def test_fan_rows_recover_smooth_local_directions():
    boxes, clouds, truth = [], [], []
    frame = math.pi / 2
    for x in np.arange(-8., 9., 2.):
        yaw = frame + math.radians(1.2 * x)
        center = np.array([x, 0.])
        clouds.append(_surface(center, yaw))
        boxes.append(OrientedBox(center=(*center, 1), size=(3.2, 1.3, 2),
                                 yaw=frame))
        truth.append(yaw)
    _refine_layout_directions(boxes, np.vstack(clouds), frame, mesh_mode=True)
    for box, expected in zip(boxes, truth):
        error = math.remainder(box.yaw - expected, math.pi / 2)
        assert abs(math.degrees(error)) < 2, box.meta
        assert box.size[1] < 1.1


def test_straight_row_corrects_bad_seed_without_rotating_neighbors():
    points = _surface(np.array([0., 0.]), math.pi / 2)
    box = OrientedBox(center=(0, 0, 1), size=(3.5, 1.8, 2),
                      yaw=math.pi / 2 + .35)
    _refine_layout_directions([box], points, math.pi / 2, mesh_mode=True)
    assert abs(math.remainder(box.yaw - math.pi / 2, math.pi)) < .02
    assert box.size[1] < .9


def test_independent_slanted_structure_keeps_strong_local_evidence():
    boxes, clouds = [], []
    for x in (-4., -2., 0., 2., 4.):
        angle = math.pi / 2 + (math.radians(20) if x == 0 else 0.)
        clouds.append(_surface(np.array([x, 0.]), angle))
        boxes.append(OrientedBox(center=(x, 0, 1), size=(3.4, 1.8, 2),
                                 yaw=math.pi / 2))
    _refine_layout_directions(boxes, np.vstack(clouds), math.pi / 2,
                              mesh_mode=True)
    assert abs(math.degrees(boxes[2].yaw - math.pi / 2) - 20) < 2


def test_local_refine_normalizes_pca_short_axis_to_long_box_axis():
    # The cloud is long along local Y while the seed's declared long axis is
    # X. The corrected OBB must swap size axes and rotate by 90 degrees once.
    points = _surface(np.array([0., 0.]), math.pi / 2)
    box = OrientedBox(center=(0, 0, 1), size=(.8, 3.2, 2), yaw=0.0)
    _refine_layout_directions([box], points, 0.0, mesh_mode=True)
    assert box.size[0] > box.size[1]
    assert abs(math.remainder(box.yaw - math.pi / 2, math.pi)) < .02


def test_local_refine_uses_seed_obb_and_centered_fit_not_neighbor_aabb():
    own = _surface(np.array([10., 10.]), math.pi / 2)
    neighbor = _surface(np.array([10., 11.3]), math.pi / 2)
    box = OrientedBox(center=(10, 10, 1), size=(3.2, .9, 2),
                      yaw=math.pi / 2)
    _refine_layout_directions([box], np.vstack([own, neighbor]),
                              math.pi / 2, mesh_mode=True)
    assert abs(box.center[0] - 10) < .05
    assert abs(box.center[1] - 10) < .20
    assert box.size[1] < 1.1
    assert box.meta["orientation_refine"]["pool_points"] < len(own) + len(neighbor)
