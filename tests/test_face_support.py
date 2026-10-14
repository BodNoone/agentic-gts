import json
import math

import numpy as np
import pytest

from agentic_gts.core.models import Confidence, OrientedBox, Scene
from agentic_gts.tools.geometry import (filter_outer_wall_boxes,
                                        flag_insufficient_face_support)


def surface(box, axis, sign):
    half = np.asarray(box.size) / 2
    tangent = [i for i in range(3) if i != axis]
    coordinates = [np.arange(-half[i] + 0.05, half[i], 0.10)
                   for i in tangent]
    u, v = np.meshgrid(*coordinates)
    points = np.zeros((u.size, 3))
    points[:, axis] = sign * half[axis]
    points[:, tangent[0]] = u.ravel()
    points[:, tangent[1]] = v.ravel()
    return box.local_to_world(points)


@pytest.mark.parametrize("yaw", [0, math.pi / 2, 0.37])
@pytest.mark.parametrize("face_count", [2, 3])
def test_three_supported_faces_required_without_deletion(yaw, face_count):
    box = OrientedBox(center=(3, -2, 1), size=(1, 1, 2), yaw=yaw,
                      confidence=Confidence.HIGH)
    specifications = [(1, 1), (1, -1), (2, 1)]
    scene = Scene(points=np.vstack([surface(box, *s)
                                    for s in specifications[:face_count]]),
                  boxes=[box])
    flagged = flag_insufficient_face_support(scene)
    assert scene.boxes == [box]
    assert flagged == (face_count < 3)
    assert box.meta["face_support"]["matched_faces"] == face_count
    assert box.meta["face_support"]["min_coverage"] == 0.90
    assert box.confidence == (Confidence.MID if face_count < 3
                              else Confidence.HIGH)
    json.dumps(box.to_dict())


def test_dense_fragment_and_floor_do_not_qualify_faces():
    box = OrientedBox(center=(0, 0, 1), size=(1, 1, 2))
    fragment = np.tile([0, 0.5, 1], (10000, 1))
    scene = Scene(points=np.vstack([fragment, surface(box, 2, -1)]),
                  boxes=[box])
    assert flag_insufficient_face_support(scene) == 1
    assert box.meta["face_support"]["matched_faces"] == 0


def test_partial_face_coverage_below_ninety_percent_fails():
    box = OrientedBox(center=(0, 0, 1), size=(1, 1, 2))
    face = surface(box, 1, 1)
    local = box.world_to_local(face)
    face = face[local[:, 0] < 0.0]  # cover only half the front face
    scene = Scene(points=face, boxes=[box])
    flag_insufficient_face_support(scene)
    front = box.meta["face_support"]["faces"]["front"]
    assert front["coverage"] < 0.90
    assert not front["passed"]


def test_three_faces_promote_low_confidence_once():
    box = OrientedBox(center=(0, 0, 1), size=(1, 1, 2),
                      confidence=Confidence.LOW)
    scene = Scene(points=np.vstack([surface(box, *s) for s in
                                    [(1, 1), (1, -1), (2, 1)]]), boxes=[box])
    assert flag_insufficient_face_support(scene) == 0
    assert box.confidence == Confidence.MID
    flag_insufficient_face_support(scene)
    assert box.confidence == Confidence.MID


@pytest.mark.parametrize("initial, expected", [
    (Confidence.HIGH, Confidence.MID),
    (Confidence.MID, Confidence.LOW),
    (Confidence.LOW, Confidence.LOW),
])
def test_missing_faces_demote_once(initial, expected):
    box = OrientedBox(center=(0, 0, 1), size=(1, 1, 2), confidence=initial)
    scene = Scene(points=np.empty((0, 3)), boxes=[box])
    flag_insufficient_face_support(scene)
    flag_insufficient_face_support(scene)
    assert box.confidence == expected


@pytest.mark.parametrize("flag", ["type_suspect", "structural_geom"])
def test_supported_faces_cannot_override_rejection(flag):
    box = OrientedBox(center=(0, 0, 1), size=(1, 1, 2),
                       confidence=Confidence.LOW, meta={flag: True})
    scene = Scene(points=np.vstack([surface(box, *s) for s in
                                    [(1, 1), (1, -1), (2, 1)]]), boxes=[box])
    flag_insufficient_face_support(scene)
    assert box.confidence == Confidence.LOW


@pytest.mark.parametrize("yaw", [0, math.pi / 2, 0.37])
def test_outer_thin_wall_with_empty_inner_face_is_removed(yaw):
    wall = OrientedBox(center=(0.2, 1.0, 1.0), size=(0.3, 2.0, 2.0),
                       yaw=yaw)
    outer = surface(wall, 0, -1)
    scene = Scene(points=outer, boxes=[wall])
    flag_insufficient_face_support(scene)
    assert filter_outer_wall_boxes(scene) == 1
    assert scene.boxes == []
    assert "wall_false_positive" in wall.meta
    assert scene.meta["wall_filter_audit"][0]["box_id"] == wall.box_id


def test_thick_box_containing_only_wall_and_air_is_removed():
    box = OrientedBox(center=(0.3, 1.0, 1.0), size=(1.0, 2.0, 2.0))
    scene = Scene(points=surface(box, 0, -1), boxes=[box])
    flag_insufficient_face_support(scene)
    assert filter_outer_wall_boxes(scene) == 1
    assert scene.boxes == []


@pytest.mark.parametrize("yaw", [0, 0.37, math.pi / 2])
def test_partial_outer_wall_face_is_removed(yaw):
    box = OrientedBox(center=(0, 0, 1), size=(1, 2, 2), yaw=yaw)
    wall = surface(box, 0, -1)
    local = box.world_to_local(wall)
    wall = wall[(np.abs(local[:, 1]) < 0.4) & (local[:, 2] < 0)]
    scene = Scene(points=wall, boxes=[box])
    flag_insufficient_face_support(scene)
    assert not box.meta["face_support"]["faces"]["left"]["passed"]
    assert filter_outer_wall_boxes(scene) == 1


def test_one_empty_face_does_not_remove_body_supported_equipment():
    box = OrientedBox(center=(0, 0, 1), size=(1, 2, 2))
    scene = Scene(points=np.vstack([surface(box, 0, -1), surface(box, 2, 1)]),
                  boxes=[box])
    flag_insufficient_face_support(scene)
    assert filter_outer_wall_boxes(scene) == 0
    assert scene.boxes == [box]


def test_thin_boundary_equipment_with_three_supported_faces_is_kept():
    box = OrientedBox(center=(0, 0, 1), size=(0.4, 1, 2))
    scene = Scene(points=np.vstack([surface(box, *s) for s in
                                    [(0, -1), (1, 1), (2, 1)]]), boxes=[box])
    flag_insufficient_face_support(scene)
    assert filter_outer_wall_boxes(scene) == 0
    assert scene.boxes == [box]


def test_supported_room_facing_face_is_not_mistaken_for_outer_wall():
    box = OrientedBox(center=(0, 0, 1), size=(0.4, 1, 2))
    # Extend room geometry past the supported positive-X face, so that it
    # is interior rather than the outer boundary. Negative-X face is empty.
    points = np.vstack([surface(box, 0, 1), [[3, 0, 1]]])
    scene = Scene(points=points, boxes=[box])
    flag_insufficient_face_support(scene)
    assert filter_outer_wall_boxes(scene) == 0
    assert scene.boxes == [box]


@pytest.mark.parametrize("yaw", [0, 0.37, math.pi / 2])
def test_long_thin_interior_wall_with_empty_top_is_removed(yaw):
    box = OrientedBox(center=(0, 0, 1), size=(3, 0.4, 2), yaw=yaw)
    points = surface(box, 1, 1)
    points = points[box.world_to_local(points)[:, 2] < 0.8]
    # Room extends past the candidate on every side: the older boundary
    # rule cannot remove this interior wall strip.
    points = np.vstack([points, box.local_to_world(np.array([
        [-5, -5, 1], [5, 5, 1]]))])
    scene = Scene(points=points, boxes=[box])
    flag_insufficient_face_support(scene)
    assert box.meta["face_support"]["faces"]["top"]["points"] == 0
    assert filter_outer_wall_boxes(scene) == 1
    assert not scene.boxes
    assert box.meta["wall_false_positive"]["top_points"] == 0


@pytest.mark.parametrize("size, with_top", [
    ((3, 1.1, 2), False),   # normal-depth joined cabinets
    ((1, 0.4, 2), False),   # short equipment
    ((3, 0.4, 2), True),    # thin equipment with a real top
])
def test_long_thin_rule_keeps_equipment_outside_its_conditions(size, with_top):
    box = OrientedBox(center=(0, 0, 1), size=size)
    points = surface(box, 1, 1)
    points = points[box.world_to_local(points)[:, 2] < 0.8]
    if with_top:
        points = np.vstack([points, surface(box, 2, 1)])
    points = np.vstack([points, [[-5, -5, 1], [5, 5, 1]]])
    scene = Scene(points=points, boxes=[box])
    flag_insufficient_face_support(scene)
    assert filter_outer_wall_boxes(scene) == 0
    assert scene.boxes == [box]
