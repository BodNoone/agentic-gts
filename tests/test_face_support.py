import json
import math

import numpy as np
import pytest

from agentic_gts.core.models import Confidence, OrientedBox, Scene
from agentic_gts.tools.geometry import flag_insufficient_face_support


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
    assert box.confidence == (Confidence.LOW if face_count < 3
                              else Confidence.HIGH)
    json.dumps(box.to_dict())


def test_dense_fragment_and_floor_do_not_qualify_faces():
    box = OrientedBox(center=(0, 0, 1), size=(1, 1, 2))
    fragment = np.tile([0, 0.5, 1], (10000, 1))
    scene = Scene(points=np.vstack([fragment, surface(box, 2, -1)]),
                  boxes=[box])
    assert flag_insufficient_face_support(scene) == 1
    assert box.meta["face_support"]["matched_faces"] == 0


def test_three_faces_do_not_promote_existing_low_confidence():
    box = OrientedBox(center=(0, 0, 1), size=(1, 1, 2),
                      confidence=Confidence.LOW)
    scene = Scene(points=np.vstack([surface(box, *s) for s in
                                    [(1, 1), (1, -1), (2, 1)]]), boxes=[box])
    assert flag_insufficient_face_support(scene) == 0
    assert box.confidence == Confidence.LOW
