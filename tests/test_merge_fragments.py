"""B0 fragment-merge tests: per-view back-projection fragment scenarios."""
from __future__ import annotations

import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np

from agentic_gts.core.models import Scene
from agentic_gts.tools import geometry as geo


def _rack_cloud(cx: float, cy: float, yaw: float = 0.0,
                L: float = 0.6, W: float = 1.1, H: float = 2.0,
                n: int = 4000, seed: int = 0) -> np.ndarray:
    rng = np.random.default_rng(seed)
    cos, sin = np.cos(yaw), np.sin(yaw)
    axis = np.array([cos, sin])
    cross = np.array([-sin, cos])

    def _world(a: np.ndarray, c: np.ndarray, z: np.ndarray) -> np.ndarray:
        return np.stack([axis[0] * a + cross[0] * c + cx,
                         axis[1] * a + cross[1] * c + cy, z], axis=1)

    pts = []
    for fa, fc in [(0.5, 0.5), (-0.5, 0.5), (0.5, -0.5), (-0.5, -0.5)]:
        a = rng.uniform(-L / 2, L / 2, n // 8)
        c = rng.uniform(-W / 2, W / 2, n // 8)
        z = rng.uniform(0, H, n // 8)
        pts.append(_world(a, c, z) + np.full((n // 8, 3), [0, 0, 0])
                   + fa * 0 + 0)  # corners not needed; faces below
    # four side faces
    for fixed, val in [(0, L / 2), (0, -L / 2), (1, W / 2), (1, -W / 2)]:
        u = rng.uniform(-L / 2, L / 2, n // 4)
        v = rng.uniform(0, H, n // 4)
        if fixed == 0:
            pts.append(_world(np.full(n // 4, val), u, v))
        else:
            pts.append(_world(u, np.full(n // 4, val), v))
    return np.vstack(pts)


def test_same_spot_fragments_merge():
    """Two overlapping fragments of the same rack -> one box."""
    cloud = _rack_cloud(2.0, 3.0)
    scene = Scene(points=cloud)
    # view A sees the left half, view B the right half (footprints overlap)
    scene.boxes = [
        _mk(1.85, 3.0, 0.35, 1.1, 2.0),
        _mk(2.15, 3.0, 0.35, 1.1, 2.0),
    ]
    out, absorbed = geo.merge_fragments(scene, yaw=0.0)
    assert len(out) == 1 and absorbed == 1, (len(out), absorbed)
    b = out[0]
    assert abs(b.center[0] - 2.0) < 0.15, b.center
    assert 0.4 < b.size[0] < 0.9, b.size
    print(f"PASS same-spot: 2 -> {len(out)}, size={tuple(round(s,2) for s in b.size)}")


def test_front_back_fragments_merge():
    """Front-half + back-half fragments of the same rack -> one box."""
    cloud = _rack_cloud(0.0, 0.0)
    scene = Scene(points=cloud)
    # fragment 1 covers cross in [-0.55, -0.05], fragment 2 in [0.0, 0.55]
    scene.boxes = [
        _mk(0.0, -0.30, 0.6, 0.5, 2.0),
        _mk(0.0, 0.28, 0.6, 0.55, 2.0),
    ]
    out, absorbed = geo.merge_fragments(scene, yaw=0.0)
    assert len(out) == 1 and absorbed == 1, (len(out), absorbed)
    b = out[0]
    assert abs(b.center[1]) < 0.15, b.center
    assert 0.8 < b.size[1] < 1.4, b.size
    print(f"PASS front-back: 2 -> {len(out)}, depth={b.size[1]:.2f}")


def test_adjacent_racks_not_merged():
    """Two distinct racks side by side in a row must stay separate."""
    cloud = np.vstack([_rack_cloud(0.0, 0.0, seed=1), _rack_cloud(0.61, 0.0, seed=2)])
    scene = Scene(points=cloud)
    scene.boxes = [
        _mk(0.0, 0.0, 0.6, 1.1, 2.0),
        _mk(0.61, 0.0, 0.6, 1.1, 2.0),
    ]
    out, absorbed = geo.merge_fragments(scene, yaw=0.0)
    assert len(out) == 2, f"adjacent racks wrongly merged ({absorbed} absorbed)"
    print("PASS adjacent racks kept separate")


def test_back_to_back_racks_not_merged():
    """Back-to-back racks (cross union ~2.2m) must stay separate."""
    cloud = np.vstack([_rack_cloud(0.0, 0.0, seed=3), _rack_cloud(0.0, 1.2, seed=4)])
    scene = Scene(points=cloud)
    scene.boxes = [
        _mk(0.0, 0.0, 0.6, 1.1, 2.0),
        _mk(0.0, 1.2, 0.6, 1.1, 2.0),
    ]
    out, absorbed = geo.merge_fragments(scene, yaw=0.0)
    assert len(out) == 2, f"back-to-back wrongly merged ({absorbed} absorbed)"
    print("PASS back-to-back racks kept separate")


def _mk(cx: float, cy: float, L: float, W: float, H: float):
    from agentic_gts.core.models import OrientedBox
    return OrientedBox(center=(cx, cy, H / 2), size=(L, W, H), yaw=0.0)


if __name__ == "__main__":
    fns = [v for k, v in list(globals().items()) if k.startswith("test_")]
    passed = 0
    for fn in fns:
        try:
            fn()
            passed += 1
        except AssertionError as e:
            print(f"FAIL  {fn.__name__}: {e}")
        except Exception as e:
            print(f"ERROR {fn.__name__}: {type(e).__name__}: {e}")
    print(f"\n{passed}/{len(fns)} tests passed")
