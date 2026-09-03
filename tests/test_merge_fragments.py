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


def test_face_support_rescues_single_view_fragment():
    """A full-depth box with only ONE observed face: interior support is
    structurally low, face support must rescue it; a floating box with no
    observation gets nothing."""
    rng = np.random.default_rng(5)
    a = rng.uniform(-0.3, 0.3, 3000)
    z = rng.uniform(0, 2.0, 3000)
    c = np.full(3000, 0.55)                     # only the front face observed
    scene = Scene(points=np.stack([a, c, z], axis=1))
    frag = _mk(0.0, 0.0, 0.6, 1.1, 2.0)
    interior = geo.support_fraction(scene, frag)
    face = geo.face_support_fraction(scene, frag)
    assert interior < 0.15, f"interior {interior:.2f} (expected structurally low)"
    assert face > 0.5, f"face {face:.2f} (observed face should be covered)"
    # floor-backed false positive must NOT be rescued via the bottom face
    floor = np.stack([rng.uniform(3.0, 5.0, 2000),
                      rng.uniform(3.0, 5.0, 2000),
                      np.zeros(2000)], axis=1)
    scene2 = Scene(points=floor)
    fp = _mk(4.0, 4.0, 0.6, 1.1, 2.0)           # sits on the floor, empty
    assert geo.face_support_fraction(scene2, fp) < 0.1, \
        "floor points leaked through the bottom-face rescue"
    print(f"PASS face-support: fragment interior={interior:.2f} -> face={face:.2f}; "
          f"floor-backed FP face={geo.face_support_fraction(scene2, fp):.2f}")


def test_trust_input_boxes_skips_wall_filter():
    """A single-view thin-sheet fragment must survive the wall filter when
    the boxes are trusted external input, while the wall filter still runs
    (and drops the same sheet) for geometric Stage-A candidates."""
    rng = np.random.default_rng(7)
    # one long thin vertical sheet: indistinguishable from a wall fragment
    # by geometry alone -- only the input's provenance separates them
    a = rng.uniform(-2.0, 2.0, 8000)
    z = rng.uniform(0, 2.2, 8000)
    c = np.zeros(8000) + rng.normal(0, 0.05, 8000)
    scene = Scene(points=np.stack([a, c, z], axis=1))
    from agentic_gts.rules.rules import apply_rules
    from agentic_gts.core.models import OrientedBox as OB
    # 2.5m sheet: big enough to be classified as a wall sheet by the default
    # path. In the trusted path it survives the wall filter (the aspect
    # splitter may cut it into rack-width pieces -- correct, allowed).
    frag = OB(center=(0.0, 0.0, 1.1), size=(2.5, 0.3, 2.2), yaw=0.0)

    scene.boxes = [frag]
    _, _ = apply_rules(scene, opts={"yaw": 0.0, "trust_input_boxes": True})
    kept_trusted = len(scene.boxes)
    assert kept_trusted >= 1, f"trusted fragment was dropped ({kept_trusted} left)"

    scene.boxes = [OB(center=(0.0, 0.0, 1.1), size=(2.5, 0.3, 2.2), yaw=0.0)]
    _, _ = apply_rules(scene, opts={"yaw": 0.0})
    kept_default = len(scene.boxes)
    assert kept_default == 0, f"untrusted wall-like sheet survived ({kept_default} left)"
    print(f"PASS trust-input: thin sheet kept={kept_trusted} (trusted) vs "
          f"kept={kept_default} (default path)")


def _mk(cx: float, cy: float, L: float, W: float, H: float):
    from agentic_gts.core.models import OrientedBox
    return OrientedBox(center=(cx, cy, H / 2), size=(L, W, H), yaw=0.0)


def test_trust_input_no_synthetic_boxes():
    """Trusted detector input: rules must NOT add boxes, must merge fine
    fragments, and must leave a wide (merged-rack) box untouched for the
    agent to split -- no geometry priors sprinkled on trusted input."""
    from agentic_gts.rules.rules import apply_rules
    from agentic_gts.core.models import DeviceType as DT

    rng = np.random.default_rng(9)
    pts = []
    # a row of contiguous racks
    for k in range(6):
        cx = k * 0.7
        for _ in range(1500):
            pts.append([cx + rng.uniform(-0.3, 0.3), rng.uniform(-0.55, 0.55),
                        rng.uniform(0.0, 2.0)])
    scene = Scene(points=np.array(pts))

    # one rack split into 3 fine fragments + one 2-rack merged box + one clean
    scene.boxes = [_mk(0.0, 0.0, 0.6, 1.1, 2.0),
                   _mk(0.35, 0.05, 0.35, 1.0, 2.0),
                   _mk(0.15, -0.05, 0.3, 1.0, 2.0),
                   _mk(0.7, 0.0, 1.35, 1.1, 2.0),   # two racks fused
                   _mk(1.4, 0.0, 0.6, 1.1, 2.0)]    # clean rack
    n_in = len(scene.boxes)
    boxes, _ = apply_rules(scene, opts={"yaw": 0.0, "trust_input_boxes": True})

    # 1) no rule-synthesized boxes: the count should stay tiny (5 -> 3 after
    #    the two extra fragment boxes were absorbed into the first rack)
    assert len(boxes) < n_in, f"rules added boxes ({n_in}->{len(boxes)})"
    # 2) no box wider than a single rack unit survived as-is (the merged one
    #    must be preserved FOR THE AGENT, not erased) -- so exactly one wide
    #    box remains
    wide = [b for b in boxes if b.size[0] > 1.2]
    assert len(wide) == 1, f"expected 1 merged-rack box preserved, got {len(wide)}"
    # 3) the fine fragments were merged: no leftover 0.3-0.35m slivers
    tiny = [b for b in boxes if b.size[0] < 0.45]
    assert len(tiny) == 0, f"fine fragments not merged: {[b.size[0] for b in tiny]}"
    print(f"PASS trust-input: {n_in}->{len(boxes)} boxes, "
          f"{len(wide)} wide kept for agent, no rule-synthesized boxes")


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
