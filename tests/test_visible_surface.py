import numpy as np

from agentic_gts.core.models import OrientedBox, Scene
from agentic_gts.output.gs_render import Cam, unproject_depth_mask
from agentic_gts.agent.mask_refine import (
    _apply_face_surface_constraints, _measure_face_surface,
    _measure_end_surface, _visible_mask_points)
from agentic_gts.tools.geometry import snap_box_faces, snap_box_ends, snap_row_seams


def camera():
    return Cam(eye=np.array([0., 3., 1.]), target=np.array([0., 0., 1.]),
               up=np.array([0., 0., 1.]), fovy_deg=60, W=32, H=32)


def test_depth_unprojection_roundtrip_and_alpha_exclusion():
    cam = camera()
    depth = np.full((32, 32), 2.5)
    alpha = np.ones((32, 32))
    mask = np.zeros((32, 32), bool)
    mask[10:20, 10:20] = True
    alpha[10, 10] = 0.1
    points = unproject_depth_mask(cam, depth, alpha, mask)
    assert len(points) == 99
    assert np.allclose(points[:, 1], 0.5)
    uv = cam.project_cv(points)
    assert np.allclose(uv, np.rint(uv))
    exclusion = mask.copy()
    assert len(unproject_depth_mask(cam, depth, alpha, mask, exclusion)) == 0


def measurement(face, cross):
    return {"face": face, "cross_local": cross, "points": 100,
            "along_local": [-2, 2], "spread_m": 0.01, "view": face}


def test_dual_and_single_surface_constraints_preserve_depth_steps():
    seed = OrientedBox(center=(0, 0, 1), size=(4, 1.6, 2))
    a = {"fitted": OrientedBox(center=(-1, 0, 1), size=(2, 1.6, 2)),
         "surface_measurements": [measurement("front", 0.5),
                                  measurement("back", -0.4)]}
    b = {"fitted": OrientedBox(center=(1, 0, 1), size=(2, 1.6, 2)),
         "surface_measurements": [measurement("front", 0.65)]}
    _apply_face_surface_constraints([a, b], seed)
    assert np.isclose(a["fitted"].size[1], 0.9)
    assert np.isclose(b["fitted"].center[1] + b["fitted"].size[1] / 2, 0.65)
    assert np.isclose(b["fitted"].center[1] - b["fitted"].size[1] / 2, -0.8)
    depths = [i["fitted"].size[1] for i in (a, b)]
    snap_row_seams([a["fitted"], b["fitted"]], 0, preserve_cross=True)
    assert [i["fitted"].size[1] for i in (a, b)] == depths


def test_observed_planes_do_not_move_to_other_profile_peaks():
    rng = np.random.default_rng(60)
    box = OrientedBox(center=(0, 0, 1), size=(2, 1, 2),
                       meta={"observed_faces": {"front": {}, "back": {}}})
    points = np.vstack([np.column_stack([
        rng.uniform(-1, 1, 1000), np.full(1000, y),
        rng.uniform(0.35, 2, 1000)]) for y in (-0.3, 0.3)])
    result, info = snap_box_faces(box, points)
    assert result is box
    assert not info["moved"]
    assert all(r["reason"] == "observed_surface_locked" for r in info["debug"])


def test_visible_depth_uses_geometry_only_as_validation_not_hidden_points():
    cam = camera()
    depth = np.full((32, 32), 2.5)
    alpha = np.ones((32, 32))
    mask = np.zeros((32, 32), bool)
    mask[5:27, 5:27] = True  # larger mask for better coverage
    surface = unproject_depth_mask(cam, depth, alpha, mask)
    hidden = surface + np.array([0, -0.5, 0])
    box = OrientedBox(center=(0, 0, 1), size=(2, 2, 2))
    scene = Scene(points=np.vstack([surface, hidden]))
    view = {"cam": cam, "depth": depth, "alpha": alpha, "name": "front"}
    points, source = _visible_mask_points(scene, box, mask, view)
    assert source == "gs_depth"
    assert np.allclose(points[:, 1], 0.5)
    assert _measure_face_surface(points, box, view) is not None
    assert _measure_face_surface(hidden + np.random.default_rng(1).normal(
        0, 0.5, hidden.shape), box, view) is None


def test_surface_constraints_survive_merge_and_primary_adoption():
    from agentic_gts.agent.mask_refine import _merge_cross_view, _build_split_pieces
    from agentic_gts.agent.loop import LayoutAgent

    seed = OrientedBox(center=(0, 0, 1), size=(4, 1.6, 2))
    points = np.column_stack([np.linspace(-1, 1, 100),
                              np.full(100, 0.5), np.ones(100)])
    spans = [{"lo": -1, "hi": 1, "pts": points, "ms": 0.9,
              "label": "rack", "view": face,
              "surface_measurements": [measurement(face, cross)]}
             for face, cross in (("front", 0.5), ("back", -0.4))]
    merged = _merge_cross_view(spans)
    instances = _build_split_pieces(merged, seed)
    _apply_face_surface_constraints(instances, seed)
    scene = Scene(points=points, boxes=[seed])
    result = LayoutAgent._adopt_refit(scene, seed, instances[0]["fitted"])
    assert set(result.meta["observed_faces"]) == {"front", "back"}
    assert result.meta["sam_split_seed"] == seed.box_id
    assert result.meta["surface_span_measured"]


def test_measured_instance_seam_does_not_average_clear_boundaries():
    a = OrientedBox(center=(-0.5, 0, 1), size=(1, 1, 2),
                    meta={"surface_span_measured": True})
    b = OrientedBox(center=(0.55, 0, 1), size=(1, 1, 2),
                    meta={"surface_span_measured": True})
    assert snap_row_seams([a, b], 0) == 0
    assert a.center == (-0.5, 0, 1)
    assert b.center == (0.55, 0, 1)


def side_camera():
    """Camera looking along the row from the right (+X side)."""
    return Cam(eye=np.array([3., 0., 1.]), target=np.array([0., 0., 1.]),
               up=np.array([0., 0., 1.]), fovy_deg=60, W=32, H=32)


def end_measurement(face, along):
    """Helper to create an end-surface measurement (left/right from side view)."""
    return {"face": face, "along_local": along, "points": 100,
            "cross_local": [-1, 1], "spread_m": 0.01, "view": "side"}


def test_side_end_surface_measurement_from_visible_points():
    """Test that side view can measure left/right end surfaces."""
    cam = side_camera()
    depth = np.full((32, 32), 2.5)
    alpha = np.ones((32, 32))
    mask = np.zeros((32, 32), bool)
    mask[5:27, 5:27] = True  # larger mask for better coverage
    surface = unproject_depth_mask(cam, depth, alpha, mask)
    box = OrientedBox(center=(0, 0, 1), size=(4, 2, 2))
    view = {"cam": cam, "depth": depth, "alpha": alpha, "name": "side"}
    m = _measure_end_surface(surface, box, view)
    assert m is not None
    assert m["face"] == "right"
    assert -0.5 <= m["along_local"] <= 0.5
    assert m["points"] == len(surface)


def test_side_end_constraints_adjust_piece_length():
    """Test that left/right end measurements constrain piece length."""
    seed = OrientedBox(center=(0, 0, 1), size=(4, 1.6, 2))
    a = {"fitted": OrientedBox(center=(-1, 0, 1), size=(2, 1.6, 2)),
         "surface_measurements": [end_measurement("left", -1.8),
                                  end_measurement("right", -0.4)]}
    b = {"fitted": OrientedBox(center=(1, 0, 1), size=(2, 1.6, 2)),
         "surface_measurements": [end_measurement("left", 0.3),
                                  end_measurement("right", 1.75)]}
    _apply_face_surface_constraints([a, b], seed)
    assert np.isclose(a["fitted"].size[0], 1.4)
    assert np.isclose(a["fitted"].center[0], -1.1)
    assert np.isclose(b["fitted"].size[0], 1.45)
    assert np.isclose(b["fitted"].center[0], 1.025)
    assert a["fitted"].meta["surface_span_measured"]
    assert "left" in a["fitted"].meta["observed_faces"]
    assert "right" in a["fitted"].meta["observed_faces"]


def test_combined_front_back_and_left_right_constraints():
    """Test that all four face constraints work together."""
    seed = OrientedBox(center=(0, 0, 1), size=(4, 1.6, 2))
    inst = {"fitted": OrientedBox(center=(0, 0, 1), size=(2, 1.6, 2)),
            "surface_measurements": [
                measurement("front", 0.65),
                measurement("back", -0.45),
                end_measurement("left", -0.9),
                end_measurement("right", 0.85)
            ]}
    _apply_face_surface_constraints([inst], seed)
    assert np.isclose(inst["fitted"].size[0], 1.75)
    assert np.isclose(inst["fitted"].size[1], 1.10)
    assert np.isclose(inst["fitted"].center[0], -0.025)
    assert np.isclose(inst["fitted"].center[1], 0.10)
    assert inst["fitted"].meta["surface_span_measured"]
    faces = inst["fitted"].meta["observed_faces"]
    assert set(faces.keys()) == {"front", "back", "left", "right"}


def test_observed_end_surfaces_do_not_snap_in_stage_f():
    """Test that boxes with observed end surfaces resist Stage F end snapping."""
    box = OrientedBox(center=(0, 0, 1), size=(2, 1, 2),
                      meta={"observed_faces": {"left": {}, "right": {}}})
    rng = np.random.default_rng(61)
    points = np.vstack([np.column_stack([
        np.full(1000, x), rng.uniform(-0.3, 0.3, 1000),
        rng.uniform(0.35, 2, 1000)]) for x in (-0.8, 0.8)])
    result, info = snap_box_ends(box, points)
    assert result is box
    assert not info["moved"]
    assert all(r["reason"] == "observed_surface_locked" for r in info["debug"])
