"""Stage B: post-processing of candidate boxes.

Two input regimes share this module:

  * trusted detector boxes (multi-view 2D segmentation back-projected to 3D):
    semantically trustworthy and roughly correct -- just fragmented, slightly
    rotated, or with adjacent racks fused. The rules here must NOT filter,
    add, or hard-split them (the agent owns those decisions).
  * geometry candidates (no boxes: VLM god-view + row-fit, or Stage-A coarse):
    these need a light cleanup for obvious hiccups (wall-hugging, wrong
    orientation), but still no aggressive deletion/addition/splitting --
    those are discriminant problems better left to the agent.

Division of labour (kept deliberately narrow):

  fuse_fragments(...)         pure deterministic B0 merge -- the ONLY rule the
                              trusted path runs. Not a discriminant problem:
                              merging same-device fragments is geometry, and
                              doing it before the agent avoids a swarm of
                              fragments each getting an (inconsistent) VLM call.
  clean_geometry_candidates() service the no-box path: remove wall-hugging
                              sheets, snap orientation to the dominant row.
                              NO support/aspect deletion, NO row-fill, NO
                              hard split -- those delete/add/split, which is
                              the agent's call.
  clean_detector_boxes()      service the trusted path: no-op by design (the
                              agent subsumes all refinement).

`apply_rules` is kept as a thin dispatch entry for call-site compatibility
and for tests; it picks the right clean function from trust_input_boxes.
"""
from __future__ import annotations

import math

import numpy as np

from agentic_gts.core.models import (
    BoxSource,
    Confidence,
    DeviceType,
    OrientedBox,
    Issue,
    IssueType,
    Scene,
)
from agentic_gts.tools import geometry as geo


def fuse_fragments(scene: Scene, yaw: float = 0.0, trusted: bool = False,
                   width_unit: float = 0.6) -> tuple[list[OrientedBox], int]:
    """B0: fuse per-view back-projection fragments of the SAME device.

    Pure geometry; never filters, adds, or splits. A box clearly WIDER than a
    single rack unit (i.e. a genuine multi-device / merged-rack box) is held
    out so fragment merging does not absorb it -- that is the agent's to split.
    The bar is deliberately loose (2.0 x unit) so a single device's front/back
    or side fragments, which can be close to one rack width, still participate
    in the merge. `trusted` disables the left/right complementary merge
    (usually adjacent devices, not a split one).

    Returns (boxes, n_absorbed).
    """
    wide = [b for b in scene.boxes if b.size[0] > width_unit * 2.0]
    frag = [b for b in scene.boxes if b.size[0] <= width_unit * 2.0]
    tmp = Scene(points=scene.points, boxes=frag)
    merged, n_absorbed = geo.merge_fragments(tmp, yaw=yaw, trusted=trusted)
    return merged + wide, n_absorbed


def clean_detector_boxes(scene: Scene, opts: dict | None = None) -> list[Issue]:
    """Service the trusted (detector-input) path: near no-op.

    The detector boxes are semantically trustworthy; the agent subsumes
    refinement (split / shrink / delete / gap-fill). Deliberately nothing
    here filters, adds or splits. Returns an empty issue list.

    (Kept as a distinct function so the two paths are explicit and so future
    light safety checks can live here without touching the no-box path.)
    """
    return []


def clean_geometry_candidates(scene: Scene,
                              opts: dict | None = None) -> list[Issue]:
    """Service the no-box (geometry/VLM-god-view) path: light cleanup only.

    Removes wall-hugging thin sheets (an obvious artefact) and snaps each
    box's orientation to the dominant row direction. It deliberately does NOT
    do anything that adds or deletes semantically-meaningful candidates:
    no support/aspect deletion, no row-fill, no hard merge-row split -- those
    are the agent's judgement calls.
    """
    opts = opts or {}
    yaw = float(opts.get("yaw", scene.meta.get("yaw", 0.0)))
    issues: list[Issue] = []

    # wall-sheet removal (obvious artefact, safe for geometry candidates)
    non_wall: list[OrientedBox] = []
    for b in scene.boxes:
        if _is_wall_sheet(scene, b):
            issues.append(Issue(IssueType.FALSE_POSITIVE, [b.box_id],
                                _box_region(b), detail="wall sheet"))
            continue
        non_wall.append(b)
    scene.boxes = non_wall

    # orientation snap to the dominant row direction (no force per-box motion)
    rows = geo.row_structure(scene, yaw=yaw)
    for r in rows:
        axis = np.asarray(r["axis"])
        fixed_yaw = math.atan2(axis[1], axis[0])
        for b in r["boxes"]:
            if not geo.is_aligned(b, axis, tol_deg=12.0):
                b.yaw = fixed_yaw
                b.source = BoxSource.RULE_FIX
                issues.append(Issue(IssueType.MISALIGNED, [b.box_id], _box_region(b)))
    return issues


def apply_rules(scene: Scene, opts: dict | None = None) -> tuple[list[OrientedBox], list[Issue]]:
    """Compatibility entry point. Dispatches on trust_input_boxes.

    Always runs fuse_fragments (B0) first -- the one rule that must precede
    the agent so it does not adjudicate a cloud of fragments. Then applies the
    path-appropriate clean function. Returns (scene.boxes, issues).

    NOTE: for the trusted path this performs only the merge (no filter/add/
    split); for the no-box path a light cleanup minus the aggressive rules.
    """
    opts = opts or {}
    yaw = float(opts.get("yaw", scene.meta.get("yaw", 0.0)))
    width_unit = float(opts.get("width_unit", 0.6))

    merged, n_absorbed = fuse_fragments(scene, yaw=yaw, trusted=opts.get("trust_input_boxes", False),
                                        width_unit=width_unit)
    scene.boxes = merged
    print(f"[diag][B] fragment merge (B0): absorbed {n_absorbed} -> {len(merged)} boxes")

    if opts.get("trust_input_boxes"):
        issues = clean_detector_boxes(scene, opts)
        print("[diag][B] cleanup: none (trusted input; agent subsumes refinement)")
    else:
        issues = clean_geometry_candidates(scene, opts)
        print(f"[diag][B] cleanup: wall/orientation only -> {len(scene.boxes)} boxes, "
              f"{len(issues)} issues (no delete/add/split)")
    return scene.boxes, issues


def refine_boxes(scene: Scene, opts: dict | None = None) -> tuple[list[OrientedBox], list[Issue]]:
    return apply_rules(scene, opts)


def _box_region(b: OrientedBox) -> tuple[float, float, float, float]:
    c = np.asarray(b.center[:2])
    half = np.asarray(b.size[:2]) / 2.0
    r = b.rotation[:2, :2]
    corners = np.array([[-1, -1], [1, -1], [1, 1], [-1, 1]]) * half
    world = corners @ r.T + c
    return (float(world[:, 0].min()), float(world[:, 1].min()),
            float(world[:, 0].max()), float(world[:, 1].max()))


def _is_wall_sheet(scene: Scene, box: OrientedBox, depth_thr: float = 0.35) -> bool:
    """Wall fragments are thin sheets: points occupy a narrow cross-axis slab.

    A real rack shows front+back faces ~deep apart, so its point distribution
    along the cross (depth) axis is either wide (>0.5m) or bimodal. A wall
    gives a single narrow slab. We additionally require the box to be flat
    (depth << along length) to avoid killing short AC units.

    The tall-column test catches what the thin-sheet test misses: thick /
    noisy 3DGS walls whose depth spread exceeds depth_thr. We voxelize the
    box footprint and apply the shared wall_column_mask test (structure
    top above TALL_Z AND the high part itself spans >=0.5m -- the cable-tray
    guard, see wall_column_mask). A wall box has nearly all columns tall;
    a rack with an overhead tray or next to a wall does not.
    """
    from agentic_gts.segment.coarse import wall_column_mask
    region = _box_region(box)
    pts = scene.points_in_region(region)
    big = box.size[0] >= 2.0 or box.size[1] >= 2.0
    if len(pts) < 20:
        return big  # no support -> prune big empty boxes, keep small ones
    p = pts[pts[:, 2] > 0.3]  # drop floor points
    if len(p) >= 20:
        key = np.floor(p[:, :2] / 0.15).astype(np.int64)
        uniq, inv = np.unique(key, axis=0, return_inverse=True)
        wall = wall_column_mask(p[:, 2], inv, len(uniq))
        if len(wall) and float(np.mean(wall)) > 0.5:
            return True
    if not big:
        return False  # small boxes are never thin-sheet walls
    local = box.world_to_local(pts)
    ydepth = local[:, 1]
    q10, q90 = np.percentile(ydepth, [10, 90])
    spread = q90 - q10
    depth = box.size[1]
    flat = depth <= depth_thr or (spread <= depth_thr and box.size[0] >= max(box.size[1] * 3, 0.8))
    return flat and (spread <= depth_thr)
