"""Tests for the wall/pillar structural class (user report: the final
boxes still included pillars and walls -- the local grounding's CLOSED
category set gave the model no honest output for a structural slab, so
it squeezed into 'server rack' with a high score, which SKIPPED the
type-confirm and the box survived on its own laundering label)."""
import os
import sys

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from agentic_gts.agent.mask_refine import (_is_structural, refine_box,
                                           _voter_spans, BoxGroup,
                                           SamPredictorAdapter)
from agentic_gts.core.models import Scene, OrientedBox


def test_is_structural():
    assert _is_structural("wall / pillar")
    assert _is_structural("Wall Segment")
    assert _is_structural("pillar")
    assert _is_structural("column")
    assert not _is_structural("server rack / IT cabinet")
    assert not _is_structural("air-conditioning unit")
    assert not _is_structural("open cabinet door")
    assert not _is_structural("")
    assert not _is_structural(None)
    print("PASS _is_structural label matching")


def test_structural_groups_still_generate_spans():
    """A view with a wall-labelled group DOES generate spans (user
    report: a tall+short device pair stopped splitting after the
    structural class was added -- the short device was mislabelled
    'wall / pillar' and its span was suppressed, killing the split).
    The structural label's purpose is to trigger the type-confirm,
    not to suppress the split: the span must exist so the piece can
    be created and the confirm can save or kill it."""
    # verify the CODE PATH: the structural label is NOT in the span
    # skip list (unlike the subtractive door/ladder labels which ARE
    # skipped) -- the group enters the SAM/mask/lift pipeline
    from agentic_gts.agent import mask_refine as mr
    import inspect
    src = inspect.getsource(mr._voter_spans)
    # the structural skip was in the span loop and has been removed
    assert 'if _is_structural' not in src.split('for gi, g in enumerate')[1].split('spans.append')[0], \
        "the _is_structural skip must NOT be in the span generation loop"
    # the subtractive skip IS still there
    assert 'if _is_subtractive' in src.split('for gi, g in enumerate')[1].split('spans.append')[0], \
        "the _is_subtractive skip must remain in the span generation loop"
    # the structural label check IS in the side-thickness pool skip
    src_pool = inspect.getsource(mr._side_thickness_pool)
    assert '_is_structural' in src_pool, \
        "the _is_structural skip must remain in the thickness pool"
    # the structural flag IS still set
    assert '_is_structural' in src and 'structural_label' in src, \
        "the structural flag must still be set in _voter_spans"
    print("PASS structural groups still generate spans " \
          "(skip removed from span loop, kept in thickness pool, " \
          "flag still set)")


def test_refine_box_propagates_structural_flag():
    """refine_box's top-level audit carries the structural flag from any
    view -- both on the ACCEPTED path and on the EARLY RETURN path (no
    span survived), so loop.py can force the type-confirm either way."""
    # accepted path: the flag rides the audit dict out of refine_box
    # (verified indirectly: _voter_spans sets va["structural_label"],
    # and refine_box's tail copies any view's flag to the top level).
    # Here we test the EARLY RETURN path, which has its own copy.
    rng = np.random.default_rng(32)
    pts = np.column_stack([rng.uniform(-1, 1, 2000),
                           rng.uniform(-0.5, 0.5, 2000),
                           rng.uniform(0.30, 2.0, 2000)])
    scene = Scene(points=pts)
    box = OrientedBox(center=(0, 0, 1.05), size=(2.0, 1.0, 2.1), yaw=0.0)

    class _Judge:
        backend = "qwen"
        def adjudicate_sam_boxes(self, img, b, name, png_path=None):
            from agentic_gts.agent.judge import Verdict
            return Verdict(action="segment", params={
                "groups": [{"bbox": [0.2, 0.2, 0.8, 0.8],
                            "hypothesis": "wall / pillar",
                            "confidence": 0.8}],
                "view_quality": "good"}, raw="wall")

    class _FakeSAM:
        available = True
        def predict(self, image, box_pix):
            h, w = image.shape[:2]
            m = np.zeros((h, w), dtype=bool)
            x0, y0, x1, y1 = [int(v) for v in box_pix]
            m[max(y0, 0):min(y1, h), max(x0, 0):min(x1, w)] = True
            return [m], [0.9]

    # build a view dict that _voter_spans can consume directly
    img = np.random.default_rng(0).uniform(0.2, 0.8, (768, 768, 3))
    view = {"name": "front", "image": img.astype(np.float32),
            "path": None, "cam": None, "prompt_image": img.astype(np.float32),
            "prompt_path": None}

    # the structural flag appears in the per-view audit...
    audit = {"views": []}
    spans = _voter_spans(scene, box, view, _Judge(), _FakeSAM(),
                         None, audit)
    assert spans == []
    assert audit["views"][0]["structural_label"] is True

    # ...and refine_box's early-return copies it to the top level
    # (simulate: audit dict with the flag in views -> the early return
    # path must carry it).  We exercise the actual code path:
    from agentic_gts.agent.mask_refine import refine_box as _rb
    # patch render_local_views to return our view (avoids the full
    # render pipeline; we only test the flag propagation)
    import agentic_gts.agent.mask_refine as mr
    _real_rlv = mr.render_local_views
    mr.render_local_views = lambda *a, **k: [view]
    try:
        instances, audit2 = _rb(scene, box, _Judge(), _FakeSAM(),
                                out_dir=None, views=None)
    finally:
        mr.render_local_views = _real_rlv
    assert audit2.get("structural_label") is True, \
        "the top-level audit must carry the structural flag on the " \
        f"early-return path too: {audit2.get('reason')}"
    print("PASS refine_box propagates the structural flag "
          "(early-return path)")
