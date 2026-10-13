import numpy as np
import pytest

from agentic_gts.agent import mask_refine as mr
from agentic_gts.agent.judge import Verdict
from agentic_gts.core.models import OrientedBox, Scene


def candidates():
    clean = np.zeros((64, 64), dtype=bool)
    clean[10:50, 10:50] = True
    panel = np.zeros_like(clean)
    panel[20:35, 20:35] = True
    spilled = clean.copy()
    spilled[10:50, 50:62] = True
    return clean, panel, spilled


def test_grounding_fit_rewards_complete_mask_and_penalizes_spill():
    clean, panel, spilled = candidates()
    box = np.array([10, 10, 50, 50])
    scores = [mr._mask_grounding_fit(m, box) for m in (clean, panel, spilled)]
    assert scores[0]["grounding_fit"] == 1.0
    assert scores[0]["grounding_fit"] > scores[1]["grounding_fit"]
    assert scores[0]["grounding_fit"] > scores[2]["grounding_fit"]
    assert scores[2]["outside_pixels"] == 480


@pytest.mark.parametrize("view_name", ["front", "back"])
@pytest.mark.parametrize("clean_valid, selected", [(True, 0), (False, 2)])
def test_voter_uses_grounding_fit_before_sam_score(monkeypatch, view_name,
                                                  clean_valid, selected):
    masks = candidates()
    scores = np.array([0.6, 0.95, 0.99])
    points = np.column_stack([np.linspace(-0.4, 0.4, 100),
                              np.zeros(100), np.ones(100)])
    class Judge:
        def adjudicate_sam_boxes(self, *args, **kwargs):
            return Verdict(action="segment", params={"groups": [{
                "bbox": [10 / 63 * 1000, 10 / 63 * 1000,
                         50 / 63 * 1000, 50 / 63 * 1000],
                "hypothesis": "rack"}], "view_quality": "good"})

    class Sam:
        def predict(self, *args):
            return masks, scores

    def lift(scene, box, mask, cam, **kwargs):
        return points[:0] if mask is masks[0] and not clean_valid else points

    monkeypatch.setattr(mr, "_mask_to_points", lift)
    box = OrientedBox(center=(0, 0, 1), size=(2, 1, 2))
    view = {"name": view_name, "image": np.zeros((64, 64, 3)),
            "cam": None, "path": None}
    audit = {"views": []}
    spans = mr._voter_spans(Scene(points=points), box, view, Judge(), Sam(),
                             None, audit)
    assert len(spans) == 1
    selection = audit["views"][0]["mask_selection"][0]
    assert selection["selected_mask_index"] == selected
    assert len(selection["candidates"]) == 3
    assert spans[0]["ms"] == scores[selected]
