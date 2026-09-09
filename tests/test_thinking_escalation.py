"""Tests for the thinking-model escalation tier in VLMJudge.

No network / no model weights: the backend calls are monkeypatched, the
escalation TRIGGERS are driven by injected render-quality dicts. What is
verified:
  - think-block stripping (closed / truncated / channel syntax)
  - _should_escalate gating (quality floor + configured model)
  - _parse_fit_reply keep_allok semantics
  - adjudicate_box / adjudicate_fit escalation flow: the thinking verdict
    replaces the fast one, failures keep the primary, records carry
    escalated=True
"""
import json
import os
import sys
import tempfile
import types

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from agentic_gts.agent.judge import VLMJudge, Verdict
from agentic_gts.core.models import OrientedBox

_T = "<" + "think>"
_TC = "</" + "think>"


def test_strip_think_variants():
    s = VLMJudge._strip_think
    # closed block before the answer
    assert s(f"{_T}reasoning...{_TC}\nfinal answer") == "final answer"
    # truncated block: nothing after the opener is trustworthy
    assert s(f"{_T}chain that never ends") == ""
    # clean reply is untouched
    assert s("plain reply") == "plain reply"
    # newer channel syntax
    ch = "<|channel|>analysis<|message|>chain<|end|><|channel|>final<|message|>answer"
    assert s(ch).endswith("answer")
    # think text must NOT leak into an option match
    text = s(f"{_T}maybe split... but actually keep{_TC}\nkeep")
    assert "keep" in text and "split" not in text
    print("PASS strip think variants")


def test_should_escalate_gating():
    j = VLMJudge(backend="qwen", thinking_model="Qwen/Qwen3-VL-8B-Thinking")
    low = {"front": {"score": 0.2, "visibility": 0.9, "clearance": 1.0}}
    high = {"front": {"score": 0.8, "visibility": 0.9, "clearance": 1.0}}
    assert j._should_escalate(low) is True
    assert j._should_escalate(high) is False
    assert j._should_escalate(None) is False            # unknown -> 1.0 floor
    # occlusion / embedded camera also count as hard evidence
    occl = {"front": {"score": 0.9, "visibility": 0.1, "clearance": 1.0}}
    embed = {"front": {"score": 0.9, "visibility": 0.9, "clearance": -0.2}}
    assert j._should_escalate(occl) is True
    assert j._should_escalate(embed) is True
    # no thinking model configured -> never escalate
    j2 = VLMJudge(backend="qwen")
    assert j2._should_escalate(low) is False
    print("PASS should_escalate gating")


def test_api_target_routes():
    j = VLMJudge(backend="qwen", api_base="http://a/v1",
                thinking_model="mt", thinking_api_base="http://b/v1",
                thinking_timeout=123)
    assert j._api_target(False) == ("http://a/v1", "EMPTY", j.model, 60)
    assert j._api_target(True) == ("http://b/v1", "EMPTY", "mt", 123)
    # thinking base defaults to the primary base
    j2 = VLMJudge(backend="qwen", api_base="http://a/v1", thinking_model="mt")
    assert j2._api_target(True)[0] == "http://a/v1"
    print("PASS api target routing")


def test_parse_fit_reply_keep_allok():
    ok_json = json.dumps({"x_minus": "ok", "x_plus": "ok", "yaw_dir": "ok"})
    nom_json = json.dumps({"x_minus": "short", "x_plus": "ok",
                            "yaw_dir": "cw"})
    # default contract unchanged: all-ok -> None (a keep)
    assert VLMJudge._parse_fit_reply(ok_json) is None
    assert VLMJudge._parse_fit_reply(nom_json)["x_minus"] == "short"
    # keep_allok distinguishes "fits" from "unparseable"
    assert VLMJudge._parse_fit_reply(ok_json, keep_allok=True) == {
        "x_minus": "ok", "x_plus": "ok", "yaw_dir": "ok"}
    assert VLMJudge._parse_fit_reply("garbage", keep_allok=True) is None
    print("PASS parse fit reply keep_allok")


def _mk_box():
    return OrientedBox(center=(0.0, 0.0, 1.0), size=(1.0, 0.6, 2.0), yaw=0.0)


def _low_quality():
    return {"front": {"score": 0.1, "visibility": 0.9, "clearance": 1.0}}


def test_adjudicate_box_escalates_on_low_quality():
    j = VLMJudge(backend="qwen", thinking_model="mt")
    box = _mk_box()
    # inject a low-quality render so the fast verdict is gate-capped
    j._render_quality[j._render_cache_key([box])] = _low_quality()
    calls = []

    def fake(scene, b, question, options, thinking=False):
        calls.append(thinking)
        if thinking:
            return Verdict(action="answer", params={"choice": "two devices"},
                           confidence=0.8, detail="thinking says two")
        return Verdict(action="answer", params={"choice": "one device"},
                       confidence=0.8, detail="fast says one")

    j._qwen_adjudicate = fake
    with tempfile.TemporaryDirectory() as td:
        j.set_record(os.path.join(td, "rec.jsonl"))
        v = j.adjudicate_box(None, box, "how many?", ["one device",
                                                      "two devices"])
        assert calls == [False, True], f"call sequence wrong: {calls}"
        assert v.params["choice"] == "two devices", "escalated verdict lost"
        assert "[thinking-escalated]" in v.detail
        assert v.confidence == 0.8, "escalated verdict must not stay capped"
        recs = [json.loads(l) for l in
                open(os.path.join(td, "rec.jsonl"), encoding="utf-8")]
        assert len(recs) == 1 and recs[0]["escalated"] is True
    print("PASS adjudicate_box escalation replaces fast verdict")


def test_adjudicate_box_escalation_failure_keeps_primary():
    j = VLMJudge(backend="qwen", thinking_model="mt")
    box = _mk_box()
    j._render_quality[j._render_cache_key([box])] = _low_quality()

    def fake(scene, b, question, options, thinking=False):
        if thinking:
            return None            # e.g. thinking endpoint unreachable
        return Verdict(action="answer", params={"choice": "one device"},
                       confidence=0.8, detail="fast says one")

    j._qwen_adjudicate = fake
    v = j.adjudicate_box(None, box, "how many?", ["one device"])
    assert v.params["choice"] == "one device"
    # fast verdict was quality-capped (0.5) and NOT escalated
    assert v.confidence == 0.5
    assert "thinking-escalated" not in (v.detail or "")
    print("PASS escalation failure keeps primary verdict")


def test_adjudicate_box_no_escalation_without_thinking_model():
    j = VLMJudge(backend="qwen")
    box = _mk_box()
    j._render_quality[j._render_cache_key([box])] = _low_quality()
    calls = []

    def fake(scene, b, question, options, thinking=False):
        calls.append(thinking)
        return Verdict(action="answer", params={"choice": "one device"},
                       confidence=0.8, detail="fast")

    j._qwen_adjudicate = fake
    v = j.adjudicate_box(None, box, "q?", ["one device"])
    assert calls == [False], "must not call the thinking tier unconfigured"
    print("PASS no escalation without thinking model")


def test_adjudicate_fit_escalated_allok_overrides():
    """A hard-evidence fit where the fast model nominates a correction but
    the thinking model carefully concludes the box fits: the escalated
    all-ok must WIN (keep), not fall through to the fast nomination."""
    import agentic_gts.agent.judge as J

    j = VLMJudge(backend="qwen", thinking_model="mt")
    box = _mk_box()
    scene = types.SimpleNamespace(points=np.zeros((5, 3)), meta={})

    nom = json.dumps({"x_minus": "short", "x_plus": "ok", "yaw_dir": "ok"})
    allok = json.dumps({"x_minus": "ok", "x_plus": "ok", "yaw_dir": "ok"})

    def fake_render(points, boxes, gs_ply=None, overlay=None, quality_out=None):
        if quality_out is not None:
            quality_out.update(_low_quality())
        return np.zeros((8, 8, 3), dtype=np.float32)

    def fake_call(png, prompt, max_tokens=256, thinking=False):
        assert max_tokens == 2048 if thinking else True
        return allok if thinking else nom

    orig_render, orig_call = J.render_topdown_image, j._qwen_image_call
    J.render_topdown_image = fake_render
    j._qwen_image_call = fake_call
    try:
        v = j.adjudicate_fit(scene, box)
        assert v.action == "keep", (
            f"escalated all-ok must keep, got {v.action} {v.params}")
        assert "[thinking-escalated]" in v.detail
    finally:
        J.render_topdown_image = orig_render
        j._qwen_image_call = orig_call
    print("PASS adjudicate_fit escalated all-ok overrides fast nomination")


def test_adjudicate_fit_escalated_nomination_wins():
    import agentic_gts.agent.judge as J

    j = VLMJudge(backend="qwen", thinking_model="mt")
    box = _mk_box()
    scene = types.SimpleNamespace(points=np.zeros((5, 3)), meta={})

    nom_fast = json.dumps({"x_minus": "ok", "x_plus": "ok", "yaw_dir": "ok"})
    nom_think = json.dumps({"x_minus": "ok", "x_plus": "short",
                            "yaw_dir": "cw"})

    def fake_render(points, boxes, gs_ply=None, overlay=None, quality_out=None):
        if quality_out is not None:
            quality_out.update(_low_quality())
        return np.zeros((8, 8, 3), dtype=np.float32)

    def fake_call(png, prompt, max_tokens=256, thinking=False):
        return nom_think if thinking else nom_fast

    orig_render = J.render_topdown_image
    J.render_topdown_image = fake_render
    j._qwen_image_call = fake_call
    try:
        v = j.adjudicate_fit(scene, box)
        assert v.action == "refine"
        assert v.params == {"x_minus": "ok", "x_plus": "short",
                           "yaw_dir": "cw"}
        # escalated verdict must NOT be quality-capped back to 0.5
        assert v.confidence == 0.8
        assert "[thinking-escalated]" in v.detail
    finally:
        J.render_topdown_image = orig_render
    print("PASS adjudicate_fit escalated nomination wins uncapped")


if __name__ == "__main__":
    test_strip_think_variants()
    test_should_escalate_gating()
    test_api_target_routes()
    test_parse_fit_reply_keep_allok()
    test_adjudicate_box_escalates_on_low_quality()
    test_adjudicate_box_escalation_failure_keeps_primary()
    test_adjudicate_box_no_escalation_without_thinking_model()
    test_adjudicate_fit_escalated_allok_overrides()
    test_adjudicate_fit_escalated_nomination_wins()
    print("ALL THINKING-ESCALATION TESTS PASSED")
