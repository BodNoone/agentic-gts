"""CLI contract tests that do not run the production pipeline."""
from __future__ import annotations

import sys

import pytest


def test_run_parser_keeps_local_and_sam_options(monkeypatch):
    from agentic_gts import cli

    captured = {}

    def fake_run(args):
        captured.update(vars(args))

    monkeypatch.setattr(cli, "cmd_run", fake_run)
    monkeypatch.setattr(sys, "argv", [
        "agentic-gts", "run", "--point-cloud", "room.ply",
        "--vlm", "local", "--vlm-model", "local-checkpoint",
        "--sam-checkpoint", "sam.pt", "--sam-model-cfg", "sam.yaml",
        "--yaw", "15", "--out", "runs/test",
    ])

    cli.main()

    assert captured["cmd"] == "run"
    assert captured["vlm"] == "local"
    assert captured["vlm_model"] == "local-checkpoint"
    assert captured["sam_checkpoint"] == "sam.pt"
    assert captured["sam_model_cfg"] == "sam.yaml"
    assert captured["yaw"] == 15.0


@pytest.mark.parametrize("removed_args", [
    ["--gt", "gt.json"],
    ["--edge-thr", "0.05"],
    ["--vlm-base", "http://127.0.0.1:8000/v1"],
    ["--vlm", "mock"],
])
def test_run_parser_rejects_removed_or_unsupported_options(monkeypatch,
                                                            removed_args):
    from agentic_gts import cli

    monkeypatch.setattr(sys, "argv", [
        "agentic-gts", "run", "--point-cloud", "room.ply", *removed_args,
    ])
    with pytest.raises(SystemExit):
        cli.main()


@pytest.mark.parametrize("command", ["synth", "diagnose", "view"])
def test_removed_cli_commands_are_rejected(monkeypatch, command):
    from agentic_gts import cli

    monkeypatch.setattr(sys, "argv", ["agentic-gts", command])
    with pytest.raises(SystemExit):
        cli.main()
