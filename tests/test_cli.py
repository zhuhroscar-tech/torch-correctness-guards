"""Tests for the consolidated torch-guard CLI."""
from __future__ import annotations

import json

from torch_correctness_guards import __version__
from torch_correctness_guards.cli import main
from torch_correctness_guards.guards import addcdiv_stale_scalar


def _fake_report(**overrides):
    report = {
        "torch_version": "9.9.9-fake",
        "issue_url": "https://github.com/pytorch/pytorch/issues/185382",
        "upstream_fix_pr": "https://github.com/pytorch/pytorch/pull/195040",
        "cases": [
            {
                "iterations": 6,
                "eager_final_param": [0.0, 0.0, 0.0, 0.0],
                "native_final_param": [0.1, 0.0, 0.0, 0.0],
                "guarded_final_param": [0.0, 0.0, 0.0, 0.0],
                "native_diverges": True,
                "guard_matches_eager": True,
                "max_abs_diff_native": 0.1,
                "max_abs_diff_guarded": 0.0,
            }
        ],
        "any_native_diverges": True,
        "guard_fully_correct": True,
    }
    report.update(overrides)
    return report


def test_version_flag(capsys):
    assert main(["--version"]) == 0
    assert f"torch-correctness-guards {__version__}" in capsys.readouterr().out


def test_list_includes_migrated_guard(capsys):
    assert main(["list"]) == 0
    out = capsys.readouterr().out
    assert "addcdiv-stale-scalar" in out
    assert "addcdiv_" in out


def test_run_json_includes_guard_name(monkeypatch, capsys):
    monkeypatch.setattr(addcdiv_stale_scalar, "diagnose", lambda: _fake_report())
    assert main(["run", "addcdiv-stale-scalar", "--json"]) == 0
    payload = json.loads(capsys.readouterr().out)
    assert payload["guard"] == "addcdiv-stale-scalar"
    assert payload["guard_fully_correct"] is True


def test_run_text_reports_guard_status(monkeypatch, capsys):
    monkeypatch.setattr(addcdiv_stale_scalar, "diagnose", lambda: _fake_report())
    assert main(["run", "addcdiv-stale-scalar", "--no-color"]) == 0
    out = capsys.readouterr().out
    assert "stale-scalar addcdiv_/addcmul_ divergence reproduced" in out
    assert "guard matches eager" in out
    assert "adam-step cases" in out


def test_guard_failure_sets_exit_1(monkeypatch, capsys):
    monkeypatch.setattr(addcdiv_stale_scalar, "diagnose", lambda: _fake_report(guard_fully_correct=False))
    assert main(["run", "addcdiv-stale-scalar", "--json"]) == 1
    assert json.loads(capsys.readouterr().out)["guard_fully_correct"] is False


def test_torch_unavailable_sets_exit_2(monkeypatch, capsys):
    def _raise():
        raise addcdiv_stale_scalar.TorchUnavailableError("torch missing")

    monkeypatch.setattr(addcdiv_stale_scalar, "diagnose", _raise)
    assert main(["run", "addcdiv-stale-scalar", "--json"]) == 2
    assert json.loads(capsys.readouterr().out) == {
        "guard": "addcdiv-stale-scalar",
        "error": "torch missing",
    }
