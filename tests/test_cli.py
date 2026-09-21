"""Tests for the consolidated torch-guard CLI."""
from __future__ import annotations

import json

from torch_correctness_guards import __version__
from torch_correctness_guards.cli import main
from torch_correctness_guards.guards import addcdiv_stale_scalar, as_strided_restride_oob


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
    assert "as-strided-restride-oob" in out
    assert "as_strided" in out


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


def _fake_as_strided_report(**overrides):
    report = {
        "torch_version": "9.9.9-fake",
        "issue_urls": [
            "https://github.com/pytorch/pytorch/issues/197431",
            "https://github.com/pytorch/pytorch/issues/192226",
        ],
        "cases": [
            {
                "call_index": 0,
                "x_values": (7.0, 9.0),
                "eager_ok": True,
                "eager_value": [[7.0, 9.0], [7.0, 9.0], [7.0, 9.0]],
                "compiled_ok": False,
                "compiled_value": None,
                "compiled_error": "RuntimeError: setStorage out of bounds",
                "guard_ok": True,
                "guard_value": [[7.0, 9.0], [7.0, 9.0], [7.0, 9.0]],
                "guard_error": "",
                "divergence_bug": True,
                "guard_matches_eager": True,
            }
        ],
        "any_divergence_bug": True,
        "guard_fully_correct": True,
    }
    report.update(overrides)
    return report


def test_run_as_strided_json_includes_guard_name(monkeypatch, capsys):
    monkeypatch.setattr(as_strided_restride_oob, "diagnose", lambda: _fake_as_strided_report())
    assert main(["run", "as-strided-restride-oob", "--json"]) == 0
    payload = json.loads(capsys.readouterr().out)
    assert payload["guard"] == "as-strided-restride-oob"
    assert payload["guard_fully_correct"] is True


def test_run_as_strided_text_reports_guard_status(monkeypatch, capsys):
    monkeypatch.setattr(as_strided_restride_oob, "diagnose", lambda: _fake_as_strided_report())
    assert main(["run", "as-strided-restride-oob", "--no-color"]) == 0
    out = capsys.readouterr().out
    assert "as_strided restride divergence reproduced" in out
    assert "safe_as_strided() matches eager" in out
    assert "RuntimeError: setStorage out of bounds" in out
