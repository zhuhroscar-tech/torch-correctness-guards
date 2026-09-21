"""Tests for the consolidated torch-guard CLI."""
from __future__ import annotations

import json

from torch_correctness_guards import __version__
from torch_correctness_guards.cli import main
from torch_correctness_guards.guards import (
    addcdiv_stale_scalar,
    as_strided_restride_oob,
    checkpoint_noise,
    dynamic_clamp,
    normal_dtype_promotion,
)


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
    assert "checkpoint-noise" in out
    assert "F.rrelu" in out
    assert "dynamic-clamp" in out
    assert "torch.clamp" in out
    assert "normal-dtype-promotion" in out
    assert "Normal.sample" in out


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


def _fake_checkpoint_noise_report(**overrides):
    report = {
        "torch_version": "9.9.9-fake",
        "issue_urls": ["https://github.com/pytorch/pytorch/issues/193671"],
        "checkpoint_case": {
            "mode": "checkpoint",
            "buggy_diff": 1.0,
            "buggy_forward_diff": 0.0,
            "guard_diff": 0.0,
            "guard_has_nan": False,
            "no_early_stop_diff": 0.0,
        },
        "saved_hooks_case": {
            "mode": "saved_hooks",
            "buggy_diff": 2.0,
            "buggy_forward_diff": 0.0,
            "guard_diff": 0.0,
            "guard_has_nan": False,
            "no_early_stop_diff": None,
        },
        "bug_reproduced": True,
        "mechanism_confirmed": True,
        "guard_fully_correct": True,
    }
    report.update(overrides)
    return report


def test_run_checkpoint_noise_json_includes_guard_name(monkeypatch, capsys):
    monkeypatch.setattr(checkpoint_noise, "diagnose", lambda: _fake_checkpoint_noise_report())
    assert main(["run", "checkpoint-noise", "--json"]) == 0
    payload = json.loads(capsys.readouterr().out)
    assert payload["guard"] == "checkpoint-noise"
    assert payload["guard_fully_correct"] is True


def test_run_checkpoint_noise_text_reports_guard_status(monkeypatch, capsys):
    monkeypatch.setattr(checkpoint_noise, "diagnose", lambda: _fake_checkpoint_noise_report())
    assert main(["run", "checkpoint-noise", "--no-color"]) == 0
    out = capsys.readouterr().out
    assert "F.rrelu checkpoint/saved-hooks gradient corruption reproduced" in out
    assert "safe_rrelu() produces identical" in out
    assert "non-reentrant checkpoint" in out


def _fake_dynamic_clamp_report(**overrides):
    report = {
        "torch_version": "9.9.9-fake",
        "issue_urls": [
            "https://github.com/pytorch/pytorch/issues/194976",
            "https://github.com/nvidia/Megatron-LM/issues/6918",
        ],
        "cases": [
            {
                "call_index": 2,
                "shape": (1, 4),
                "requires_grad": True,
                "limit": 1.0,
                "eager_value": 3.0,
                "compiled_value": 2.0,
                "guard_value": 3.0,
                "stale_reuse_bug": True,
                "guard_matches_eager": True,
            }
        ],
        "any_stale_reuse_bug": True,
        "guard_fully_correct": True,
    }
    report.update(overrides)
    return report


def test_run_dynamic_clamp_json_includes_guard_name(monkeypatch, capsys):
    monkeypatch.setattr(dynamic_clamp, "diagnose", lambda: _fake_dynamic_clamp_report())
    assert main(["run", "dynamic-clamp", "--json"]) == 0
    payload = json.loads(capsys.readouterr().out)
    assert payload["guard"] == "dynamic-clamp"
    assert payload["guard_fully_correct"] is True


def test_run_dynamic_clamp_text_reports_guard_status(monkeypatch, capsys):
    monkeypatch.setattr(dynamic_clamp, "diagnose", lambda: _fake_dynamic_clamp_report())
    assert main(["run", "dynamic-clamp", "--no-color"]) == 0
    out = capsys.readouterr().out
    assert "stale dynamic-float clamp reuse reproduced" in out
    assert "safe_clamp() matches eager" in out
    assert "STALE-REUSE" in out


def _fake_normal_dtype_promotion_report(**overrides):
    report = {
        "torch_version": "9.9.9-fake",
        "issue_url": "https://github.com/pytorch/pytorch/issues/194547",
        "cases": [
            {
                "loc_dtype": "torch.float16",
                "scale_dtype": "torch.float32",
                "eager_dtype": "torch.float16",
                "compiled_dtype": "torch.float32",
                "dtype_diverges": True,
                "guarded_dtype": "torch.float16",
                "guarded_matches_eager": True,
            }
        ],
        "any_native_divergence": True,
        "guard_fully_correct": True,
    }
    report.update(overrides)
    return report


def test_run_normal_dtype_promotion_json_includes_guard_name(monkeypatch, capsys):
    monkeypatch.setattr(normal_dtype_promotion, "diagnose", lambda: _fake_normal_dtype_promotion_report())
    assert main(["run", "normal-dtype-promotion", "--json"]) == 0
    payload = json.loads(capsys.readouterr().out)
    assert payload["guard"] == "normal-dtype-promotion"
    assert payload["guard_fully_correct"] is True


def test_run_normal_dtype_promotion_text_reports_guard_status(monkeypatch, capsys):
    monkeypatch.setattr(normal_dtype_promotion, "diagnose", lambda: _fake_normal_dtype_promotion_report())
    assert main(["run", "normal-dtype-promotion", "--no-color"]) == 0
    out = capsys.readouterr().out
    assert "Normal.sample() dtype-promotion divergence reproduced" in out
    assert "safe_compiled_normal_sample() restores eager" in out
    assert "DIVERGES" in out
