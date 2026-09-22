"""Tests for the consolidated torch-guard CLI."""
from __future__ import annotations

import json

from torch_correctness_guards import __version__
from torch_correctness_guards.cli import main
from torch_correctness_guards.guards import (
    addcdiv_stale_scalar,
    as_strided_restride_oob,
    checkpoint_noise,
    compile_validation,
    dynamic_clamp,
    normal_dtype_promotion,
    shuffle_sample_frozen,
    std_precision,
    transpose_argmin,
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
    assert "compile-validation" in out
    assert "input validation" in out
    assert "dynamic-clamp" in out
    assert "torch.clamp" in out
    assert "normal-dtype-promotion" in out
    assert "Normal.sample" in out
    assert "shuffle-sample-frozen" in out
    assert "random.shuffle" in out
    assert "std-precision" in out
    assert "std/var" in out
    assert "transpose-argmin" in out
    assert "argmin/argmax" in out


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


def _fake_compile_validation_case(name, is_boundary, bypassed, guard_ok):
    if is_boundary:
        eager_raised = False
        compiled_raised = False
        guarded_raised = False if guard_ok else True
    else:
        eager_raised = True
        compiled_raised = not bypassed
        guarded_raised = eager_raised if guard_ok else compiled_raised
    return {
        "name": name,
        "issue_url": "https://github.com/pytorch/pytorch/issues/185246",
        "is_boundary_case": is_boundary,
        "eager_raised": eager_raised,
        "eager_error": "RuntimeError: fake" if eager_raised else None,
        "compiled_raised": compiled_raised,
        "compiled_error": "RuntimeError: fake" if compiled_raised else None,
        "guarded_raised": guarded_raised,
        "guarded_error": "RuntimeError: fake" if guarded_raised else None,
        "validation_bypassed": bypassed,
        "guard_restores_validation": guarded_raised == eager_raised,
    }


def _fake_compile_validation_report(**overrides):
    cases = [
        _fake_compile_validation_case("fake_invalid", False, True, True),
        _fake_compile_validation_case("fake_boundary", True, False, True),
    ]
    report = {
        "torch_version": "9.9.9-fake",
        "issue_urls": ["https://github.com/pytorch/pytorch/issues/185246"],
        "cases": cases,
        "any_validation_bypassed": True,
        "guard_fully_correct": True,
        "boundary_cases_not_spuriously_flagged": True,
    }
    report.update(overrides)
    return report


def test_run_compile_validation_json_includes_guard_name(monkeypatch, capsys):
    monkeypatch.setattr(compile_validation, "diagnose", lambda: _fake_compile_validation_report())
    assert main(["run", "compile-validation", "--json"]) == 0
    payload = json.loads(capsys.readouterr().out)
    assert payload["guard"] == "compile-validation"
    assert payload["guard_fully_correct"] is True


def test_run_compile_validation_text_reports_guard_status(monkeypatch, capsys):
    monkeypatch.setattr(compile_validation, "diagnose", lambda: _fake_compile_validation_report())
    assert main(["run", "compile-validation", "--no-color"]) == 0
    out = capsys.readouterr().out
    assert "input-validation bypass reproduced" in out
    assert "guard wrappers restore eager" in out
    assert "BYPASSED" in out


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


def _fake_shuffle_sample_frozen_report(**overrides):
    report = {
        "torch_version": "9.9.9-fake",
        "issue_url": "https://github.com/pytorch/pytorch/issues/197085",
        "seed": 7,
        "calls": 3,
        "cases": [
            {
                "kind": "shuffle",
                "description": "fake shuffle case",
                "eager_sequence": [[1, 2], [2, 1]],
                "native_compiled_sequence": [[1, 2], [1, 2]],
                "guarded_compiled_sequence": [[1, 2], [2, 1]],
                "eager_shows_real_variation": True,
                "native_frozen_after_first_call": True,
                "guard_matches_eager": True,
                "guard_correct": True,
            }
        ],
        "any_native_frozen": True,
        "guard_fully_correct": True,
    }
    report.update(overrides)
    return report


def test_run_shuffle_sample_frozen_json_includes_guard_name(monkeypatch, capsys):
    monkeypatch.setattr(shuffle_sample_frozen, "diagnose", lambda: _fake_shuffle_sample_frozen_report())
    assert main(["run", "shuffle-sample-frozen", "--json"]) == 0
    payload = json.loads(capsys.readouterr().out)
    assert payload["guard"] == "shuffle-sample-frozen"
    assert payload["guard_fully_correct"] is True


def test_run_shuffle_sample_frozen_text_reports_guard_status(monkeypatch, capsys):
    monkeypatch.setattr(shuffle_sample_frozen, "diagnose", lambda: _fake_shuffle_sample_frozen_report())
    assert main(["run", "shuffle-sample-frozen", "--no-color"]) == 0
    out = capsys.readouterr().out
    assert "random.shuffle/random.sample frozen at trace time reproduced" in out
    assert "safe_shuffle()/safe_sample() restore eager" in out
    assert "FROZEN" in out


def _fake_std_precision_report(**overrides):
    report = {
        "torch_version": "9.9.9-fake",
        "issue_urls": ["https://github.com/pytorch/pytorch/issues/197089"],
        "std_cases": [
            {
                "op": "std",
                "magnitude_exponent": 30,
                "eager_value": 1.0,
                "compiled_value": float("inf"),
                "eager_is_finite": True,
                "compiled_is_finite": False,
                "diverges": True,
                "guard_compiled_value": 1.0,
                "guard_matches_eager": True,
            }
        ],
        "var_cases": [],
        "zero_gradient_cases": [
            {
                "magnitude_exponent": -30,
                "eager_output": 1e-30,
                "eager_grad_is_zero": False,
                "compiled_output": 0.0,
                "compiled_grad_is_zero": True,
                "silent_zero_gradient_bug": True,
                "guard_compiled_output": 1e-30,
                "guard_grad_is_zero": False,
                "guard_matches_eager": True,
            }
        ],
        "any_std_divergence": True,
        "any_silent_zero_gradient": True,
        "guard_fully_correct": True,
    }
    report.update(overrides)
    return report


def test_run_std_precision_json_includes_guard_name(monkeypatch, capsys):
    monkeypatch.setattr(std_precision, "diagnose", lambda: _fake_std_precision_report())
    assert main(["run", "std-precision", "--json"]) == 0
    payload = json.loads(capsys.readouterr().out)
    assert payload["guard"] == "std-precision"
    assert payload["guard_fully_correct"] is True


def test_run_std_precision_text_reports_guard_status(monkeypatch, capsys):
    monkeypatch.setattr(std_precision, "diagnose", lambda: _fake_std_precision_report())
    assert main(["run", "std-precision", "--no-color"]) == 0
    out = capsys.readouterr().out
    assert "std/var precision divergence reproduced" in out
    assert "silent all-zero std gradient reproduced" in out
    assert "safe_std()/safe_var()/safe_var_mean()/safe_std_mean() match eager" in out


def _fake_transpose_argmin_report(**overrides):
    report = {
        "torch_version": "9.9.9-fake",
        "issue_urls": ["https://github.com/pytorch/pytorch/issues/197739"],
        "cases": [
            {
                "op": "add",
                "mode": "argmin",
                "shape": [2, 2],
                "seed": 0,
                "eager_value": 1,
                "native_compiled_value": 2,
                "native_diverges": True,
                "guard_matches_eager": True,
            }
        ],
        "any_transpose_argreduce_divergence": True,
        "guard_fully_correct": True,
    }
    report.update(overrides)
    return report


def test_run_transpose_argmin_json_includes_guard_name(monkeypatch, capsys):
    monkeypatch.setattr(transpose_argmin, "diagnose", lambda: _fake_transpose_argmin_report())
    assert main(["run", "transpose-argmin", "--json"]) == 0
    payload = json.loads(capsys.readouterr().out)
    assert payload["guard"] == "transpose-argmin"
    assert payload["guard_fully_correct"] is True


def test_run_transpose_argmin_text_reports_guard_status(monkeypatch, capsys):
    monkeypatch.setattr(transpose_argmin, "diagnose", lambda: _fake_transpose_argmin_report())
    assert main(["run", "transpose-argmin", "--no-color"]) == 0
    out = capsys.readouterr().out
    assert "transpose+op+argmin/argmax index divergence reproduced" in out
    assert "safe_reduce_index() matches eager" in out
    assert "DIVERGES" in out
