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
    cpu_backward_nan_tail,
    dynamo_closure_descriptor,
    dtype_view_scatter,
    duplicate_index_writeorder,
    dynamic_clamp,
    equality_fusion,
    expand_fill,
    full_dtype,
    inplace_slice_shift_aliasing,
    int64_index_truncation,
    linalg_pinv_complex_grad,
    memory_budget_rng,
    mps_copy_dtype,
    mps_linalg_stride,
    multioutput_alias,
    normal_dtype_promotion,
    shuffle_sample_frozen,
    softmax_dim,
    std_precision,
    tiled_reduction_tail_store,
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
    assert "cpu-backward-nan-tail" in out
    assert "NaN gradients" in out
    assert "dynamo-closure-descriptor" in out
    assert "closure-captured Tensor descriptors" in out
    assert "dtype-view-scatter" in out
    assert "diagonal_scatter" in out
    assert "duplicate-index-writeorder" in out
    assert "duplicate-index" in out
    assert "dynamic-clamp" in out
    assert "torch.clamp" in out
    assert "equality-fusion" in out
    assert "division fused into equality" in out
    assert "expand-fill" in out
    assert "Tensor.expand" in out
    assert "full-dtype" in out
    assert "torch.full" in out
    assert "inplace-slice-shift-aliasing" in out
    assert "slice-shift" in out
    assert "int64-index-truncation" in out
    assert "int64 arange-multiply" in out
    assert "linalg-pinv-complex-grad" in out
    assert "complex torch.linalg.pinv gradients" in out
    assert "memory-budget-rng" in out
    assert "activation_memory_budget" in out
    assert "mps-copy-dtype" in out
    assert "MPS tensor copies" in out
    assert "mps-linalg-stride" in out
    assert "row-major layouts" in out
    assert "multioutput-alias" in out
    assert "out= tuples" in out
    assert "normal-dtype-promotion" in out
    assert "Normal.sample" in out
    assert "shuffle-sample-frozen" in out
    assert "random.shuffle" in out
    assert "softmax-dim" in out
    assert "softmax dim" in out
    assert "std-precision" in out
    assert "std/var" in out
    assert "tiled-reduction-tail-store" in out
    assert "2D-tiled reduction tail store" in out
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


def _fake_linalg_pinv_complex_grad_report(**overrides):
    report = {
        "torch_version": "9.9.9-fake",
        "issue_url": "https://github.com/pytorch/pytorch/issues/197084",
        "cases": [
            {
                "shape": (4, 3),
                "seed": 0,
                "eager_matches_dynamo_only": True,
                "aot_eager_diverges": True,
                "inductor_diverges": True,
                "guarded_matches_eager": True,
                "max_abs_diff_aot_eager": 5.5,
                "max_abs_diff_inductor": 5.5,
                "max_abs_diff_guarded": 1e-6,
            }
        ],
        "any_native_bug": True,
        "guard_fully_correct": True,
    }
    report.update(overrides)
    return report


def test_run_linalg_pinv_complex_grad_json_includes_guard_name(monkeypatch, capsys):
    monkeypatch.setattr(linalg_pinv_complex_grad, "diagnose", lambda: _fake_linalg_pinv_complex_grad_report())
    assert main(["run", "linalg-pinv-complex-grad", "--json"]) == 0
    payload = json.loads(capsys.readouterr().out)
    assert payload["guard"] == "linalg-pinv-complex-grad"
    assert payload["guard_fully_correct"] is True


def test_run_linalg_pinv_complex_grad_text_reports_guard_status(monkeypatch, capsys):
    monkeypatch.setattr(linalg_pinv_complex_grad, "diagnose", lambda: _fake_linalg_pinv_complex_grad_report())
    assert main(["run", "linalg-pinv-complex-grad", "--no-color"]) == 0
    out = capsys.readouterr().out
    assert "complex pinv/matrix_sqrth wrong-gradient bug reproduced" in out
    assert "safe_complex_pinv_grad() restores eager's correct gradient" in out
    assert "aot_eager_diff=" in out


def _fake_memory_budget_rng_report(**overrides):
    report = {
        "torch_version": "9.9.9-fake",
        "issue_url": "https://github.com/pytorch/pytorch/issues/190758",
        "unguarded_cases": [
            {
                "budget": 0.0,
                "x_grad": [[10.0]],
                "expected_grad_from_forward_mask": [[14.0]],
                "matches": False,
            }
        ],
        "guarded_cases": [
            {
                "budget": 0.0,
                "x_grad": [[14.0]],
                "expected_grad_from_forward_mask": [[14.0]],
                "matches": True,
            }
        ],
        "any_unguarded_rng_recompute_bug": True,
        "guard_fully_correct": True,
    }
    report.update(overrides)
    return report


def test_run_memory_budget_rng_json_includes_guard_name(monkeypatch, capsys):
    monkeypatch.setattr(memory_budget_rng, "diagnose", lambda: _fake_memory_budget_rng_report())
    assert main(["run", "memory-budget-rng", "--json"]) == 0
    payload = json.loads(capsys.readouterr().out)
    assert payload["guard"] == "memory-budget-rng"
    assert payload["guard_fully_correct"] is True


def test_run_memory_budget_rng_text_reports_guard_status(monkeypatch, capsys):
    monkeypatch.setattr(memory_budget_rng, "diagnose", lambda: _fake_memory_budget_rng_report())
    assert main(["run", "memory-budget-rng", "--no-color"]) == 0
    out = capsys.readouterr().out
    assert "activation_memory_budget RNG-recompute gradient bug reproduced" in out
    assert "safe_compile() prevents the divergence" in out
    assert "WRONG-GRADIENT" in out


def _fake_mps_copy_dtype_report(**overrides):
    report = {
        "torch_version": "9.9.9-fake",
        "issue_url": "https://github.com/pytorch/pytorch/issues/197715",
        "mps_functional": True,
        "cases": [
            {
                "dtype_name": "float64",
                "ran": True,
                "skip_reason": None,
                "native_to_row": [0.0],
                "guard_to_row": [1.0],
                "native_copy_row": [9.0],
                "guard_copy_row": [1.0],
                "oracle_row": [1.0],
                "native_to_matches_oracle": False,
                "guard_to_matches_oracle": True,
                "native_copy_matches_oracle": False,
                "guard_copy_matches_oracle": True,
            },
            {
                "dtype_name": "complex128",
                "ran": False,
                "skip_reason": "MPS not available on this host",
                "native_to_row": None,
                "guard_to_row": None,
                "native_copy_row": None,
                "guard_copy_row": None,
                "oracle_row": None,
                "native_to_matches_oracle": None,
                "guard_to_matches_oracle": None,
                "native_copy_matches_oracle": None,
                "guard_copy_matches_oracle": None,
            },
        ],
        "any_native_silently_wrong": True,
        "guard_fully_correct": True,
    }
    report.update(overrides)
    return report


def test_run_mps_copy_dtype_json_includes_guard_name(monkeypatch, capsys):
    monkeypatch.setattr(mps_copy_dtype, "diagnose", lambda: _fake_mps_copy_dtype_report())
    assert main(["run", "mps-copy-dtype", "--json"]) == 0
    payload = json.loads(capsys.readouterr().out)
    assert payload["guard"] == "mps-copy-dtype"
    assert payload["guard_fully_correct"] is True


def test_run_mps_copy_dtype_text_reports_guard_status(monkeypatch, capsys):
    monkeypatch.setattr(mps_copy_dtype, "diagnose", lambda: _fake_mps_copy_dtype_report())
    assert main(["run", "mps-copy-dtype", "--no-color"]) == 0
    out = capsys.readouterr().out
    assert "MPS-to-CPU float64/complex128 silent copy bug reproduced" in out
    assert "safe_to()/safe_copy_() match the CPU-oracle" in out
    assert "SILENT-WRONG" in out
    assert "skipped: MPS not available on this host" in out


def _fake_mps_linalg_stride_report(**overrides):
    report = {
        "torch_version": "9.9.9-fake",
        "issue_url": "https://github.com/pytorch/pytorch/issues/197236",
        "mps_functional": True,
        "cases": [
            {
                "op_name": "solve_triangular",
                "ran": True,
                "skip_reason": None,
                "cpu_stride": [1, 5],
                "native_mps_stride": [5, 1],
                "guard_mps_stride": [1, 5],
                "native_layout_matches_cpu": False,
                "guard_layout_matches_cpu": True,
                "native_values_match_cpu": True,
                "guard_values_match_cpu": True,
            }
        ],
        "any_native_layout_mismatch": True,
        "guard_fully_correct": True,
    }
    report.update(overrides)
    return report


def test_run_mps_linalg_stride_json_includes_guard_name(monkeypatch, capsys):
    monkeypatch.setattr(mps_linalg_stride, "diagnose", lambda: _fake_mps_linalg_stride_report())
    assert main(["run", "mps-linalg-stride", "--json"]) == 0
    payload = json.loads(capsys.readouterr().out)
    assert payload["guard"] == "mps-linalg-stride"
    assert payload["guard_fully_correct"] is True


def test_run_mps_linalg_stride_text_reports_guard_status(monkeypatch, capsys):
    monkeypatch.setattr(mps_linalg_stride, "diagnose", lambda: _fake_mps_linalg_stride_report())
    assert main(["run", "mps-linalg-stride", "--no-color"]) == 0
    out = capsys.readouterr().out
    assert "MPS row-major vs CPU column-major layout mismatch reproduced" in out
    assert "safe_solve_triangular()/safe_cholesky_solve()/safe_solve()" in out
    assert "LAYOUT-MISMATCH" in out


def _fake_multioutput_alias_report(**overrides):
    report = {
        "torch_version": "9.9.9-fake",
        "issue_urls": ["https://github.com/pytorch/pytorch/issues/195338"],
        "aminmax_cases": [
            {
                "values": [3.0, 1.0],
                "native_aliased_result": 3.0,
                "native_aliased_raised": False,
                "reference_min": 1.0,
                "reference_max": 3.0,
                "native_result_is_wrong": True,
                "guard_raised": True,
            }
        ],
        "slogdet_cases": [
            {
                "matrix": [[2.0, 0.0], [0.0, 2.0]],
                "native_aliased_result": 1.3862943611198906,
                "native_aliased_raised": False,
                "reference_sign": 1.0,
                "reference_logabsdet": 1.3862943611198906,
                "native_result_lost_sign": True,
                "guard_raised": True,
            }
        ],
        "any_aminmax_alias_bug": True,
        "any_slogdet_alias_bug": True,
        "guard_fully_correct": True,
    }
    report.update(overrides)
    return report


def test_run_multioutput_alias_json_includes_guard_name(monkeypatch, capsys):
    monkeypatch.setattr(multioutput_alias, "diagnose", lambda: _fake_multioutput_alias_report())
    assert main(["run", "multioutput-alias", "--json"]) == 0
    payload = json.loads(capsys.readouterr().out)
    assert payload["guard"] == "multioutput-alias"
    assert payload["guard_fully_correct"] is True


def test_run_multioutput_alias_text_reports_guard_status(monkeypatch, capsys):
    monkeypatch.setattr(multioutput_alias, "diagnose", lambda: _fake_multioutput_alias_report())
    assert main(["run", "multioutput-alias", "--no-color"]) == 0
    out = capsys.readouterr().out
    assert "torch.aminmax(out=(t, t)) same-tensor aliasing bug reproduced" in out
    assert "torch.linalg.slogdet(out=(t, t)) same-tensor aliasing bug reproduced" in out
    assert "safe_aminmax()/safe_slogdet() raise before computing" in out
    assert "LOST-SIGN" in out



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


def _fake_duplicate_index_writeorder_report(**overrides):
    report = {
        "torch_version": "9.9.9-fake",
        "issue_url": "https://github.com/pytorch/pytorch/issues/197582",
        "cases": [
            {
                "kind": "computed_duplicate_index",
                "description": "fake case",
                "eager_result": [2.0, 3.0, 3.0, 4.0],
                "native_compiled_result": [3.0, 4.0, 3.0, 4.0],
                "guarded_compiled_result": [2.0, 3.0, 3.0, 4.0],
                "native_matches_eager": False,
                "guard_matches_eager": True,
                "guard_correct": True,
            }
        ],
        "bug_reproduced": True,
        "isolation_confirmed": True,
        "guard_fully_correct": True,
    }
    report.update(overrides)
    return report


def test_run_duplicate_index_writeorder_json_includes_guard_name(monkeypatch, capsys):
    monkeypatch.setattr(duplicate_index_writeorder, "diagnose", lambda: _fake_duplicate_index_writeorder_report())
    assert main(["run", "duplicate-index-writeorder", "--json"]) == 0
    payload = json.loads(capsys.readouterr().out)
    assert payload["guard"] == "duplicate-index-writeorder"
    assert payload["guard_fully_correct"] is True


def test_run_duplicate_index_writeorder_text_reports_guard_status(monkeypatch, capsys):
    monkeypatch.setattr(duplicate_index_writeorder, "diagnose", lambda: _fake_duplicate_index_writeorder_report())
    assert main(["run", "duplicate-index-writeorder", "--no-color"]) == 0
    out = capsys.readouterr().out
    assert "computed duplicate-index write-order miscompilation reproduced" in out
    assert "safe_dup_index_assign() restores eager write-order semantics" in out
    assert "computed_duplicate_index" in out


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


def _fake_equality_fusion_report(**overrides):
    report = {
        "torch_version": "9.9.9-fake",
        "issue_url": "https://github.com/pytorch/pytorch/issues/195214",
        "cases": [
            {
                "description": "fake bf16 case",
                "dtype": "torch.bfloat16",
                "eager_tie_counts": [1],
                "inductor_default_tie_counts": [0],
                "inductor_default_matches_eager": False,
                "inductor_default_any_nonfinite_downstream": True,
                "inductor_emulate_precision_tie_counts": [1],
                "inductor_emulate_precision_matches_eager": True,
                "guarded_tie_counts": [1],
                "guarded_matches_eager": True,
            }
        ],
        "any_default_inductor_diverges_on_bf16": True,
        "guard_fully_correct": True,
        "emulate_precision_casts_fully_correct": True,
    }
    report.update(overrides)
    return report


def test_run_equality_fusion_json_includes_guard_name(monkeypatch, capsys):
    monkeypatch.setattr(equality_fusion, "diagnose", lambda: _fake_equality_fusion_report())
    assert main(["run", "equality-fusion", "--json"]) == 0
    payload = json.loads(capsys.readouterr().out)
    assert payload["guard"] == "equality-fusion"
    assert payload["guard_fully_correct"] is True


def test_run_equality_fusion_text_reports_guard_status(monkeypatch, capsys):
    monkeypatch.setattr(equality_fusion, "diagnose", lambda: _fake_equality_fusion_report())
    assert main(["run", "equality-fusion", "--no-color"]) == 0
    out = capsys.readouterr().out
    assert "division/equality fusion divergence reproduced" in out
    assert "precision_safe_division_compare() matches eager" in out
    assert "DIVERGES" in out


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


def _fake_cpu_backward_nan_tail_report(**overrides):
    report = {
        "torch_version": "9.9.9-fake",
        "issue_urls": ["https://github.com/pytorch/pytorch/issues/195075"],
        "cases": [
            {
                "op": "hardtanh_backward",
                "length": 9,
                "buggy_grad_first": 0.0,
                "buggy_grad_last": 1.0,
                "buggy_position_dependent": True,
                "guard_grad_first": 1.0,
                "guard_grad_last": 1.0,
                "guard_consistent": True,
            }
        ],
        "any_bug_present": True,
        "guard_fully_effective": True,
        "guard_fully_correct": True,
    }
    report.update(overrides)
    return report


def test_run_cpu_backward_nan_tail_json_includes_guard_name(monkeypatch, capsys):
    monkeypatch.setattr(cpu_backward_nan_tail, "diagnose", lambda: _fake_cpu_backward_nan_tail_report())
    assert main(["run", "cpu-backward-nan-tail", "--json"]) == 0
    payload = json.loads(capsys.readouterr().out)
    assert payload["guard"] == "cpu-backward-nan-tail"
    assert payload["guard_fully_effective"] is True


def test_run_cpu_backward_nan_tail_text_reports_guard_status(monkeypatch, capsys):
    monkeypatch.setattr(cpu_backward_nan_tail, "diagnose", lambda: _fake_cpu_backward_nan_tail_report())
    assert main(["run", "cpu-backward-nan-tail", "--no-color"]) == 0
    out = capsys.readouterr().out
    assert "length-dependent NaN-gradient divergence reproduced" in out
    assert "every safe_* backward guard is length-independent" in out
    assert "LENGTH-DEP-BUG" in out


def _fake_dynamo_closure_descriptor_report(**overrides):
    report = {
        "torch_version": "9.9.9-fake",
        "issue_urls": ["https://github.com/pytorch/pytorch/issues/197811"],
        "cases": [
            {
                "op_name": "__mul__",
                "a": 6.0,
                "b": 3.0,
                "eager_result": 18.0,
                "compiled_result": 9.0,
                "guard_result": 18.0,
                "stale_reuse_bug": True,
                "guard_matches_eager": True,
            }
        ],
        "any_stale_reuse_bug": True,
        "guard_fully_correct": True,
    }
    report.update(overrides)
    return report


def test_run_dynamo_closure_descriptor_json_includes_guard_name(monkeypatch, capsys):
    monkeypatch.setattr(dynamo_closure_descriptor, "diagnose", lambda: _fake_dynamo_closure_descriptor_report())
    assert main(["run", "dynamo-closure-descriptor", "--json"]) == 0
    payload = json.loads(capsys.readouterr().out)
    assert payload["guard"] == "dynamo-closure-descriptor"
    assert payload["guard_fully_correct"] is True


def test_run_dynamo_closure_descriptor_text_reports_guard_status(monkeypatch, capsys):
    monkeypatch.setattr(dynamo_closure_descriptor, "diagnose", lambda: _fake_dynamo_closure_descriptor_report())
    assert main(["run", "dynamo-closure-descriptor", "--no-color"]) == 0
    out = capsys.readouterr().out
    assert "closure-descriptor graph-reuse divergence reproduced" in out
    assert "safe_call() matches eager" in out
    assert "STALE-REUSE" in out


def _fake_dtype_view_scatter_report(**overrides):
    report = {
        "torch_version": "9.9.9-fake",
        "issue_url": "https://github.com/pytorch/pytorch/issues/197408",
        "related_issue_url": "https://github.com/pytorch/pytorch/issues/195451",
        "cases": [
            {
                "cache0": [0, 0, 0, 0],
                "data": [1.0, 2.0, 3.0, 4.0],
                "diag": [-1.0, -1.0],
                "eager_aliases_input": False,
                "compiled_aliases_input": True,
                "eager_values": [1.0, -1.0, 3.0, -1.0],
                "compiled_values": [1.0, float("nan"), 3.0, float("nan")],
                "compiled_values_match_eager": False,
                "compiled_input_corrupted_after_output_mutation": True,
                "guarded_values_match_eager": True,
                "guarded_aliases_input_matches_eager": True,
                "guarded_input_matches_eager_after_output_mutation": True,
            }
        ],
        "any_native_value_bug": True,
        "any_native_alias_bug": True,
        "any_native_corruption": True,
        "guard_fully_correct": True,
    }
    report.update(overrides)
    return report


def test_run_dtype_view_scatter_json_includes_guard_name(monkeypatch, capsys):
    monkeypatch.setattr(dtype_view_scatter, "diagnose", lambda: _fake_dtype_view_scatter_report())
    assert main(["run", "dtype-view-scatter", "--json"]) == 0
    payload = json.loads(capsys.readouterr().out)
    assert payload["guard"] == "dtype-view-scatter"
    assert payload["guard_fully_correct"] is True


def test_run_dtype_view_scatter_text_reports_guard_status(monkeypatch, capsys):
    monkeypatch.setattr(dtype_view_scatter, "diagnose", lambda: _fake_dtype_view_scatter_report())
    assert main(["run", "dtype-view-scatter", "--no-color"]) == 0
    out = capsys.readouterr().out
    assert "dtype-view/diagonal_scatter value+aliasing bug reproduced" in out
    assert "safe_compiled_dtype_view_diagonal_scatter() restores eager" in out
    assert "WRONG-VALUES+ALIASED" in out


def _fake_expand_fill_report(**overrides):
    report = {
        "torch_version": "9.9.9-fake",
        "issue_urls": ["https://github.com/pytorch/pytorch/issues/197448"],
        "cases": [
            {
                "expand_rows": 3,
                "base_values": [1.0, 2.0],
                "fill_value": 0.0,
                "eager_result": [0.0, 0.0],
                "compiled_native_result": [9.0, 9.0],
                "compiled_guarded_result": [0.0, 0.0],
                "native_diverges": True,
                "guard_matches_eager": True,
            }
        ],
        "any_expand_fill_divergence": True,
        "guard_fully_correct": True,
    }
    report.update(overrides)
    return report


def test_run_expand_fill_json_includes_guard_name(monkeypatch, capsys):
    monkeypatch.setattr(expand_fill, "diagnose", lambda: _fake_expand_fill_report())
    assert main(["run", "expand-fill", "--json"]) == 0
    payload = json.loads(capsys.readouterr().out)
    assert payload["guard"] == "expand-fill"
    assert payload["guard_fully_correct"] is True


def test_run_expand_fill_text_reports_guard_status(monkeypatch, capsys):
    monkeypatch.setattr(expand_fill, "diagnose", lambda: _fake_expand_fill_report())
    assert main(["run", "expand-fill", "--no-color"]) == 0
    out = capsys.readouterr().out
    assert "expand()+fill_() divergence reproduced" in out
    assert "safe_fill_() matches eager" in out
    assert "DIVERGES" in out


def _fake_full_dtype_report(**overrides):
    report = {
        "torch_version": "9.9.9-fake",
        "issue_urls": ["https://github.com/pytorch/pytorch/issues/194062"],
        "bool_fill_cases": [
            {
                "fill_value": 3,
                "eager_result": 2,
                "compiled_native_result": 6,
                "compiled_guarded_result": 2,
                "native_diverges": True,
                "guard_matches_eager": True,
            }
        ],
        "int8_overflow_cases": [
            {
                "fill_value": 300,
                "eager_raised": True,
                "compiled_native_raised": False,
                "compiled_native_silent_value": 44,
                "compiled_guarded_raised": True,
                "native_silently_wrong": True,
                "guard_matches_eager": True,
                "overflow_dtype": "int8",
            }
        ],
        "extra_overflow_dtype_cases": [],
        "any_bool_fill_divergence": True,
        "any_int8_silent_overflow": True,
        "any_overflow_dtype_silent_overflow": True,
        "guard_fully_correct": True,
    }
    report.update(overrides)
    return report


def test_run_full_dtype_json_includes_guard_name(monkeypatch, capsys):
    monkeypatch.setattr(full_dtype, "diagnose", lambda: _fake_full_dtype_report())
    assert main(["run", "full-dtype", "--json"]) == 0
    payload = json.loads(capsys.readouterr().out)
    assert payload["guard"] == "full-dtype"
    assert payload["guard_fully_correct"] is True


def test_run_full_dtype_text_reports_guard_status(monkeypatch, capsys):
    monkeypatch.setattr(full_dtype, "diagnose", lambda: _fake_full_dtype_report())
    assert main(["run", "full-dtype", "--no-color"]) == 0
    out = capsys.readouterr().out
    assert "bool-fill dtype-cast divergence reproduced" in out
    assert "int8 overflow check silently skipped" in out
    assert "safe_full() matches eager" in out


def _fake_inplace_slice_shift_aliasing_report(**overrides):
    report = {
        "torch_version": "9.9.9-fake",
        "issue_url": "https://github.com/pytorch/pytorch/issues/197829",
        "cases": [
            {
                "shape": [3, 16, 16],
                "shift": 1,
                "eager_correct": True,
                "compiled_mismatches": 256,
                "compiled_total": 768,
                "compiled_matches_eager": False,
                "guarded_matches_eager": True,
            }
        ],
        "any_native_bug": True,
        "guard_fully_correct": True,
    }
    report.update(overrides)
    return report


def test_run_inplace_slice_shift_aliasing_json_includes_guard_name(monkeypatch, capsys):
    monkeypatch.setattr(inplace_slice_shift_aliasing, "diagnose", lambda: _fake_inplace_slice_shift_aliasing_report())
    assert main(["run", "inplace-slice-shift-aliasing", "--json"]) == 0
    payload = json.loads(capsys.readouterr().out)
    assert payload["guard"] == "inplace-slice-shift-aliasing"
    assert payload["guard_fully_correct"] is True


def test_run_inplace_slice_shift_aliasing_text_reports_guard_status(monkeypatch, capsys):
    monkeypatch.setattr(inplace_slice_shift_aliasing, "diagnose", lambda: _fake_inplace_slice_shift_aliasing_report())
    assert main(["run", "inplace-slice-shift-aliasing", "--no-color"]) == 0
    out = capsys.readouterr().out
    assert "in-place slice-shift aliasing bug reproduced" in out
    assert "safe_slice_shift() restores eager" in out
    assert "WRONG (256/768)" in out


def _fake_int64_index_truncation_report(**overrides):
    report = {
        "torch_version": "9.9.9-fake",
        "issue_url": "https://github.com/pytorch/pytorch/issues/183901",
        "cases": [
            {
                "description": "fake overflowing arange-multiply case",
                "multiplier": 1500000000,
                "arange_end": 9,
                "eager_result": [0, 1500000000, 3000000000],
                "native_compiled_result": [0, 1500000000, -1294967296],
                "guarded_compiled_result": [0, 1500000000, 3000000000],
                "native_diverges": True,
                "guard_matches_eager": True,
            }
        ],
        "any_native_diverges": True,
        "guard_fully_correct": True,
    }
    report.update(overrides)
    return report


def test_run_int64_index_truncation_json_includes_guard_name(monkeypatch, capsys):
    monkeypatch.setattr(int64_index_truncation, "diagnose", lambda: _fake_int64_index_truncation_report())
    assert main(["run", "int64-index-truncation", "--json"]) == 0
    payload = json.loads(capsys.readouterr().out)
    assert payload["guard"] == "int64-index-truncation"
    assert payload["guard_fully_correct"] is True


def test_run_int64_index_truncation_text_reports_guard_status(monkeypatch, capsys):
    monkeypatch.setattr(int64_index_truncation, "diagnose", lambda: _fake_int64_index_truncation_report())
    assert main(["run", "int64-index-truncation", "--no-color"]) == 0
    out = capsys.readouterr().out
    assert "int64 arange-multiply truncation reproduced" in out
    assert "safe_int64_arange_mul() matches eager" in out
    assert "SILENT-WRONG" in out


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


def _fake_tiled_reduction_tail_store_report(**overrides):
    report = {
        "torch_version": "9.9.9-fake",
        "issue_url": "https://github.com/pytorch/pytorch/issues/196681",
        "fix_pr_url": "https://github.com/pytorch/pytorch/pull/196882",
        "cases": [
            {
                "size": 66,
                "aligned_to_vector_width": False,
                "max_abs_diff": 42.0,
                "bug_reproduced": True,
                "crashed_bare": False,
                "guard_raised": True,
                "crashed_guarded": False,
                "guard_behaved_correctly": True,
            }
        ],
        "any_bug_reproduced": True,
        "any_crash_observed": False,
        "guard_fully_correct": True,
    }
    report.update(overrides)
    return report


def test_run_tiled_reduction_tail_store_json_includes_guard_name(monkeypatch, capsys):
    monkeypatch.setattr(tiled_reduction_tail_store, "diagnose", lambda: _fake_tiled_reduction_tail_store_report())
    assert main(["run", "tiled-reduction-tail-store", "--json"]) == 0
    payload = json.loads(capsys.readouterr().out)
    assert payload["guard"] == "tiled-reduction-tail-store"
    assert payload["guard_fully_correct"] is True


def test_run_tiled_reduction_tail_store_text_reports_guard_status(monkeypatch, capsys):
    monkeypatch.setattr(tiled_reduction_tail_store, "diagnose", lambda: _fake_tiled_reduction_tail_store_report())
    assert main(["run", "tiled-reduction-tail-store", "--no-color"]) == 0
    out = capsys.readouterr().out
    assert "2D-tiled reduction tail-store bug reproduced" in out
    assert "guard never silently trusts" in out
    assert "BUG" in out


def _fake_softmax_dim_report(**overrides):
    report = {
        "torch_version": "9.9.9-fake",
        "issue_urls": ["https://github.com/pytorch/pytorch/issues/196468"],
        "cases": [
            {
                "dim": 0,
                "shape": [4, 5, 5, 8],
                "eager_requested_vs_compiled_native_rel_diff": 0.58,
                "eager_lastaxis_vs_compiled_native_rel_diff": 1.7e-7,
                "native_diverges": True,
                "native_matches_wrong_lastaxis": True,
                "guard_matches_eager": True,
            }
        ],
        "any_softmax_dim_divergence": True,
        "guard_fully_correct": True,
    }
    report.update(overrides)
    return report


def test_run_softmax_dim_json_includes_guard_name(monkeypatch, capsys):
    monkeypatch.setattr(softmax_dim, "diagnose", lambda: _fake_softmax_dim_report())
    assert main(["run", "softmax-dim", "--json"]) == 0
    payload = json.loads(capsys.readouterr().out)
    assert payload["guard"] == "softmax-dim"
    assert payload["guard_fully_correct"] is True


def test_run_softmax_dim_text_reports_guard_status(monkeypatch, capsys):
    monkeypatch.setattr(softmax_dim, "diagnose", lambda: _fake_softmax_dim_report())
    assert main(["run", "softmax-dim", "--no-color"]) == 0
    out = capsys.readouterr().out
    assert "softmax-dim attention rewrite divergence reproduced" in out
    assert "safe_softmax_attention() matches eager" in out
    assert "matches-wrong-lastaxis" in out
