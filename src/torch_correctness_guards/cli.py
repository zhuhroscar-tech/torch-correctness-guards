"""Command-line interface for consolidated PyTorch correctness guards."""
from __future__ import annotations

import argparse
import json

from .style import print_fields, resolve_style, section, status_headline

_GUARDS = {
    "addcdiv-stale-scalar": {
        "description": "Inductor stale Python scalar in addcdiv_/addcmul_ Adam-style arithmetic",
        "module": "torch_correctness_guards.guards.addcdiv_stale_scalar",
    },
    "as-strided-restride-oob": {
        "description": "Inductor as_strided storage-span miscomputation for repeated+sliced restrided views",
        "module": "torch_correctness_guards.guards.as_strided_restride_oob",
    },
    "checkpoint-noise": {
        "description": "F.rrelu mutable noise buffer captured before fill under checkpoint/saved_tensors_hooks",
        "module": "torch_correctness_guards.guards.checkpoint_noise",
    },
    "compile-validation": {
        "description": "torch.compile skips eager input validation for several out-of-domain operator calls",
        "module": "torch_correctness_guards.guards.compile_validation",
    },
    "cpu-backward-nan-tail": {
        "description": "CPU backward kernels return NaN gradients differently in SIMD vector blocks versus scalar tails",
        "module": "torch_correctness_guards.guards.cpu_backward_nan_tail",
    },
    "dynamo-closure-descriptor": {
        "description": "Dynamo stale graph reuse for closure-captured Tensor descriptors and related guard omissions",
        "module": "torch_correctness_guards.guards.dynamo_closure_descriptor",
    },
    "dtype-view-scatter": {
        "description": "Inductor dtype-view custom-op diagonal_scatter returns NaN values and aliases inputs",
        "module": "torch_correctness_guards.guards.dtype_view_scatter",
    },
    "duplicate-index-writeorder": {
        "description": "Inductor computed duplicate-index read-modify-write assignment write-order divergence",
        "module": "torch_correctness_guards.guards.duplicate_index_writeorder",
    },
    "dynamic-clamp": {
        "description": "Inductor stale automatically-dynamic Python float reused in torch.clamp bounds",
        "module": "torch_correctness_guards.guards.dynamic_clamp",
    },
    "equality-fusion": {
        "description": "Inductor low-precision division fused into equality/argmax comparison skips rounding boundary",
        "module": "torch_correctness_guards.guards.equality_fusion",
    },
    "embeddingbag-freq-scale": {
        "description": "MPS embedding_bag silently ignores scale_grad_by_freq=True in backward",
        "module": "torch_correctness_guards.guards.embeddingbag_freq_scale",
    },
    "expand-fill": {
        "description": "Inductor wrong values for fill_ on broadcast views created by Tensor.expand()",
        "module": "torch_correctness_guards.guards.expand_fill",
    },
    "fp16-layernorm-tail": {
        "description": "CPU float16 layer_norm returns nonzero output for exact-constant rows",
        "module": "torch_correctness_guards.guards.fp16_layernorm_tail",
    },
    "full-dtype": {
        "description": "Inductor torch.full symbolic fill skips dtype cast and narrow-integer overflow checks",
        "module": "torch_correctness_guards.guards.full_dtype",
    },
    "inplace-slice-shift-aliasing": {
        "description": "Inductor in-place slice-shift assignment corrupts overlapping source/target storage",
        "module": "torch_correctness_guards.guards.inplace_slice_shift_aliasing",
    },
    "int64-index-truncation": {
        "description": "Inductor truncates int64 arange-multiply expressions to 32-bit-range arithmetic",
        "module": "torch_correctness_guards.guards.int64_index_truncation",
    },
    "linalg-nan": {
        "description": "torch.linalg values-only decompositions silently swallow NaN/Inf input",
        "module": "torch_correctness_guards.guards.linalg_nan",
    },
    "linalg-pinv-complex-grad": {
        "description": "AOTAutograd computes wrong complex torch.linalg.pinv gradients under torch.compile",
        "module": "torch_correctness_guards.guards.linalg_pinv_complex_grad",
    },
    "memory-budget-rng": {
        "description": "AOTAutograd activation_memory_budget can recompute RNG ops with fresh randomness in backward",
        "module": "torch_correctness_guards.guards.memory_budget_rng",
    },
    "mps-copy-dtype": {
        "description": "MPS tensor copies into CPU float64/complex128 destinations silently lose data",
        "module": "torch_correctness_guards.guards.mps_copy_dtype",
    },
    "mps-linalg-stride": {
        "description": "MPS linalg solves return row-major layouts where CPU/CUDA return column-major",
        "module": "torch_correctness_guards.guards.mps_linalg_stride",
    },
    "scatter-copyback-alias": {
        "description": "Inductor return-value aliasing drift for scatter copy-back and no-op elimination rewrites",
        "module": "torch_correctness_guards.guards.scatter_copyback_alias",
    },
    "softmax-dim": {
        "description": "Inductor attention-pattern rewrite ignores explicit non-last-axis softmax dim",
        "module": "torch_correctness_guards.guards.softmax_dim",
    },
    "tiled-reduction-tail-store": {
        "description": "Inductor CPU 2D-tiled reduction tail store can overrun or corrupt outputs",
        "module": "torch_correctness_guards.guards.tiled_reduction_tail_store",
    },
    "normal-dtype-promotion": {
        "description": "torch.compile Normal.sample() silently promotes dtype away from eager loc dtype",
        "module": "torch_correctness_guards.guards.normal_dtype_promotion",
    },
    "shuffle-sample-frozen": {
        "description": "Dynamo freezes random.shuffle/random.sample results at trace time inside torch.compile",
        "module": "torch_correctness_guards.guards.shuffle_sample_frozen",
    },
    "std-precision": {
        "description": "Inductor std/var-family reductions use float32 accumulation where CPU eager uses double precision",
        "module": "torch_correctness_guards.guards.std_precision",
    },
    "transpose-argmin": {
        "description": "Inductor wrong argmin/argmax flat index after transpose plus intervening op",
        "module": "torch_correctness_guards.guards.transpose_argmin",
    },
}


def _load_guard(name: str):
    if name == "addcdiv-stale-scalar":
        from .guards import addcdiv_stale_scalar

        return addcdiv_stale_scalar
    if name == "as-strided-restride-oob":
        from .guards import as_strided_restride_oob

        return as_strided_restride_oob
    if name == "checkpoint-noise":
        from .guards import checkpoint_noise

        return checkpoint_noise
    if name == "compile-validation":
        from .guards import compile_validation

        return compile_validation
    if name == "cpu-backward-nan-tail":
        from .guards import cpu_backward_nan_tail

        return cpu_backward_nan_tail
    if name == "dynamo-closure-descriptor":
        from .guards import dynamo_closure_descriptor

        return dynamo_closure_descriptor
    if name == "dtype-view-scatter":
        from .guards import dtype_view_scatter

        return dtype_view_scatter
    if name == "duplicate-index-writeorder":
        from .guards import duplicate_index_writeorder

        return duplicate_index_writeorder
    if name == "dynamic-clamp":
        from .guards import dynamic_clamp

        return dynamic_clamp
    if name == "equality-fusion":
        from .guards import equality_fusion

        return equality_fusion
    if name == "embeddingbag-freq-scale":
        from .guards import embeddingbag_freq_scale

        return embeddingbag_freq_scale
    if name == "expand-fill":
        from .guards import expand_fill

        return expand_fill
    if name == "fp16-layernorm-tail":
        from .guards import fp16_layernorm_tail

        return fp16_layernorm_tail
    if name == "full-dtype":
        from .guards import full_dtype

        return full_dtype
    if name == "inplace-slice-shift-aliasing":
        from .guards import inplace_slice_shift_aliasing

        return inplace_slice_shift_aliasing
    if name == "int64-index-truncation":
        from .guards import int64_index_truncation

        return int64_index_truncation
    if name == "linalg-nan":
        from .guards import linalg_nan

        return linalg_nan
    if name == "linalg-pinv-complex-grad":
        from .guards import linalg_pinv_complex_grad

        return linalg_pinv_complex_grad
    if name == "memory-budget-rng":
        from .guards import memory_budget_rng

        return memory_budget_rng
    if name == "mps-copy-dtype":
        from .guards import mps_copy_dtype

        return mps_copy_dtype
    if name == "mps-linalg-stride":
        from .guards import mps_linalg_stride

        return mps_linalg_stride
    if name == "scatter-copyback-alias":
        from .guards import scatter_copyback_alias

        return scatter_copyback_alias
    if name == "softmax-dim":
        from .guards import softmax_dim

        return softmax_dim
    if name == "tiled-reduction-tail-store":
        from .guards import tiled_reduction_tail_store

        return tiled_reduction_tail_store
    if name == "normal-dtype-promotion":
        from .guards import normal_dtype_promotion

        return normal_dtype_promotion
    if name == "shuffle-sample-frozen":
        from .guards import shuffle_sample_frozen

        return shuffle_sample_frozen
    if name == "std-precision":
        from .guards import std_precision

        return std_precision
    if name == "transpose-argmin":
        from .guards import transpose_argmin

        return transpose_argmin
    raise KeyError(name)  # defensive; argparse constrains this.


def _print_addcdiv_report(report, *, no_color: bool) -> None:
    style = resolve_style(no_color_flag=no_color)
    print_fields([("torch version", report["torch_version"])])

    if report["any_native_diverges"]:
        print(status_headline(style, "fail", "torch.compile(inductor) stale-scalar addcdiv_/addcmul_ divergence reproduced on this host"))
    else:
        print(status_headline(style, "info", "no stale-scalar divergence reproduced on this host's installed torch build"))

    if report["guard_fully_correct"]:
        print(status_headline(style, "ok", "guard matches eager on every case, including under torch.compile"))
    else:
        print(status_headline(style, "fail", "guard did NOT match eager on at least one case"))

    section("adam-step cases (iteration count -> eager vs compiled(native) vs compiled(guarded))")
    for c in report["cases"]:
        flag = "DIVERGES" if c["native_diverges"] else "ok"
        guard_flag = "guard-ok" if c["guard_matches_eager"] else "GUARD-FAILED"
        print_fields(
            [
                (
                    f"iters={c['iterations']}",
                    f"max|eager-native|={c['max_abs_diff_native']:.6e}  "
                    f"max|eager-guarded|={c['max_abs_diff_guarded']:.6e}  {flag:9s}  {guard_flag}",
                )
            ]
        )


def _print_as_strided_report(report, *, no_color: bool) -> None:
    style = resolve_style(no_color_flag=no_color)
    print_fields([("torch version", report["torch_version"])])

    if report["any_divergence_bug"]:
        print(status_headline(style, "warn", "as_strided restride divergence reproduced on this host"))
    else:
        print(status_headline(style, "info", "no as_strided restride divergence reproduced on this host's installed torch build"))

    if report["guard_fully_correct"]:
        print(status_headline(style, "ok", "safe_as_strided() matches eager on every call, including under torch.compile"))
    else:
        print(status_headline(style, "fail", "guard did NOT match eager on at least one call"))

    section("calls (index -> input values -> eager vs compiled vs guard)")
    for c in report["cases"]:
        flag = "DIVERGENCE" if c["divergence_bug"] else "ok"
        guard_flag = "guard-ok" if c["guard_matches_eager"] else "GUARD-FAILED"
        compiled_desc = c["compiled_error"] if not c["compiled_ok"] else str(c["compiled_value"])
        print_fields(
            [
                (
                    f"call {c['call_index']}",
                    f"x={c['x_values']!s:16s} eager={c['eager_value']}  "
                    f"compiled={compiled_desc}  {flag:10s}  {guard_flag}",
                )
            ]
        )


def _print_checkpoint_noise_report(report, *, no_color: bool) -> None:
    style = resolve_style(no_color_flag=no_color)
    print_fields([("torch version", report["torch_version"])])

    if report["bug_reproduced"]:
        print(status_headline(style, "warn", "F.rrelu checkpoint/saved-hooks gradient corruption reproduced on this host (#193671)"))
    else:
        print(status_headline(style, "info", "F.rrelu checkpoint/saved-hooks bug did NOT reproduce on this host's installed torch build"))

    if report["bug_reproduced"] and not report["mechanism_confirmed"]:
        print(status_headline(style, "warn", "bug reproduced but the expected root-cause signature did not fully match"))

    if report["guard_fully_correct"]:
        print(status_headline(style, "ok", "safe_rrelu() produces identical, NaN-free gradients across eager, checkpoint, and saved-tensors-hooks execution"))
    else:
        print(status_headline(style, "fail", "safe_rrelu() did NOT match expected behavior in at least one mode"))

    section("non-reentrant checkpoint")
    c = report["checkpoint_case"]
    print_fields(
        [
            ("buggy F.rrelu gradient diff vs uncheckpointed", f"{c['buggy_diff']!s}"),
            ("buggy forward-output diff (expected 0.0)", f"{c['buggy_forward_diff']!s}"),
            ("buggy diff with early-stop disabled (expected 0.0)", f"{c['no_early_stop_diff']!s}"),
            ("guard (safe_rrelu) gradient diff", f"{c['guard_diff']!s}"),
            ("guard has NaN", f"{c['guard_has_nan']!s}"),
        ]
    )

    section("saved_tensors_hooks (clone-based pack/unpack)")
    c = report["saved_hooks_case"]
    print_fields(
        [
            ("buggy F.rrelu gradient diff vs unhooked", f"{c['buggy_diff']!s}"),
            ("guard (safe_rrelu) gradient diff", f"{c['guard_diff']!s}"),
            ("guard has NaN", f"{c['guard_has_nan']!s}"),
        ]
    )


def _print_duplicate_index_writeorder_report(report, *, no_color: bool) -> None:
    style = resolve_style(no_color_flag=no_color)
    print_fields([("torch version", report["torch_version"]), ("tracking issue", report["issue_url"])])

    if report["bug_reproduced"]:
        print(status_headline(style, "fail", "computed duplicate-index write-order miscompilation reproduced on this host"))
    else:
        print(status_headline(style, "info", "no computed duplicate-index write-order miscompilation reproduced on this host's installed torch build"))

    if report["isolation_confirmed"]:
        print(status_headline(style, "ok", "literal duplicate-index control case is unaffected"))
    else:
        print(status_headline(style, "warn", "literal duplicate-index control case also diverged"))

    if report["guard_fully_correct"]:
        print(status_headline(style, "ok", "safe_dup_index_assign() restores eager write-order semantics"))
    else:
        print(status_headline(style, "fail", "safe_dup_index_assign() did NOT restore eager semantics"))

    section("cases (kind -> eager vs native vs guarded)")
    for c in report["cases"]:
        native_flag = "DIVERGED" if not c["native_matches_eager"] else "ok"
        guard_flag = "guard-ok" if c["guard_matches_eager"] else "GUARD-FAILED"
        print_fields(
            [
                (
                    c["kind"],
                    f"native={native_flag:9s}  guard_matches_eager={str(c['guard_matches_eager']):5s}  {guard_flag}",
                )
            ]
        )


def _print_dynamic_clamp_report(report, *, no_color: bool) -> None:
    style = resolve_style(no_color_flag=no_color)
    print_fields([("torch version", report["torch_version"])])

    if report["any_stale_reuse_bug"]:
        print(status_headline(style, "warn", "stale dynamic-float clamp reuse reproduced on this host"))
    else:
        print(status_headline(style, "info", "no stale-reuse divergence reproduced on this host's installed torch build"))

    if report["guard_fully_correct"]:
        print(status_headline(style, "ok", "safe_clamp() matches eager on every call, including under torch.compile"))
    else:
        print(status_headline(style, "fail", "guard did NOT match eager on at least one call"))

    section("calls (index -> shape/requires_grad/limit -> eager vs compiled vs guard)")
    for c in report["cases"]:
        flag = "STALE-REUSE" if c["stale_reuse_bug"] else "ok"
        guard_flag = "guard-ok" if c["guard_matches_eager"] else "GUARD-FAILED"
        print_fields(
            [
                (
                    f"call {c['call_index']}",
                    f"shape={c['shape']!s:10s} requires_grad={c['requires_grad']!s:5s} limit={c['limit']}  "
                    f"eager={c['eager_value']:.6f}  compiled={c['compiled_value']:.6f}  {flag:11s}  {guard_flag}",
                )
            ]
        )


def _print_equality_fusion_report(report, *, no_color: bool) -> None:
    style = resolve_style(no_color_flag=no_color)
    print_fields([("torch version", report["torch_version"]), ("tracking issue", report["issue_url"])])

    if report["any_default_inductor_diverges_on_bf16"]:
        print(status_headline(style, "fail", "Inductor bf16 division/equality fusion divergence reproduced on this host"))
    else:
        print(status_headline(style, "info", "no bf16 division/equality fusion divergence reproduced on this host's installed torch build"))

    if report["guard_fully_correct"]:
        print(status_headline(style, "ok", "precision_safe_division_compare() matches eager on every case"))
    else:
        print(status_headline(style, "fail", "precision_safe_division_compare() did NOT match eager on at least one case"))

    if report["emulate_precision_casts_fully_correct"]:
        print(status_headline(style, "ok", "emulate_precision_casts=True also matches eager on every case"))
    else:
        print(status_headline(style, "warn", "emulate_precision_casts=True did not match eager on at least one case"))

    section("cases (dtype -> eager vs default-inductor vs guarded)")
    for c in report["cases"]:
        default_flag = "MATCH" if c["inductor_default_matches_eager"] else "DIVERGES"
        guard_flag = "guard-ok" if c["guarded_matches_eager"] else "GUARD-FAILED"
        nonfinite_flag = " nonfinite-downstream!" if c["inductor_default_any_nonfinite_downstream"] else ""
        print_fields(
            [
                (
                    c["description"][:48],
                    f"default={default_flag:9s}{nonfinite_flag}  {guard_flag}",
                )
            ]
        )


def _print_embeddingbag_freq_scale_report(report, *, no_color: bool) -> None:
    style = resolve_style(no_color_flag=no_color)
    print_fields([("torch version", report["torch_version"]), ("tracking issue", report["issue_url"])])

    if report["any_native_silently_wrong"]:
        print(status_headline(style, "fail", "MPS embedding_bag(scale_grad_by_freq=True) silent bug reproduced on this host"))
    else:
        print(status_headline(style, "info", "no silent scale_grad_by_freq bug reproduced on this host's available devices"))

    if report["guard_fully_correct"]:
        print(status_headline(style, "ok", "safe_embedding_bag() matches the CPU-oracle gradient on every exercised device"))
    else:
        print(status_headline(style, "fail", "guard did NOT match the expected contract on at least one device"))

    section("per-device results")
    for c in report["cases"]:
        if not c["ran"]:
            print_fields([(c["device"], f"skipped: {c['skip_reason']}")])
            continue
        native_flag = "matches" if c["native_matches_oracle"] else "SILENTLY-WRONG"
        guard_flag = "guard-ok" if c["guard_matches_oracle"] else "GUARD-FAILED"
        print_fields(
            [
                (
                    c["device"],
                    f"native={c['native_grad_row1']} oracle={c['cpu_oracle_grad_row1']} "
                    f"guard={c['guard_grad_row1']} native={native_flag} {guard_flag}",
                )
            ]
        )


def _print_expand_fill_report(report, *, no_color: bool) -> None:
    style = resolve_style(no_color_flag=no_color)
    print_fields([("torch version", report["torch_version"])])

    if report["any_expand_fill_divergence"]:
        print(status_headline(style, "fail", "torch.compile(inductor) expand()+fill_() divergence reproduced on this host"))
    else:
        print(status_headline(style, "info", "no expand()+fill_() divergence reproduced on this host's installed torch build"))

    if report["guard_fully_correct"]:
        print(status_headline(style, "ok", "safe_fill_() matches eager on every case, including under torch.compile"))
    else:
        print(status_headline(style, "fail", "guard did NOT match eager on at least one case"))

    section("expand()+fill_() cases (expand_rows, base -> eager vs compiled(native) vs compiled(guarded))")
    for c in report["cases"]:
        flag = "DIVERGES" if c["native_diverges"] else "ok"
        guard_flag = "guard-ok" if c["guard_matches_eager"] else "GUARD-FAILED"
        print_fields(
            [
                (
                    f"rows={c['expand_rows']} base={c['base_values']} fill={c['fill_value']}",
                    f"eager={c['eager_result']}  compiled_native={c['compiled_native_result']}  "
                    f"compiled_guarded={c['compiled_guarded_result']}  {flag:9s}  {guard_flag}",
                )
            ]
        )


def _print_fp16_layernorm_tail_report(report, *, no_color: bool) -> None:
    style = resolve_style(no_color_flag=no_color)
    print_fields([("torch version", report["torch_version"])])

    if report["any_bug_present"]:
        print(status_headline(style, "fail", "CPU float16 layer_norm exact-constant-row bug reproduced on this host"))
        if report["bug_signature_confirmed"]:
            print(status_headline(style, "info", "confirmed signature: wrong nonzero outputs match a documented tail or whole-row pattern"))
    else:
        print(status_headline(style, "info", "bug NOT reproduced on this host's installed torch build"))

    if report["guard_fully_correct"]:
        print(status_headline(style, "ok", "safe_layer_norm() is exact-zero on constant rows and at least as accurate on non-constant rows"))
    else:
        print(status_headline(style, "fail", "safe_layer_norm() did NOT produce a correct result on at least one case"))

    section("constant-row cases (length -> native nonzero count vs guard nonzero count)")
    for c in report["constant_cases"]:
        native_flag = "BUG" if c["buggy_nonzero_count"] else "ok"
        guard_flag = "guard-ok" if c["guard_is_correct"] else "GUARD-FAILED"
        print_fields(
            [
                (
                    f"n={c['length']}",
                    f"native_nonzero={c['buggy_nonzero_count']} expected={c['expected_nonzero_count']} "
                    f"guard_nonzero={c['guard_nonzero_count']} {native_flag:4s} {guard_flag}",
                )
            ]
        )

    section("non-constant accuracy cases (length -> native diff vs guard diff to fp64 oracle)")
    for c in report["nonconstant_cases"]:
        guard_flag = "guard-ok" if c["guard_at_least_as_accurate"] else "GUARD-FAILED"
        print_fields(
            [
                (
                    f"n={c['length']}",
                    f"native_diff={c['max_abs_diff_native_vs_reference']:.6e} "
                    f"guard_diff={c['max_abs_diff_guard_vs_reference']:.6e} {guard_flag}",
                )
            ]
        )


def _print_full_dtype_report(report, *, no_color: bool) -> None:
    style = resolve_style(no_color_flag=no_color)
    print_fields([("torch version", report["torch_version"])])

    if report["any_bool_fill_divergence"]:
        print(status_headline(style, "warn", "torch.compile(inductor) bool-fill dtype-cast divergence reproduced on this host"))
    else:
        print(status_headline(style, "info", "no bool-fill divergence reproduced on this host's installed torch build"))

    if report["any_int8_silent_overflow"]:
        print(status_headline(style, "fail", "int8 overflow check silently skipped under torch.compile (eager raises, compiled does not)"))
    else:
        print(status_headline(style, "info", "no silent int8-overflow case reproduced on this host"))

    if report.get("any_overflow_dtype_silent_overflow"):
        print(status_headline(style, "fail", "at least one overflow case (int8, or int16/uint8 under this process's numpy state) silently skipped its check under torch.compile"))
    else:
        print(status_headline(style, "info", "no silent overflow case reproduced across the checked integer dtypes on this host/process"))

    if report["guard_fully_correct"]:
        print(status_headline(style, "ok", "safe_full() matches eager on every case, including under torch.compile"))
    else:
        print(status_headline(style, "fail", "safe_full() did NOT match eager on at least one case"))

    section("bool-fill cases (fill value -> eager vs compiled(native) vs compiled(guarded))")
    for c in report["bool_fill_cases"]:
        flag = "DIVERGES" if c["native_diverges"] else "ok"
        guard_flag = "guard-ok" if c["guard_matches_eager"] else "GUARD-FAILED"
        print_fields(
            [
                (
                    f"fill={c['fill_value']}",
                    f"eager={c['eager_result']}  compiled_native={c['compiled_native_result']}  "
                    f"compiled_guarded={c['compiled_guarded_result']}  {flag:9s}  {guard_flag}",
                )
            ]
        )

    section("int8-overflow cases (fill value -> did each path raise the overflow error?)")
    for c in report["int8_overflow_cases"]:
        flag = "SILENTLY-WRONG" if c["native_silently_wrong"] else "ok"
        guard_flag = "guard-ok" if c["guard_matches_eager"] else "GUARD-FAILED"
        print_fields(
            [
                (
                    f"fill={c['fill_value']}",
                    f"eager_raised={c['eager_raised']!s:5s}  compiled_native_raised={c['compiled_native_raised']!s:5s}  "
                    f"compiled_guarded_raised={c['compiled_guarded_raised']!s:5s}  {flag:15s}  {guard_flag}",
                )
            ]
        )

    extra_cases = report.get("extra_overflow_dtype_cases") or []
    if extra_cases:
        section("additional narrow-integer-dtype overflow cases (int16/uint8; bug presence here is numpy-process-state dependent)")
        for c in extra_cases:
            flag = "SILENTLY-WRONG" if c["native_silently_wrong"] else "ok"
            guard_flag = "guard-ok" if c["guard_matches_eager"] else "GUARD-FAILED"
            print_fields(
                [
                    (
                        f"dtype={c.get('overflow_dtype', '?')} fill={c['fill_value']}",
                        f"eager_raised={c['eager_raised']!s:5s}  compiled_native_raised={c['compiled_native_raised']!s:5s}  "
                        f"compiled_guarded_raised={c['compiled_guarded_raised']!s:5s}  {flag:15s}  {guard_flag}",
                    )
                ]
            )


def _print_inplace_slice_shift_aliasing_report(report, *, no_color: bool) -> None:
    style = resolve_style(no_color_flag=no_color)
    print_fields([("torch version", report["torch_version"]), ("tracking issue", report["issue_url"])])

    if report["any_native_bug"]:
        print(status_headline(style, "fail", "Inductor in-place slice-shift aliasing bug reproduced on this host (pytorch#197829)"))
    else:
        print(status_headline(style, "info", "no in-place slice-shift aliasing divergence reproduced on this host's installed torch build"))

    if report["guard_fully_correct"]:
        print(status_headline(style, "ok", "safe_slice_shift() restores eager's correct semantics on every case"))
    else:
        print(status_headline(style, "fail", "guard did NOT restore correct semantics on at least one case"))

    section("cases (shape, shift -> compiled mismatches / guard result)")
    for c in report["cases"]:
        native_flag = (
            f"WRONG ({c['compiled_mismatches']}/{c['compiled_total']})"
            if not c["compiled_matches_eager"]
            else "ok"
        )
        guard_flag = "guard-ok" if c["guarded_matches_eager"] else "GUARD-FAILED"
        print_fields(
            [
                (
                    f"shape={c['shape']} shift={c['shift']}",
                    f"eager_correct={c['eager_correct']}  native={native_flag:20s}  {guard_flag}",
                )
            ]
        )


def _print_int64_index_truncation_report(report, *, no_color: bool) -> None:
    style = resolve_style(no_color_flag=no_color)
    print_fields([("torch version", report["torch_version"]), ("tracking issue", report["issue_url"])])

    if report["any_native_diverges"]:
        print(status_headline(style, "fail", "Inductor int64 arange-multiply truncation reproduced on this host"))
    else:
        print(status_headline(style, "info", "no int64 arange-multiply truncation reproduced on this host's installed torch build"))

    if report["guard_fully_correct"]:
        print(status_headline(style, "ok", "safe_int64_arange_mul() matches eager on every case"))
    else:
        print(status_headline(style, "fail", "safe_int64_arange_mul() did NOT match eager on at least one case"))

    section("cases (range/multiplier -> native vs guard)")
    for c in report["cases"]:
        native_flag = "SILENT-WRONG" if c["native_diverges"] else "ok"
        guard_flag = "guard-ok" if c["guard_matches_eager"] else "GUARD-FAILED"
        print_fields(
            [
                (
                    c["description"][:60],
                    f"native={native_flag:12s}  {guard_flag}",
                )
            ]
        )


def _print_scatter_copyback_alias_report(report, *, no_color: bool) -> None:
    style = resolve_style(no_color_flag=no_color)
    print_fields([("torch version", report["torch_version"])])

    if report["any_native_alias_bug"]:
        print(status_headline(style, "fail", "Inductor direct-scatter-copyback aliasing bug reproduced on this host (pytorch#195451)"))
    else:
        print(status_headline(style, "info", "no direct-scatter-copyback aliasing divergence reproduced on this host's installed torch build"))

    if report["any_noop_alias_bug"]:
        print(status_headline(style, "fail", "Inductor no-op-elimination aliasing bug reproduced on this host (pytorch#197893)"))
    else:
        print(status_headline(style, "info", "no no-op-elimination aliasing divergence reproduced on this host's installed torch build"))

    if report["guard_fully_correct"] and report["noop_guard_fully_correct"]:
        print(status_headline(style, "ok", "safe_compiled_scatter_returning() restores eager's non-aliasing contract for both root causes"))
    else:
        print(status_headline(style, "fail", "guard did NOT restore eager's non-aliasing contract on at least one case"))

    section("scatter-copyback cases")
    for c in report["cases"]:
        native_flag = "ALIASED+CORRUPTED" if c["compiled_input_corrupted_after_output_mutation"] else (
            "aliased" if c["compiled_aliases_input"] else "ok"
        )
        guard_flag = "guard-ok" if (
            not c["guarded_aliases_input"]
            and not c["guarded_input_corrupted_after_output_mutation"]
            and c["guarded_values_match_eager"]
        ) else "GUARD-FAILED"
        print_fields([(f"x0={c['x0']} src={c['src']}", f"native={native_flag:18s}  {guard_flag}")])

    section("no-op-elimination cases")
    for c in report["noop_cases"]:
        native_flag = "ALIASED+CORRUPTED" if c["compiled_input_corrupted_after_output_mutation"] else (
            "aliased" if c["compiled_aliases_input"] else "ok"
        )
        guard_flag = "guard-ok" if (
            not c["guarded_aliases_input"]
            and not c["guarded_input_corrupted_after_output_mutation"]
            and c["guarded_values_match_eager"]
        ) else "GUARD-FAILED"
        print_fields([(f"op={c['op_name']}", f"native={native_flag:18s}  {guard_flag}")])


def _print_softmax_dim_report(report, *, no_color: bool) -> None:
    style = resolve_style(no_color_flag=no_color)
    print_fields([("torch version", report["torch_version"])])

    if report["any_softmax_dim_divergence"]:
        print(status_headline(style, "fail", "torch.compile(inductor) softmax-dim attention rewrite divergence reproduced on this host"))
    else:
        print(status_headline(style, "info", "no softmax-dim attention rewrite divergence reproduced on this host's installed torch build"))

    if report["guard_fully_correct"]:
        print(status_headline(style, "ok", "safe_softmax_attention() matches eager on every case, including under torch.compile"))
    else:
        print(status_headline(style, "fail", "safe_softmax_attention() did NOT match eager on at least one case"))

    section("softmax-dim cases (dim, shape -> requested-vs-native diff / wrong-last-axis diff)")
    for c in report["cases"]:
        flag = "DIVERGES" if c["native_diverges"] else "ok"
        wrong_axis_flag = "matches-wrong-lastaxis" if c["native_matches_wrong_lastaxis"] else "not-lastaxis-match"
        guard_flag = "guard-ok" if c["guard_matches_eager"] else "GUARD-FAILED"
        print_fields(
            [
                (
                    f"dim={c['dim']} shape={c['shape']}",
                    f"requested={c['eager_requested_vs_compiled_native_rel_diff']:.6f}  "
                    f"lastaxis={c['eager_lastaxis_vs_compiled_native_rel_diff']:.2e}  "
                    f"{flag:9s}  {wrong_axis_flag:24s}  {guard_flag}",
                )
            ]
        )


def _print_compile_validation_report(report, *, no_color: bool) -> None:
    style = resolve_style(no_color_flag=no_color)
    print_fields([("torch version", report["torch_version"])])

    if report["any_validation_bypassed"]:
        print(status_headline(style, "fail", "torch.compile input-validation bypass reproduced on this host"))
    else:
        print(status_headline(style, "info", "no input-validation bypass reproduced on this host's installed torch build"))

    if report["guard_fully_correct"]:
        print(status_headline(style, "ok", "guard wrappers restore eager's validation contract on every case"))
    else:
        print(status_headline(style, "fail", "at least one guard wrapper did NOT restore eager's validation contract"))

    if report["boundary_cases_not_spuriously_flagged"]:
        print(status_headline(style, "ok", "valid boundary inputs are never spuriously flagged"))
    else:
        print(status_headline(style, "fail", "at least one VALID boundary input was incorrectly flagged"))

    section("cases (fixture -> eager vs compiled(native) vs guarded)")
    for c in report["cases"]:
        if c["is_boundary_case"]:
            flag = "boundary-ok" if not c["guarded_raised"] else "BOUNDARY-FAILED"
        else:
            flag = "BYPASSED" if c["validation_bypassed"] else "ok"
        guard_flag = "guard-ok" if c["guard_restores_validation"] else "GUARD-FAILED"
        print_fields(
            [
                (
                    c["name"],
                    f"eager_raised={c['eager_raised']!s:5s}  "
                    f"compiled_raised={c['compiled_raised']!s:5s}  "
                    f"guarded_raised={c['guarded_raised']!s:5s}  "
                    f"{flag:15s}  {guard_flag}",
                )
            ]
        )


def _print_cpu_backward_nan_tail_report(report, *, no_color: bool) -> None:
    style = resolve_style(no_color_flag=no_color)
    print_fields(
        [
            ("torch version", report["torch_version"]),
            ("tracking issue", ", ".join(report["issue_urls"])),
        ]
    )

    if report["any_bug_present"]:
        print(status_headline(style, "warn", "length-dependent NaN-gradient divergence reproduced on this host's installed torch build"))
    else:
        print(status_headline(style, "info", "bug NOT reproduced on this host's installed torch build (fixed upstream)"))

    if report["guard_fully_effective"]:
        print(status_headline(style, "ok", "every safe_* backward guard is length-independent at every tested length/op"))
    else:
        print(status_headline(style, "fail", "at least one backward guard is still length-dependent"))

    section("per-case results (op x tensor length)")
    for c in report["cases"]:
        bug_flag = "LENGTH-DEP-BUG" if c["buggy_position_dependent"] else "consistent"
        guard_flag = "guard-ok" if c["guard_consistent"] else "GUARD-FAILED"
        print_fields(
            [
                (
                    f"{c['op']} n={c['length']}",
                    f"unguarded[0]={c['buggy_grad_first']!s:>6s} unguarded[-1]={c['buggy_grad_last']!s:>6s}  {bug_flag:15s}  {guard_flag}",
                )
            ]
        )


def _print_normal_dtype_promotion_report(report, *, no_color: bool) -> None:
    style = resolve_style(no_color_flag=no_color)
    print_fields([("torch version", report["torch_version"])])

    if report["any_native_divergence"]:
        print(status_headline(style, "fail", "torch.compile Normal.sample() dtype-promotion divergence reproduced on this host (pytorch#194547)"))
    else:
        print(status_headline(style, "info", "no Normal.sample() dtype-promotion divergence reproduced on this host's installed torch build"))

    if report["guard_fully_correct"]:
        print(status_headline(style, "ok", "safe_compiled_normal_sample() restores eager's dtype-preservation contract on every case"))
    else:
        print(status_headline(style, "fail", "guard did NOT restore eager's dtype contract on at least one case"))

    section("cases (loc dtype, scale dtype -> eager/compiled/guarded dtype)")
    for c in report["cases"]:
        native_flag = "DIVERGES" if c["dtype_diverges"] else "matches"
        guard_flag = "guard-ok" if c["guarded_matches_eager"] else "GUARD-FAILED"
        print_fields(
            [
                (
                    f"loc={c['loc_dtype']:16s} scale={c['scale_dtype']:16s}",
                    f"eager={c['eager_dtype']:16s} compiled={c['compiled_dtype']:16s} "
                    f"native={native_flag:8s}  {guard_flag}",
                )
            ]
        )


def _print_shuffle_sample_frozen_report(report, *, no_color: bool) -> None:
    style = resolve_style(no_color_flag=no_color)
    print_fields([("torch version", report["torch_version"]), ("seed", report["seed"]), ("calls", report["calls"])])

    if report["any_native_frozen"]:
        print(status_headline(style, "fail", "random.shuffle/random.sample frozen at trace time reproduced on this host (pytorch#197085)"))
    else:
        print(status_headline(style, "info", "no trace-time freeze reproduced on this host's installed torch build"))

    if report["guard_fully_correct"]:
        print(status_headline(style, "ok", "safe_shuffle()/safe_sample() restore eager per-call randomness on every case"))
    else:
        print(status_headline(style, "fail", "guards did NOT restore eager semantics on at least one case"))

    section("cases (kind -> native vs guard vs eager)")
    for c in report["cases"]:
        native_flag = "FROZEN" if c["native_frozen_after_first_call"] else "ok"
        guard_flag = "guard-ok" if c["guard_correct"] else "GUARD-FAILED"
        print_fields(
            [
                (
                    c["kind"],
                    f"native={native_flag:8s}  guard_matches_eager={str(c['guard_matches_eager']):5s}  {guard_flag}",
                )
            ]
        )


def _print_std_precision_report(report, *, no_color: bool) -> None:
    style = resolve_style(no_color_flag=no_color)
    print_fields([("torch version", report["torch_version"])])

    if report["any_std_divergence"]:
        print(status_headline(style, "warn", "torch.compile(inductor) std/var precision divergence reproduced on this host"))
    else:
        print(status_headline(style, "info", "no eager-vs-compiled std/var precision divergence reproduced on this host"))

    if report["any_silent_zero_gradient"]:
        print(status_headline(style, "fail", "silent all-zero std gradient reproduced under torch.compile for small-magnitude input"))
    else:
        print(status_headline(style, "info", "no silent zero-gradient case reproduced on this host"))

    if report["guard_fully_correct"]:
        print(status_headline(style, "ok", "safe_std()/safe_var()/safe_var_mean()/safe_std_mean() match eager on every case"))
    else:
        print(status_headline(style, "fail", "guards did NOT match eager on at least one case"))

    section("std cases (magnitude exponent -> eager vs compiled value)")
    for c in report["std_cases"]:
        flag = "DIVERGES" if c["diverges"] else "ok"
        guard_flag = "guard-ok" if c["guard_matches_eager"] else "GUARD-FAILED"
        print_fields(
            [
                (
                    f"1e{c['magnitude_exponent']:+d}",
                    f"eager={c['eager_value']!s:>24}  compiled={c['compiled_value']!s:>10}  {flag:9s}  {guard_flag}",
                )
            ]
        )

    section("zero-gradient cases (magnitude exponent -> eager vs compiled gradient)")
    for c in report["zero_gradient_cases"]:
        flag = "SILENT-ZERO-GRAD" if c["silent_zero_gradient_bug"] else "ok"
        guard_flag = "guard-ok" if c["guard_matches_eager"] else "GUARD-FAILED"
        print_fields(
            [
                (
                    f"1e{c['magnitude_exponent']:+d}",
                    f"eager_grad_zero={c['eager_grad_is_zero']!s:5s}  compiled_grad_zero={c['compiled_grad_is_zero']!s:5s}  {flag:17s}  {guard_flag}",
                )
            ]
        )


def _print_transpose_argmin_report(report, *, no_color: bool) -> None:
    style = resolve_style(no_color_flag=no_color)
    print_fields([("torch version", report["torch_version"])])

    if report["any_transpose_argreduce_divergence"]:
        print(status_headline(style, "fail", "torch.compile(inductor) transpose+op+argmin/argmax index divergence reproduced on this host"))
    else:
        print(status_headline(style, "info", "no transpose+op+argmin/argmax divergence reproduced on this host's installed torch build"))

    if report["guard_fully_correct"]:
        print(status_headline(style, "ok", "safe_reduce_index() matches eager on every case, including under torch.compile"))
    else:
        print(status_headline(style, "fail", "guard did NOT match eager on at least one case"))

    section("cases (op, mode, shape -> eager value vs compiled(native) value)")
    for c in report["cases"]:
        flag = "DIVERGES" if c["native_diverges"] else "ok"
        guard_flag = "guard-ok" if c["guard_matches_eager"] else "GUARD-FAILED"
        print_fields(
            [
                (
                    f"op={c['op']} mode={c['mode']} shape={c['shape']}",
                    f"eager={c['eager_value']}  compiled(native)={c['native_compiled_value']}  "
                    f"{flag:9s}  {guard_flag}",
                )
            ]
        )


def _print_tiled_reduction_tail_store_report(report, *, no_color: bool) -> None:
    style = resolve_style(no_color_flag=no_color)
    print_fields([("torch version", report["torch_version"]), ("tracking issue", report["issue_url"])])

    if report["any_bug_reproduced"]:
        print(status_headline(style, "fail", "Inductor 2D-tiled reduction tail-store bug reproduced on this host"))
    else:
        print(status_headline(style, "info", "bug NOT reproduced on this host's installed torch build"))

    if report["any_crash_observed"]:
        print(status_headline(style, "warn", "at least one raw compiled case crashed in an isolated worker process"))

    if report["guard_fully_correct"]:
        print(status_headline(style, "ok", "guard never silently trusts a corrupted result across all exercised sizes"))
    else:
        print(status_headline(style, "fail", "guard did NOT behave correctly for at least one size"))

    section("tile-boundary cases (size -> raw divergence/crash vs guarded behavior)")
    for c in report["cases"]:
        bug_flag = "BUG" if c["bug_reproduced"] else "ok"
        guard_flag = "guard-ok" if c["guard_behaved_correctly"] else "GUARD-FAILED"
        diff = c["max_abs_diff"] if c["max_abs_diff"] is not None else "n/a"
        print_fields(
            [
                (
                    f"n={c['size']}",
                    f"aligned={c['aligned_to_vector_width']!s:5s} max_diff={diff!s:>8s} "
                    f"bare_crashed={c['crashed_bare']!s:5s} guard_raised={c['guard_raised']!s:5s} "
                    f"guard_crashed={c['crashed_guarded']!s:5s} {bug_flag:4s} {guard_flag}",
                )
            ]
        )


def _print_dynamo_closure_descriptor_report(report, *, no_color: bool) -> None:
    style = resolve_style(no_color_flag=no_color)
    print_fields([("torch version", report["torch_version"])])

    if report["any_stale_reuse_bug"]:
        print(status_headline(style, "warn", "Dynamo closure-descriptor graph-reuse divergence reproduced on this host"))
    else:
        print(status_headline(style, "info", "no closure-descriptor graph-reuse divergence reproduced on this host's installed torch build"))

    if report["guard_fully_correct"]:
        print(status_headline(style, "ok", "safe_call() matches eager on every call, including under torch.compile"))
    else:
        print(status_headline(style, "fail", "safe_call() did NOT match eager on at least one call"))

    section("cases (captured op -> eager vs compiled(native) vs guarded)")
    for c in report["cases"]:
        flag = "STALE-REUSE" if c["stale_reuse_bug"] else "ok"
        guard_flag = "guard-ok" if c["guard_matches_eager"] else "GUARD-FAILED"
        print_fields(
            [
                (
                    f"op {c['op_name']}",
                    f"a={c['a']} b={c['b']} eager={c['eager_result']:.6f} "
                    f"compiled={c['compiled_result']:.6f} guarded={c['guard_result']:.6f} "
                    f"{flag:11s} {guard_flag}",
                )
            ]
        )


def _print_dtype_view_scatter_report(report, *, no_color: bool) -> None:
    style = resolve_style(no_color_flag=no_color)
    print_fields([("torch version", report["torch_version"]), ("tracking issue", report["issue_url"])])

    if report["any_native_value_bug"] or report["any_native_alias_bug"]:
        print(status_headline(style, "fail", "Inductor dtype-view/diagonal_scatter value+aliasing bug reproduced on this host"))
    else:
        print(status_headline(style, "info", "no Inductor dtype-view/diagonal_scatter divergence reproduced on this host's installed torch build"))

    if report["guard_fully_correct"]:
        print(status_headline(style, "ok", "safe_compiled_dtype_view_diagonal_scatter() restores eager's values and non-aliasing contract on every case"))
    else:
        print(status_headline(style, "fail", "guard did NOT restore eager's values/aliasing contract on at least one case"))

    section("cases (n -> eager/compiled/guarded value+aliasing behavior)")
    for c in report["cases"]:
        native_flag = (
            "WRONG-VALUES+ALIASED"
            if (not c["compiled_values_match_eager"]) and c["compiled_aliases_input"]
            else "WRONG-VALUES" if not c["compiled_values_match_eager"]
            else "aliased" if c["compiled_aliases_input"]
            else "ok"
        )
        guard_flag = "guard-ok" if (
            c["guarded_values_match_eager"]
            and c["guarded_aliases_input_matches_eager"]
            and c["guarded_input_matches_eager_after_output_mutation"]
        ) else "GUARD-FAILED"
        print_fields(
            [
                (
                    f"n={int(len(c['diag']))}",
                    f"eager_aliases={c['eager_aliases_input']}  native={native_flag:22s}  {guard_flag}",
                )
            ]
        )


def _print_linalg_nan_report(report, *, no_color: bool) -> None:
    style = resolve_style(no_color_flag=no_color)
    print_fields([("torch version", report["torch_version"])])

    if report["any_bug_present"]:
        print(status_headline(style, "warn", "torch.linalg values-only decomposition silently returned finite output for NaN input on this host"))
    else:
        print(status_headline(style, "info", "no silent finite-output NaN swallow reproduced on this host's LAPACK backend"))

    if report["guard_fully_effective"]:
        print(status_headline(style, "ok", "safe_svdvals() and safe_eigvalsh() reject every non-finite input case"))
    else:
        print(status_headline(style, "fail", "guard did NOT reject at least one non-finite input case"))

    section("NaN placement cases")
    for c in report["cases"]:
        native_flag = "silent-finite" if c["buggy_result_is_finite"] else ("raised" if c["buggy_raised"] else "propagates-nan")
        ref_flag = "raised" if c["reference_raised"] else ("ref-has-nan" if c["reference_has_nan"] else "REF-FINITE")
        guard_flag = "guard-raises" if c["guard_raises"] else "GUARD-MISSED"
        print_fields(
            [
                (
                    f"{c['op']} n={c['size']} pos={c['nan_position']}",
                    f"native={native_flag:14s}  reference={ref_flag:11s}  {guard_flag}",
                )
            ]
        )


def _print_linalg_pinv_complex_grad_report(report, *, no_color: bool) -> None:
    style = resolve_style(no_color_flag=no_color)
    print_fields([("torch version", report["torch_version"]), ("tracking issue", report["issue_url"])])

    if report["any_native_bug"]:
        print(status_headline(style, "fail", "AOTAutograd complex pinv/matrix_sqrth wrong-gradient bug reproduced on this host (pytorch#197084)"))
    else:
        print(status_headline(style, "info", "no complex pinv gradient divergence reproduced on this host's installed torch build"))

    if report["guard_fully_correct"]:
        print(status_headline(style, "ok", "safe_complex_pinv_grad() restores eager's correct gradient on every case"))
    else:
        print(status_headline(style, "fail", "guard did NOT restore the correct gradient on at least one case"))

    section("cases (shape, seed -> aot_eager/inductor/guarded max-abs-diff vs eager)")
    for c in report["cases"]:
        native_flag = "WRONG" if (c["aot_eager_diverges"] or c["inductor_diverges"]) else "ok"
        guard_flag = "guard-ok" if c["guarded_matches_eager"] else "GUARD-FAILED"
        print_fields(
            [
                (
                    f"shape={tuple(c['shape'])} seed={c['seed']}",
                    f"aot_eager_diff={c['max_abs_diff_aot_eager']:.3e}  "
                    f"inductor_diff={c['max_abs_diff_inductor']:.3e}  "
                    f"native={native_flag:6s}  {guard_flag}",
                )
            ]
        )


def _print_memory_budget_rng_report(report, *, no_color: bool) -> None:
    style = resolve_style(no_color_flag=no_color)
    print_fields([("torch version", report["torch_version"]), ("tracking issue", report["issue_url"])])

    if report["any_unguarded_rng_recompute_bug"]:
        print(status_headline(style, "fail", "torch.compile activation_memory_budget RNG-recompute gradient bug reproduced on this host (pytorch#190758)"))
    else:
        print(status_headline(style, "info", "no activation_memory_budget RNG-recompute divergence reproduced on this host's installed torch build"))

    if report["guard_fully_correct"]:
        print(status_headline(style, "ok", "safe_compile() prevents the divergence at every tested budget and restores the caller's budget"))
    else:
        print(status_headline(style, "fail", "safe_compile() did NOT prevent the divergence at at least one tested budget"))

    section("unguarded cases (budget -> actual gradient vs gradient implied by forward mask)")
    for c in report["unguarded_cases"]:
        flag = "ok" if c["matches"] else "WRONG-GRADIENT"
        print_fields([(f"budget={c['budget']}", flag)])

    section("guarded cases (requested budget -> guard result)")
    for c in report["guarded_cases"]:
        flag = "guard-ok" if c["matches"] else "GUARD-FAILED"
        print_fields([(f"requested_budget={c['budget']}", flag)])


def _print_mps_copy_dtype_report(report, *, no_color: bool) -> None:
    style = resolve_style(no_color_flag=no_color)
    print_fields(
        [
            ("torch version", report["torch_version"]),
            ("mps functional on this host", "yes" if report["mps_functional"] else "no (see notes below)"),
            ("tracking issue", report["issue_url"]),
        ]
    )

    if report["any_native_silently_wrong"]:
        print(status_headline(style, "fail", "MPS-to-CPU float64/complex128 silent copy bug reproduced on this host"))
    else:
        print(status_headline(style, "info", "no silent copy bug reproduced on this host's exercised devices"))

    if report["guard_fully_correct"]:
        print(status_headline(style, "ok", "safe_to()/safe_copy_() match the CPU-oracle conversion on every exercised device"))
    else:
        print(status_headline(style, "fail", "guard did NOT match the expected contract on at least one device"))

    section("per-dtype results (native .to()/.copy_() vs guarded vs CPU oracle)")
    for c in report["cases"]:
        if not c["ran"]:
            print_fields([(c["dtype_name"], f"skipped: {c['skip_reason']}")])
            continue
        to_flag = "SILENT-WRONG" if c["native_to_matches_oracle"] is False else "ok"
        copy_flag = "SILENT-WRONG" if c["native_copy_matches_oracle"] is False else "ok"
        guard_to_flag = "guard-ok" if c["guard_to_matches_oracle"] else "GUARD-FAILED"
        guard_copy_flag = "guard-ok" if c["guard_copy_matches_oracle"] else "GUARD-FAILED"
        print_fields(
            [
                (
                    c["dtype_name"],
                    f"native.to={to_flag:12s}  native.copy_={copy_flag:12s}  "
                    f"guard.to={guard_to_flag:12s}  guard.copy_={guard_copy_flag:12s}",
                )
            ]
        )


def _print_mps_linalg_stride_report(report, *, no_color: bool) -> None:
    style = resolve_style(no_color_flag=no_color)
    print_fields(
        [
            ("torch version", report["torch_version"]),
            ("mps functional on this host", "yes" if report["mps_functional"] else "no (see notes below)"),
            ("tracking issue", report["issue_url"]),
        ]
    )

    if report["any_native_layout_mismatch"]:
        print(status_headline(style, "fail", "MPS row-major vs CPU column-major layout mismatch reproduced on this host"))
    else:
        print(status_headline(style, "info", "no layout mismatch reproduced on this host's exercised devices"))

    if report["guard_fully_correct"]:
        print(status_headline(style, "ok", "safe_solve_triangular()/safe_cholesky_solve()/safe_solve() match the CPU-layout and CPU-value contract"))
    else:
        print(status_headline(style, "fail", "guard did NOT match the expected contract on at least one device"))

    section("per-op results (native vs guarded vs CPU-layout contract)")
    for c in report["cases"]:
        if not c["ran"]:
            print_fields([(c["op_name"], f"skipped: {c['skip_reason']}")])
            continue
        native_flag = "LAYOUT-MISMATCH" if c["native_layout_matches_cpu"] is False else "ok"
        guard_flag = "guard-ok" if c["guard_layout_matches_cpu"] and c["guard_values_match_cpu"] else "GUARD-FAILED"
        print_fields(
            [
                (
                    c["op_name"],
                    f"cpu.stride={tuple(c['cpu_stride'])!s:12s}  "
                    f"native.mps.stride={tuple(c['native_mps_stride'])!s:12s} ({native_flag})  "
                    f"guard.mps.stride={tuple(c['guard_mps_stride'])!s:12s} ({guard_flag})",
                )
            ]
        )



def _print_report(guard_name: str, report, *, no_color: bool) -> None:
    if guard_name == "as-strided-restride-oob":
        _print_as_strided_report(report, no_color=no_color)
    elif guard_name == "checkpoint-noise":
        _print_checkpoint_noise_report(report, no_color=no_color)
    elif guard_name == "compile-validation":
        _print_compile_validation_report(report, no_color=no_color)
    elif guard_name == "cpu-backward-nan-tail":
        _print_cpu_backward_nan_tail_report(report, no_color=no_color)
    elif guard_name == "dynamo-closure-descriptor":
        _print_dynamo_closure_descriptor_report(report, no_color=no_color)
    elif guard_name == "dtype-view-scatter":
        _print_dtype_view_scatter_report(report, no_color=no_color)
    elif guard_name == "duplicate-index-writeorder":
        _print_duplicate_index_writeorder_report(report, no_color=no_color)
    elif guard_name == "dynamic-clamp":
        _print_dynamic_clamp_report(report, no_color=no_color)
    elif guard_name == "equality-fusion":
        _print_equality_fusion_report(report, no_color=no_color)
    elif guard_name == "embeddingbag-freq-scale":
        _print_embeddingbag_freq_scale_report(report, no_color=no_color)
    elif guard_name == "expand-fill":
        _print_expand_fill_report(report, no_color=no_color)
    elif guard_name == "fp16-layernorm-tail":
        _print_fp16_layernorm_tail_report(report, no_color=no_color)
    elif guard_name == "full-dtype":
        _print_full_dtype_report(report, no_color=no_color)
    elif guard_name == "inplace-slice-shift-aliasing":
        _print_inplace_slice_shift_aliasing_report(report, no_color=no_color)
    elif guard_name == "int64-index-truncation":
        _print_int64_index_truncation_report(report, no_color=no_color)
    elif guard_name == "linalg-nan":
        _print_linalg_nan_report(report, no_color=no_color)
    elif guard_name == "linalg-pinv-complex-grad":
        _print_linalg_pinv_complex_grad_report(report, no_color=no_color)
    elif guard_name == "memory-budget-rng":
        _print_memory_budget_rng_report(report, no_color=no_color)
    elif guard_name == "mps-copy-dtype":
        _print_mps_copy_dtype_report(report, no_color=no_color)
    elif guard_name == "mps-linalg-stride":
        _print_mps_linalg_stride_report(report, no_color=no_color)
    elif guard_name == "scatter-copyback-alias":
        _print_scatter_copyback_alias_report(report, no_color=no_color)
    elif guard_name == "softmax-dim":
        _print_softmax_dim_report(report, no_color=no_color)
    elif guard_name == "normal-dtype-promotion":
        _print_normal_dtype_promotion_report(report, no_color=no_color)
    elif guard_name == "shuffle-sample-frozen":
        _print_shuffle_sample_frozen_report(report, no_color=no_color)
    elif guard_name == "std-precision":
        _print_std_precision_report(report, no_color=no_color)
    elif guard_name == "tiled-reduction-tail-store":
        _print_tiled_reduction_tail_store_report(report, no_color=no_color)
    elif guard_name == "transpose-argmin":
        _print_transpose_argmin_report(report, no_color=no_color)
    else:
        _print_addcdiv_report(report, no_color=no_color)


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(
        prog="torch-guard",
        description="Run consolidated PyTorch correctness diagnostics and call-site guard checks.",
    )
    parser.add_argument("--version", action="store_true", help="print version and exit")
    subparsers = parser.add_subparsers(dest="command")

    subparsers.add_parser("list", help="list available guards")

    run_parser = subparsers.add_parser("run", help="run a guard diagnostic")
    run_parser.add_argument("guard", choices=sorted(_GUARDS))
    run_parser.add_argument("--json", action="store_true", help="emit machine-readable JSON instead of text")
    run_parser.add_argument("--no-color", action="store_true", help="disable ANSI color even on a TTY")

    args = parser.parse_args(argv)

    if args.version:
        from . import __version__

        print(f"torch-correctness-guards {__version__}")
        return 0

    if args.command == "list":
        for name, meta in sorted(_GUARDS.items()):
            print(f"{name}\t{meta['description']}")
        return 0

    if args.command == "run":
        guard = _load_guard(args.guard)
        try:
            report = guard.diagnose()
        except guard.TorchUnavailableError as exc:
            if args.json:
                print(json.dumps({"guard": args.guard, "error": str(exc)}, indent=2))
            else:
                style = resolve_style(no_color_flag=args.no_color)
                print(status_headline(style, "fail", f"torch unavailable: {exc}"))
            return 2

        if args.json:
            payload = {"guard": args.guard, **report}
            print(json.dumps(payload, indent=2))
            return 0 if report.get("guard_fully_correct", report.get("guard_fully_effective")) else 1

        _print_report(args.guard, report, no_color=args.no_color)
        return 0 if report.get("guard_fully_correct", report.get("guard_fully_effective")) else 1

    parser.print_help()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
