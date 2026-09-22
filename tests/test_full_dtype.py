"""Regression tests for torch-inductor-full-dtype-guard.

These prove:
  1. The bug is real and reproducible from scratch on this host's
     installed torch build: torch.compile(backend="inductor") produces
     a different result than eager for torch.full(size, symbolic_fill,
     dtype=torch.bool).sum(), and silently skips the int8 overflow
     check eager raises for an out-of-range symbolic fill.
  2. safe_full is an independently-verified fix: it matches eager's
     own output (the true oracle here -- eager's cast/overflow
     semantics are the documented, correct behavior) whether called
     directly or under torch.compile, for both the bool-mask case and
     the int8-overflow case.
  3. A bug-injection-style check: the guard's underlying mechanism
     (torch.compiler.disable forcing eager execution of the fill) is
     asserted to actually change the observed compiled result relative
     to the native (unguarded) call -- i.e. the guard is not a no-op
     that happens to already match by coincidence.
"""
from __future__ import annotations

import pytest

torch = pytest.importorskip("torch")

from torch_correctness_guards.guards.full_dtype import diagnose, safe_full


class TestNativeBugReproduction:
    def test_bool_fill_diverges_for_some_fill_value_on_this_host(self):
        # Not asserted unconditionally true forever: if a future torch
        # release fixes Inductor's SymPyOps.to_dtype for symbolic
        # fills, this docstring is the record that the bug existed at
        # the version noted in the ledger/README. On torch 2.14.0
        # (this host) it reproduces reliably for fill=3.
        report = diagnose(bool_fill_values=(3,))
        assert report["any_bool_fill_divergence"] is True, (
            f"expected a bool-fill eager-vs-compiled divergence on torch "
            f"{report['torch_version']}; if this now fails, the bug may "
            "be fixed upstream (pytorch/pytorch#194062) -- update the "
            "README/ledger accordingly rather than treating this as a "
            "regression"
        )
        case = report["bool_fill_cases"][0]
        assert case["eager_result"] == 2
        assert case["compiled_native_result"] == 6

    def test_int8_overflow_check_silently_skipped_under_compile(self):
        report = diagnose(int8_overflow_fill_values=(300,))
        assert report["any_int8_silent_overflow"] is True, (
            "expected eager to raise on int8 fill=300 (overflow) while "
            f"compiled native silently succeeds, on torch {report['torch_version']}"
        )
        case = report["int8_overflow_cases"][0]
        assert case["eager_raised"] is True
        assert case["compiled_native_raised"] is False


class TestGuardMechanismIsNotACoincidentalNoOp:
    """Confirm the guard's compiled output genuinely differs from the
    native (unguarded) compiled output -- i.e. torch.compiler.disable
    is actually taking effect, not silently doing nothing."""

    def test_guarded_compiled_result_differs_from_native_compiled_result(self):
        report = diagnose(bool_fill_values=(3,))
        case = report["bool_fill_cases"][0]
        assert case["compiled_guarded_result"] != case["compiled_native_result"], (
            "the guard produced the same (wrong) result as the native "
            "call -- torch.compiler.disable did not change behavior, "
            "meaning the guard mechanism itself is not working"
        )
        assert case["compiled_guarded_result"] == case["eager_result"]

    def test_guarded_compiled_raises_where_native_does_not(self):
        report = diagnose(int8_overflow_fill_values=(300,))
        case = report["int8_overflow_cases"][0]
        assert case["compiled_guarded_raised"] != case["compiled_native_raised"], (
            "the guard produced the same (silently-wrong, non-raising) "
            "behavior as the native call -- the guard mechanism is not "
            "actually forcing eager execution"
        )
        assert case["compiled_guarded_raised"] == case["eager_raised"]


class TestSafeFullMatchesEager:
    @pytest.mark.parametrize("fill", [0, 1, 2, 3, 7, 100])
    def test_safe_full_bool_matches_eager_uncompiled(self, fill):
        eager = torch.full((3,), fill, dtype=torch.bool)
        guarded = safe_full((3,), fill, dtype=torch.bool)
        assert torch.equal(guarded, eager)

    @pytest.mark.parametrize("fill", [0, 1, 2, 3, 7])
    def test_safe_full_bool_matches_eager_under_compile(self, fill):
        def f_guarded(x, fill=fill):
            return safe_full((2,), fill, dtype=torch.bool).sum()

        def f_eager(fill=fill):
            return torch.full((2,), fill, dtype=torch.bool).sum()

        torch._dynamo.reset()
        compiled_guarded = torch.compile(f_guarded, fullgraph=False)
        guarded_val = compiled_guarded(torch.tensor(0)).item()
        eager_val = f_eager().item()
        assert guarded_val == eager_val

    def test_safe_full_int8_overflow_raises_under_compile_like_eager(self):
        def g_guarded(x):
            return safe_full((2,), x.item(), dtype=torch.int8)

        with pytest.raises(RuntimeError):
            torch.full((2,), 300, dtype=torch.int8)

        torch._dynamo.reset()
        compiled_guarded = torch.compile(g_guarded, fullgraph=False)
        with pytest.raises(RuntimeError):
            compiled_guarded(torch.tensor(300))

    def test_safe_full_ordinary_constant_fill_matches_eager(self):
        # Sanity: the guard must not break the ordinary, already-correct
        # constant-fill case.
        eager = torch.full((4, 4), 2.5, dtype=torch.float32)
        guarded = safe_full((4, 4), 2.5, dtype=torch.float32)
        assert torch.equal(guarded, eager)

    def test_safe_full_passes_through_extra_kwargs(self):
        eager = torch.full((2, 2), 1, dtype=torch.int64, requires_grad=False)
        guarded = safe_full((2, 2), 1, dtype=torch.int64, requires_grad=False)
        assert torch.equal(guarded, eager)
        assert guarded.dtype == torch.int64


class TestExtraOverflowDtypes:
    """2026-09-19: investigated whether the missing-cast overflow bug
    generalizes beyond int8 to other narrow integer dtypes (int16, uint8).

    Real finding: it is NUMPY-PROCESS-STATE DEPENDENT, not a universal
    Inductor defect like the int8 case. In a clean venv with only
    `.[dev,torch]` installed (matching this repo's actual declared deps
    and this CI workflow's install step -- no numpy), torch.compile
    correctly raises for symbolic int16/uint8 overflow fills, matching
    eager: NO bug. But if numpy happens to be importable in the same
    process (observed on this macOS host's system Python, which has
    numpy 2.5.2 on sys.path for unrelated reasons), the exact same
    int16/uint8 fills DO silently skip the overflow check exactly like
    int8 always does. Confirmed by isolating both environments directly
    (venv without numpy vs. system Python with numpy) rather than
    inferring this from one run.

    Given that dependency, the regression coverage here intentionally
    does NOT assert the buggy outcome unconditionally -- doing so would
    make this suite non-deterministic across CI runners depending on
    whichever transitive/incidental packages happen to be importable.
    What IS asserted, and IS environment-independent: (1) the widened
    report shape always includes int16/uint8 cases, and (2)
    safe_full()'s guard matches eager in either environment state,
    because torch.compiler.disable forces the real eager overflow
    check regardless of whether numpy is present -- the guard's
    correctness does not depend on this hazard, only the NATIVE
    (unguarded) bug's presence does.
    """

    def test_diagnose_default_includes_int16_and_uint8_overflow_cases(self):
        report = diagnose(bool_fill_values=(3,), int8_overflow_fill_values=(300,))
        dtypes_seen = {c["overflow_dtype"] for c in report["extra_overflow_dtype_cases"]}
        assert dtypes_seen == {"int16", "uint8"}

    def test_int16_overflow_guard_matches_eager_regardless_of_native_outcome(self):
        report = diagnose(
            bool_fill_values=(3,),
            int8_overflow_fill_values=(300,),
            extra_overflow_dtype_cases=[("int16", 40000)],
        )
        case = report["extra_overflow_dtype_cases"][0]
        assert case["overflow_dtype"] == "int16"
        assert case["eager_raised"] is True
        # native_silently_wrong is numpy-process-state dependent (see class
        # docstring) -- not asserted here. The guard's correctness is not:
        assert case["guard_matches_eager"] is True
        assert case["compiled_guarded_raised"] == case["eager_raised"]

    def test_uint8_overflow_guard_matches_eager_regardless_of_native_outcome(self):
        report = diagnose(
            bool_fill_values=(3,),
            int8_overflow_fill_values=(300,),
            extra_overflow_dtype_cases=[("uint8", 300)],
        )
        case = report["extra_overflow_dtype_cases"][0]
        assert case["overflow_dtype"] == "uint8"
        assert case["guard_matches_eager"] is True
        assert case["compiled_guarded_raised"] == case["eager_raised"]

    def test_any_overflow_dtype_silent_overflow_flag_is_well_defined(self):
        # Whatever this host's numpy-process-state hazard resolves to, the
        # flag must be a plain bool and must be consistent with the int8
        # case (which IS a universal, numpy-independent bug) always being
        # counted: since int8 is included by default, this must be True.
        report = diagnose(bool_fill_values=(3,), int8_overflow_fill_values=(300,))
        assert isinstance(report["any_overflow_dtype_silent_overflow"], bool)
        assert report["any_overflow_dtype_silent_overflow"] is True

    def test_guard_fully_correct_still_true_with_widened_dtype_sweep(self):
        report = diagnose()
        assert report["guard_fully_correct"] is True


class TestDiagnose:
    def test_diagnose_runs_and_reports_consistent_structure(self):
        report = diagnose(bool_fill_values=(1, 3), int8_overflow_fill_values=(300,))
        assert len(report["bool_fill_cases"]) == 2
        assert len(report["int8_overflow_cases"]) == 1
        assert isinstance(report["torch_version"], str)
        assert report["issue_urls"] == ["https://github.com/pytorch/pytorch/issues/194062"]

    def test_guard_fully_correct_flag_is_true(self):
        report = diagnose()
        assert report["guard_fully_correct"] is True
