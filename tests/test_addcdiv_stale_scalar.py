"""Regression tests for torch-addcdiv-stale-scalar-guard.

These prove:
  1. The bug is real and reproducible from scratch on this host's
     installed torch build: a hand-written Adam-style optimizer step
     that reads a step counter via .item() inside a helper function,
     then uses it in bias-correction arithmetic feeding addcdiv_'s
     value= argument, diverges from eager under
     torch.compile(backend="inductor") once the loop runs >= 3
     iterations, with the error compounding at higher iteration counts.
  2. make_safe_stale_scalar_step is an independently-verified fix: it
     matches eager's own output (the true oracle -- eager's semantics
     are the documented, correct behavior) even under torch.compile.
  3. A bug-injection-style check: the guard's underlying mechanism
     (torch.compiler.disable forcing eager execution of the
     scalar-dependent computation) is asserted to actually change the
     observed compiled result relative to the native (unguarded) call
     -- i.e. the guard is not a no-op that happens to already match by
     coincidence.
"""
from __future__ import annotations

import pytest

torch = pytest.importorskip("torch")

from torch_correctness_guards.guards.addcdiv_stale_scalar import (
    _adam_step_guarded,
    _adam_step_native,
    diagnose,
    make_safe_stale_scalar_step,
    safe_stale_scalar_step,
)


class TestNativeBugReproduction:
    def test_native_diverges_at_default_iteration_counts_on_this_host(self):
        # Not asserted unconditionally true forever: if a future torch
        # release backports pytorch/pytorch#195040 (merged on main
        # 2026-08-27/28, fixing the underlying stale float
        # specialization) into a released version, this docstring is
        # the record that the bug existed at the version noted in the
        # ledger/README. On torch 2.14.0 (this host, pinned before the
        # fix) it reproduces reliably at 3/6/10 iterations.
        report = diagnose(iteration_counts=(6,))
        assert report["any_native_diverges"] is True, (
            f"expected an eager-vs-compiled(inductor) stale-scalar "
            f"divergence on torch {report['torch_version']}; if this now "
            "fails, the bug may be fixed upstream "
            "(pytorch/pytorch#185382, fixed by #195040) -- update the "
            "README/ledger accordingly rather than treating this as a "
            "regression"
        )
        case = report["cases"][0]
        assert case["native_diverges"] is True
        assert case["max_abs_diff_native"] > 1e-6

    def test_divergence_requires_at_least_three_iterations(self):
        # The upstream issue's own documented trigger condition: the
        # bug does not manifest before the 3rd iteration (the first
        # compile establishes the cache entry; subsequent calls with a
        # changed scalar hit the poisoned cache).
        report = diagnose(iteration_counts=(1, 2, 3))
        one, two, three = report["cases"]
        assert one["native_diverges"] is False
        assert two["native_diverges"] is False
        assert three["native_diverges"] is True

    def test_error_compounds_with_more_iterations(self):
        report = diagnose(iteration_counts=(3, 10))
        small, large = report["cases"]
        assert large["max_abs_diff_native"] > small["max_abs_diff_native"]


class TestGuardMechanismIsNotACoincidentalNoOp:
    """Confirm the guard's compiled output genuinely differs from the
    native (unguarded) compiled output -- i.e. torch.compiler.disable
    is actually taking effect, not silently doing nothing."""

    def test_guarded_compiled_result_differs_from_native_compiled_result(self):
        report = diagnose(iteration_counts=(6,))
        case = report["cases"][0]
        assert case["guarded_final_param"] != case["native_final_param"], (
            "the guard produced the same (wrong) result as the native "
            "call -- torch.compiler.disable did not change behavior, "
            "meaning the guard mechanism itself is not working"
        )

    def test_guarded_matches_eager_where_native_does_not(self):
        report = diagnose(iteration_counts=(6,))
        case = report["cases"][0]
        assert case["guard_matches_eager"] is True
        assert case["native_diverges"] is True


class TestSafeStaleScalarStepMatchesEager:
    def test_safe_stale_scalar_step_matches_plain_eager_computation(self):
        step_t = torch.tensor(5.0, dtype=torch.float64)
        step, bc1, bc2, step_size = safe_stale_scalar_step(step_t, 0.9, 0.999, 0.001)
        assert step == 5.0
        assert bc1 == pytest.approx(1 - 0.9 ** 5.0)
        assert bc2 == pytest.approx(1 - 0.999 ** 5.0)
        assert step_size == pytest.approx(0.001 / (1 - 0.9 ** 5.0))

    def test_make_safe_stale_scalar_step_returns_callable_bound_to_torch_module(self):
        safe_fn = make_safe_stale_scalar_step(torch)
        step_t = torch.tensor(2.0, dtype=torch.float64)
        step, bc1, bc2, step_size = safe_fn(step_t, 0.9, 0.999, 0.01)
        assert step == 2.0
        assert step_size == pytest.approx(0.01 / (1 - 0.9 ** 2.0))

    def test_safe_stale_scalar_step_works_under_compile_without_error(self):
        def f(step_t, safe_fn):
            return safe_fn(step_t, 0.9, 0.999, 0.001)[3]

        safe_fn = make_safe_stale_scalar_step(torch)
        torch._dynamo.reset()
        compiled = torch.compile(f, fullgraph=False)
        step_t = torch.tensor(3.0, dtype=torch.float64)
        result = compiled(step_t, safe_fn)
        expected = 0.001 / (1 - 0.9 ** 3.0)
        assert result == pytest.approx(expected)


class TestDiagnose:
    def test_diagnose_default_runs_and_reports_consistent_structure(self):
        report = diagnose(iteration_counts=(3, 6))
        assert len(report["cases"]) == 2
        assert isinstance(report["torch_version"], str)
        assert report["issue_url"] == "https://github.com/pytorch/pytorch/issues/185382"
        assert report["upstream_fix_pr"] == "https://github.com/pytorch/pytorch/pull/195040"

    def test_guard_fully_correct_flag_is_true(self):
        report = diagnose(iteration_counts=(3, 6, 10))
        assert report["guard_fully_correct"] is True

    def test_each_case_has_matching_lengths(self):
        report = diagnose(iteration_counts=(4,))
        case = report["cases"][0]
        assert len(case["eager_final_param"]) == 4
        assert len(case["native_final_param"]) == 4
        assert len(case["guarded_final_param"]) == 4


class TestAdamStepHelpersDirectlyInEagerMode:
    """Exercise _adam_step_native/_adam_step_guarded's own statements
    directly, without torch.compile.

    diagnose()'s existing tests only invoke these two functions through
    torch.compile(backend="inductor"): Dynamo traces and re-emits the
    function body into a compiled graph, so coverage.py's line tracer
    never sees the *source* lines of _adam_step_native/_adam_step_guarded
    execute -- only the compiled kernel runs. That left those functions'
    own statements (this class's target) unproven independent of the
    inductor-specific compiled path, and unrecorded by coverage tooling.
    Calling them directly in eager mode closes that gap and independently
    confirms the two step functions are equivalent in eager execution
    (the compiled-vs-eager divergence proven elsewhere is specifically an
    inductor-compilation artifact, not a difference between these two
    hand-written functions themselves).
    """

    def _fresh_state(self, seed: int):
        torch.manual_seed(seed)
        p0 = torch.randn(4, dtype=torch.float64)
        g = torch.randn(4, dtype=torch.float64)
        return (
            p0.clone(),
            g,
            torch.zeros(4, dtype=torch.float64),
            torch.zeros(4, dtype=torch.float64),
            torch.zeros((), dtype=torch.float64),
        )

    def test_native_and_guarded_step_functions_agree_in_eager_mode(self):
        # In eager mode there is no stale-scalar-cache bug to trigger, so
        # the native and guarded helpers must produce identical results
        # step-for-step -- proving the guard changes nothing about
        # eager semantics, only compiled semantics.
        p_native, g, m_native, v_native, st_native = self._fresh_state(seed=7)
        p_guarded, _, m_guarded, v_guarded, st_guarded = self._fresh_state(seed=7)
        safe_fn = make_safe_stale_scalar_step(torch)

        for _ in range(5):
            _adam_step_native(torch, p_native, g, m_native, v_native, st_native)
            _adam_step_guarded(
                torch, safe_fn, p_guarded, g, m_guarded, v_guarded, st_guarded
            )

        assert torch.allclose(p_native, p_guarded, atol=1e-12)
        assert st_native.item() == st_guarded.item() == 5.0

    def test_guarded_step_updates_step_counter_and_moments(self):
        p, g, m, v, st = self._fresh_state(seed=11)
        safe_fn = make_safe_stale_scalar_step(torch)
        m_before, v_before = m.clone(), v.clone()

        result = _adam_step_guarded(torch, safe_fn, p, g, m, v, st)

        assert result is p
        assert st.item() == 1.0
        assert not torch.equal(m, m_before)
        assert not torch.equal(v, v_before)

    def test_native_step_updates_step_counter_and_moments(self):
        p, g, m, v, st = self._fresh_state(seed=13)
        m_before, v_before = m.clone(), v.clone()

        result = _adam_step_native(torch, p, g, m, v, st)

        assert result is p
        assert st.item() == 1.0
        assert not torch.equal(m, m_before)
        assert not torch.equal(v, v_before)
