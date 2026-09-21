"""Regression tests for torch-compile-std-precision-guard.

These prove:
  1. The bug is real and reproducible from scratch on this host's
     installed torch build: torch.compile(backend="inductor") diverges
     from eager for torch.std on large-magnitude input (inf vs finite)
     and silently produces an all-zero gradient for small-magnitude
     input where eager's gradient is nonzero.
  2. safe_std/safe_var/safe_var_mean/safe_std_mean are independently
     verified fixes: they match eager's own output (an independent
     oracle -- eager already accumulates in double, matching a float64
     reference) whether called directly or under torch.compile.
  3. A bug-injection test proves the divergence-detection logic itself
     is non-tautological: a version of "divergence" that only checks
     finite-vs-non-finite (not magnitude collapse) misses the real
     exp=-30 case where compiled silently flushes to exactly 0.0 while
     technically remaining "finite" -- this is asserted directly
     against the shipped diagnose() logic to guard against a future
     regression re-introducing that blind spot.
"""
from __future__ import annotations

import pytest

torch = pytest.importorskip("torch")

from torch_correctness_guards.guards.std_precision import (
    diagnose,
    safe_std,
    safe_var,
    safe_var_mean,
    safe_std_mean,
)


class TestNativeBugReproduction:
    def test_std_diverges_for_some_magnitude_on_this_host(self):
        # Not asserted unconditionally true forever: if a future torch
        # release fixes Inductor's accumulation precision, this
        # docstring is the record that the bug existed at the version
        # noted in the ledger/README. On torch 2.14.0 (this host) it
        # reproduces reliably across large and small magnitudes.
        report = diagnose(magnitude_exponents=(-30, 0, 30))
        assert report["any_std_divergence"] is True, (
            f"expected a std eager-vs-compiled divergence on torch "
            f"{report['torch_version']}; if this now fails, the bug "
            "may be fixed upstream (pytorch/pytorch#197089) -- update "
            "the README/ledger accordingly rather than treating this "
            "as a regression"
        )

    def test_silent_zero_gradient_reproduces_on_this_host(self):
        report = diagnose(magnitude_exponents=(-30, 0, 30))
        assert report["any_silent_zero_gradient"] is True, (
            "expected the silent all-zero-gradient case for small-"
            f"magnitude input on torch {report['torch_version']}"
        )

    def test_large_magnitude_case_is_finite_vs_inf(self):
        report = diagnose(magnitude_exponents=(30,))
        case = report["std_cases"][0]
        assert case["eager_is_finite"] is True
        assert case["compiled_is_finite"] is False
        assert case["diverges"] is True

    def test_small_magnitude_case_is_silent_flush_to_zero(self):
        # This is the subtle case: BOTH eager and compiled report a
        # "finite" number, but compiled has silently collapsed to
        # exactly 0.0 while eager retains the true nonzero value. A
        # naive finite-vs-non-finite divergence check would miss this
        # entirely -- assert the shipped diagnose() catches it anyway.
        report = diagnose(magnitude_exponents=(-30,))
        case = report["std_cases"][0]
        assert case["eager_is_finite"] is True
        assert case["compiled_is_finite"] is True
        assert case["compiled_value"] == 0.0
        assert case["eager_value"] != 0.0
        assert case["diverges"] is True, (
            "diagnose() must flag this as a divergence even though both "
            "sides are 'finite' -- a silent flush-to-zero is just as "
            "real a correctness bug as an inf/NaN mismatch"
        )


class TestBugInjectionNonTautological:
    """Prove the divergence-detection tests above are not tautological
    by re-running the exact finite-vs-non-finite-only check that an
    earlier draft of this module used, and confirming it WOULD miss
    the exp=-30 silent-flush-to-zero case that the shipped
    diverges-with-magnitude check correctly catches."""

    def test_naive_finite_only_check_would_miss_the_flush_to_zero_bug(self):
        report = diagnose(magnitude_exponents=(-30,))
        case = report["std_cases"][0]
        naive_diverges = case["eager_is_finite"] != case["compiled_is_finite"]
        assert naive_diverges is False, (
            "sanity check: the naive finite-vs-finite comparison really "
            "would report 'no divergence' for this case (both sides "
            "finite), which is exactly why diagnose() also checks "
            "magnitude, not just finiteness"
        )
        assert case["diverges"] is True, (
            "the shipped diagnose() must still catch it via the "
            "magnitude-divergence check"
        )


class TestSafeStdMatchesEager:
    @pytest.mark.parametrize("exponent", [-30, -15, 0, 15, 30])
    def test_safe_std_matches_eager_uncompiled(self, exponent):
        torch.manual_seed(exponent + 100)
        x = torch.randn(4, 8) * (10.0 ** exponent)
        eager = torch.std(x)
        guarded = safe_std(x)
        if torch.isfinite(eager):
            assert torch.isclose(guarded, eager, rtol=1e-5, atol=0.0) or (
                guarded.item() == eager.item()
            )
        else:
            # both should agree on being non-finite
            assert not torch.isfinite(guarded)

    @pytest.mark.parametrize("exponent", [-30, 0, 30])
    def test_safe_std_matches_eager_under_compile(self, exponent):
        torch.manual_seed(exponent + 200)
        x = torch.randn(4, 8) * (10.0 ** exponent)
        eager = torch.std(x)

        torch._dynamo.reset()
        compiled_guard = torch.compile(safe_std)
        guarded = compiled_guard(x)

        if torch.isfinite(eager):
            rel_err = abs(guarded.item() - eager.item()) / max(abs(eager.item()), 1e-300)
            assert rel_err < 1e-3, (
                f"exponent={exponent}: guarded={guarded.item()} vs "
                f"eager={eager.item()}, rel_err={rel_err}"
            )
        else:
            assert not torch.isfinite(guarded)

    def test_safe_std_preserves_gradient_under_compile(self):
        torch.manual_seed(42)
        x = torch.randn(5) * 1e-30

        def grads(fn, xin):
            xx = xin.clone().requires_grad_(True)
            y = fn(xx)
            (g,) = torch.autograd.grad(y, xx)
            return y, g

        eager_y, eager_g = grads(torch.std, x)

        torch._dynamo.reset()
        compiled_guard = torch.compile(safe_std)
        guard_y, guard_g = grads(compiled_guard, x)

        assert not torch.all(eager_g == 0), "sanity: eager gradient must be genuinely nonzero"
        assert not torch.all(guard_g == 0), (
            "safe_std under torch.compile produced an all-zero gradient -- "
            "the guard failed to avoid the bug it targets"
        )
        assert torch.allclose(guard_g, eager_g, rtol=1e-3, atol=1e-38)


class TestSafeVarVariants:
    def test_safe_var_mean_matches_eager(self):
        torch.manual_seed(7)
        x = torch.randn(4, 8) * 1e15
        ev, em = torch.var_mean(x)
        gv, gm = safe_var_mean(x)
        assert torch.isclose(gv, ev, rtol=1e-5)
        assert torch.isclose(gm, em, rtol=1e-5)

    def test_safe_std_mean_matches_eager(self):
        torch.manual_seed(8)
        x = torch.randn(4, 8) * 1e15
        es, em = torch.std_mean(x)
        gs, gm = safe_std_mean(x)
        assert torch.isclose(gs, es, rtol=1e-5)
        assert torch.isclose(gm, em, rtol=1e-5)

    def test_safe_var_matches_eager_ordinary_magnitude(self):
        torch.manual_seed(9)
        x = torch.randn(4, 8) * 5.0
        ev = torch.var(x)
        gv = safe_var(x)
        assert torch.isclose(gv, ev, rtol=1e-5)

    def test_safe_std_dim_wise_matches_eager(self):
        torch.manual_seed(10)
        x = torch.randn(3, 4, 5) * 1e15
        eager = torch.std(x, dim=1)
        guarded = safe_std(x, dim=1)
        assert torch.allclose(guarded, eager, rtol=1e-5)

    def test_safe_var_dim_wise_matches_eager(self):
        """safe_var's dim= branch (core.py line ~108) was previously
        untested -- only the dim=None path had coverage. Extreme
        magnitude input is used so a broken dim-wise implementation
        (e.g. one that silently fell back to the buggy native
        torch.var(x, dim=...) instead of the float64-upcast path) would
        actually show a divergence, not pass by coincidence."""
        torch.manual_seed(11)
        x = torch.randn(3, 4, 5) * 1e15
        eager = torch.var(x, dim=1)
        guarded = safe_var(x, dim=1)
        assert torch.allclose(guarded, eager, rtol=1e-5)

    def test_safe_var_mean_dim_wise_matches_eager(self):
        """safe_var_mean's dim= branch (core.py line ~120) was previously
        untested -- only the dim=None path had coverage."""
        torch.manual_seed(12)
        x = torch.randn(3, 4, 5) * 1e15
        ev, em = torch.var_mean(x, dim=1)
        gv, gm = safe_var_mean(x, dim=1)
        assert torch.allclose(gv, ev, rtol=1e-5)
        assert torch.allclose(gm, em, rtol=1e-5)

    def test_safe_std_mean_dim_wise_matches_eager(self):
        """safe_std_mean's dim= branch (core.py line ~132) was previously
        untested -- only the dim=None path had coverage."""
        torch.manual_seed(13)
        x = torch.randn(3, 4, 5) * 1e15
        es, em = torch.std_mean(x, dim=1)
        gs, gm = safe_std_mean(x, dim=1)
        assert torch.allclose(gs, es, rtol=1e-5)
        assert torch.allclose(gm, em, rtol=1e-5)


class TestDiagnose:
    def test_diagnose_runs_and_reports_consistent_structure(self):
        report = diagnose(magnitude_exponents=(-15, 0, 15))
        assert len(report["std_cases"]) == 3
        assert len(report["var_cases"]) == 3
        assert len(report["zero_gradient_cases"]) == 3
        assert isinstance(report["torch_version"], str)

    def test_guard_fully_correct_flag_is_true(self):
        report = diagnose()
        assert report["guard_fully_correct"] is True
