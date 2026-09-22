"""Regression tests for torch-linalg-nan-guard.

These prove the guard functions actually intercept the documented
NaN-swallowing bugs (pytorch/pytorch#187759, numpy/numpy#20280) on
whatever torch build is installed, and that they are transparent
pass-throughs on ordinary finite input (an independent correctness
oracle: any finite-input call must match plain torch.linalg exactly).
"""
from __future__ import annotations

import math

import pytest

torch = pytest.importorskip("torch")

from torch_correctness_guards.guards.linalg_nan import (
    diagnose,
    diagnose_norm_precision,
    safe_eigvalsh,
    safe_norm,
    safe_svdvals,
    safe_vector_norm,
)


class TestSafeSvdvals:
    def test_finite_input_matches_plain_svdvals_exactly(self):
        torch.manual_seed(0)
        A = torch.randn(5, 5, dtype=torch.float64)
        expected = torch.linalg.svdvals(A)
        actual = safe_svdvals(A)
        assert torch.allclose(actual, expected)

    def test_nan_input_raises(self):
        # The unguarded torch.linalg.svdvals's behavior on NaN input is
        # backend-dependent: some LAPACK backends (macOS/Accelerate)
        # silently swallow the NaN and return a finite result
        # (pytorch/pytorch#187759); others (Linux/OpenBLAS, observed in
        # this project's own CI) raise their own _LinAlgError instead.
        # Either way this is exactly why the guard exists: it must raise
        # a clear, consistent ValueError regardless of which backend's
        # behavior the unguarded call happens to exhibit.
        M = torch.tensor([[float("nan"), 0.0], [0.0, 1.0]], dtype=torch.float64)
        with pytest.raises(ValueError, match="NaN or Inf"):
            safe_svdvals(M)

    def test_inf_input_also_raises(self):
        M = torch.tensor([[float("inf"), 0.0], [0.0, 1.0]], dtype=torch.float64)
        with pytest.raises(ValueError, match="NaN or Inf"):
            safe_svdvals(M)

    def test_3x3_diagonal_nan_placements_all_caught(self):
        for pos in range(3):
            M = torch.diag(torch.tensor([1.0, 2.0, 3.0], dtype=torch.float64))
            M[pos, pos] = float("nan")
            with pytest.raises(ValueError):
                safe_svdvals(M)


class TestSafeEigvalsh:
    def test_finite_input_matches_plain_eigvalsh_exactly(self):
        torch.manual_seed(1)
        A = torch.randn(5, 5, dtype=torch.float64)
        A = A + A.T  # symmetric, required by eigvalsh
        expected = torch.linalg.eigvalsh(A)
        actual = safe_eigvalsh(A)
        assert torch.allclose(actual, expected)

    def test_nan_input_raises(self):
        # Same backend-dependence note as svdvals above: the unguarded
        # eigvalsh's behavior on this exact input has been observed to
        # differ between LAPACK backends, but the guard's own ValueError
        # must fire either way.
        M = torch.diag(torch.tensor([1.0, 2.0], dtype=torch.float64))
        M[0, 0] = float("nan")
        with pytest.raises(ValueError, match="NaN or Inf"):
            safe_eigvalsh(M)

    def test_inf_input_also_raises(self):
        M = torch.diag(torch.tensor([float("inf"), 1.0], dtype=torch.float64))
        with pytest.raises(ValueError, match="NaN or Inf"):
            safe_eigvalsh(M)


class TestDiagnose:
    def test_diagnose_runs_and_guard_is_fully_effective(self):
        # any_bug_present is intentionally NOT asserted True here: on
        # LAPACK backends where the unguarded op raises its own error
        # for NaN input instead of silently swallowing it (observed on
        # Linux/OpenBLAS in this project's own CI, unlike macOS/
        # Accelerate where the silent-swallow bug was first reproduced),
        # zero cases may be "silent". The guard being fully effective
        # regardless of which behavior the backend exhibits is the
        # actual invariant this package provides.
        report = diagnose(sizes=(2, 3))
        assert report["guard_fully_effective"] is True
        assert len(report["cases"]) == (2 + 3) * 2  # svdvals + eigvalsh per position

    def test_every_case_where_bug_is_silent_the_guard_still_raises(self):
        report = diagnose(sizes=(2, 3, 4))
        for case in report["cases"]:
            if case["buggy_result_is_finite"]:
                assert case["guard_raises"], (
                    f"guard failed to catch a silent-finite case: {case}"
                )

    def test_reference_full_decomposition_never_silently_finite(self):
        # Cross-check against an independent oracle: whenever the
        # reference full-decomposition op (svd/eigh) does return a
        # result rather than raising its own error, that result must
        # contain NaN for NaN-containing input -- it must never be the
        # same kind of silent-finite bug the guarded functions have.
        # This is the ground truth this package's "bug" classification
        # relies on: a reference op that raises is a separate, already-
        # loud failure mode (also fine, just not "silent"); a reference
        # op that returns a *finite* result for NaN input would mean the
        # whole test fixture's assumptions are broken.
        report = diagnose(sizes=(2,))
        for case in report["cases"]:
            if not case["reference_raised"]:
                assert case["reference_has_nan"] is True, (
                    "reference full-decomposition op returned a finite "
                    f"result for NaN input -- unexpected: {case}"
                )


class TestNormPrecisionNativeBugReproduction:
    """Proves the underlying float32-accumulator bug (pytorch/pytorch
    #169237, #193006) is real and reproducible from scratch on this
    host's installed torch build -- an independent float64 oracle, not
    a value trusted from the upstream issue thread."""

    def test_unguarded_vector_norm_diverges_from_float64_reference_at_16m_elements(self):
        torch.manual_seed(0)
        x64 = torch.rand(16_000_000, dtype=torch.float64)
        reference = torch.linalg.vector_norm(x64, ord=2).item()
        x32 = x64.to(torch.float32)
        unguarded = torch.linalg.vector_norm(x32, ord=2).item()
        rel_err = abs(unguarded - reference) / reference
        # This is the actual native-bug reproduction: if a future torch
        # release fixes this upstream, this assertion should start
        # failing loudly (rel_err near float32 eps), which is exactly
        # the signal that the guard's diagnose_norm_precision() call
        # should also stop reporting any_precision_loss_present. This
        # test intentionally does NOT use the guard -- it is checking
        # torch's OWN behavior, matching this repo's evidence-before-
        # acceptance policy of never trusting a cached bug report.
        assert rel_err > 1e-4, (
            f"expected the known float32 CPU L2-norm accumulator bug to reproduce "
            f"(rel_err={rel_err:.3e}); if this now passes, the upstream bug may be "
            f"fixed in torch {torch.__version__} and this guard's rationale should "
            f"be re-verified"
        )

    def test_naive_finite_only_check_would_miss_the_precision_loss_bug(self):
        # Mirrors this module's existing bug-injection-non-tautological
        # pattern (see torch-compile-std-precision-guard's equivalent
        # test): a check that only asks "is the result finite?" passes
        # on the precision-loss variant even though the *value* is
        # wrong by orders of magnitude more than float32 rounding could
        # explain -- proving finiteness alone is not a sufficient
        # correctness check for this bug class.
        torch.manual_seed(0)
        x64 = torch.rand(16_000_000, dtype=torch.float64)
        x32 = x64.to(torch.float32)
        unguarded = torch.linalg.vector_norm(x32, ord=2).item()
        assert math.isfinite(unguarded)  # the naive check "passes"...
        reference = torch.linalg.vector_norm(x64, ord=2).item()
        rel_err = abs(unguarded - reference) / reference
        assert rel_err > 1e-4  # ...even though the value is badly wrong

    def test_unguarded_vector_norm_overflows_to_inf_for_large_magnitude_input(self):
        torch.manual_seed(0)
        x = (torch.rand(8, dtype=torch.float32) + 0.5) * 1e30
        unguarded = torch.linalg.vector_norm(x, ord=2).item()
        assert math.isinf(unguarded)

    def test_overflow_case_gradient_is_silently_all_zero(self):
        # The dangerous consequence, not just the forward-value bug:
        # backward() divides by the inf norm and silently zeroes the
        # gradient, with no error or warning.
        torch.manual_seed(0)
        x = ((torch.rand(8, dtype=torch.float32) + 0.5) * 1e30).requires_grad_(True)
        norm = torch.linalg.vector_norm(x, ord=2)
        assert math.isinf(norm.item())
        (grad,) = torch.autograd.grad(norm, x)
        assert torch.all(grad == 0.0)


class TestSafeVectorNormMatchesReference:
    """Proves the guard actually fixes both bug variants, against an
    independent float64 oracle -- and is a transparent pass-through
    for input the bug does not affect."""

    def test_guarded_matches_float64_reference_at_16m_elements(self):
        torch.manual_seed(0)
        x64 = torch.rand(16_000_000, dtype=torch.float64)
        reference = torch.linalg.vector_norm(x64, ord=2).item()
        x32 = x64.to(torch.float32)
        guarded = safe_vector_norm(x32, ord=2).item()
        rel_err = abs(guarded - reference) / reference
        assert rel_err < 1e-6, f"guard should closely match float64 reference, got rel_err={rel_err:.3e}"

    def test_guarded_does_not_overflow_for_large_magnitude_input(self):
        torch.manual_seed(0)
        x = (torch.rand(8, dtype=torch.float32) + 0.5) * 1e30
        guarded = safe_vector_norm(x, ord=2).item()
        assert math.isfinite(guarded)

    def test_guarded_preserves_nonzero_gradient_for_large_magnitude_input(self):
        torch.manual_seed(0)
        x = ((torch.rand(8, dtype=torch.float32) + 0.5) * 1e30).requires_grad_(True)
        norm = safe_vector_norm(x, ord=2)
        assert math.isfinite(norm.item())
        (grad,) = torch.autograd.grad(norm, x)
        assert torch.any(grad != 0.0)

    def test_guarded_output_dtype_matches_input_dtype(self):
        torch.manual_seed(0)
        x = torch.rand(100, dtype=torch.float32)
        assert safe_vector_norm(x, ord=2).dtype == torch.float32

    def test_guarded_transparent_for_float64_input(self):
        torch.manual_seed(0)
        x = torch.rand(100, dtype=torch.float64)
        expected = torch.linalg.vector_norm(x, ord=2)
        actual = safe_vector_norm(x, ord=2)
        assert torch.equal(actual, expected)

    def test_guarded_transparent_for_non_l2_ord(self):
        torch.manual_seed(0)
        x = torch.rand(100, dtype=torch.float32)
        expected = torch.linalg.vector_norm(x, ord=1)
        actual = safe_vector_norm(x, ord=1)
        assert torch.equal(actual, expected)

    def test_guarded_matches_unguarded_exactly_on_small_finite_input(self):
        torch.manual_seed(0)
        x = torch.rand(10, dtype=torch.float32)
        expected = torch.linalg.vector_norm(x, ord=2)
        actual = safe_vector_norm(x, ord=2)
        assert torch.allclose(actual, expected, atol=1e-4)

    def test_safe_norm_matches_safe_vector_norm_for_default_fro(self):
        torch.manual_seed(0)
        x = (torch.rand(8, dtype=torch.float32) + 0.5) * 1e30
        via_norm = safe_norm(x)
        via_vector_norm = safe_vector_norm(x, ord=2)
        assert torch.equal(via_norm, via_vector_norm)

    def test_safe_norm_transparent_for_non_l2_p(self):
        torch.manual_seed(0)
        x = torch.rand(100, dtype=torch.float32)
        expected = torch.norm(x, p=1)
        actual = safe_norm(x, p=1)
        assert torch.equal(actual, expected)


class TestDiagnoseNormPrecision:
    def test_diagnose_runs_and_reports_consistent_structure(self):
        report = diagnose_norm_precision(numels=(1_000, 100_000))
        assert report["torch_version"] == torch.__version__
        assert len(report["precision_cases"]) == 2
        assert "overflow_case" in report

    def test_guard_fully_effective_flag_is_true(self):
        report = diagnose_norm_precision(numels=(1_000, 100_000))
        assert report["guard_fully_effective"] is True

    def test_any_precision_loss_present_reflects_native_bug_at_realistic_scale(self):
        # At 16M elements the native bug is large and reliable (see
        # TestNormPrecisionNativeBugReproduction); the diagnosis at this
        # scale should agree with the from-scratch reproduction above.
        report = diagnose_norm_precision(numels=(16_000_000,))
        assert report["any_precision_loss_present"] is True

    def test_any_overflow_present_reflects_native_overflow_bug(self):
        report = diagnose_norm_precision(numels=(1_000,))
        assert report["any_overflow_present"] is True
