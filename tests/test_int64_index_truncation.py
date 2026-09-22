"""Regression tests for torch-inductor-int64-index-truncation-guard.

These prove:
  1. The bug is real and reproducible from scratch on this host's
     installed torch build: torch.compile(backend="inductor") silently
     computes an int64 arange-multiply expression using truncated
     (kernel-index-dtype, often int32) arithmetic before storing the
     result into a declared-int64 output tensor -- both tensors report
     dtype torch.int64, so the divergence is only visible in the
     actual numeric values, not the reported dtype.
  2. This reproduces on CPU (backend="inductor" without CUDA), unlike
     the upstream issue's own CUDA-only repro -- confirming the bug is
     not CUDA-specific and is genuinely verifiable on ubuntu-latest/
     macos-latest CI without a GPU runner.
  3. make_safe_int64_arange_mul is an independently-verified fix: its
     output matches eager exactly on every case, including a case
     large enough to overflow 32-bit range and a no-overflow control
     case (proving the guard doesn't introduce a spurious divergence
     when there is nothing to fix).
"""
from __future__ import annotations

import pytest

torch = pytest.importorskip("torch")

from torch_correctness_guards.guards.int64_index_truncation import (
    diagnose,
    guard_int64_op,
    make_safe_int64_arange_mul,
    safe_int64_arange_mul,
)


class TestNativeBugReproduction:
    def test_native_inductor_diverges_from_eager_on_overflowing_multiply(self):
        torch._dynamo.reset()

        def f(x):
            return torch.arange(0, 9, device=x.device, dtype=torch.int64) * torch.tensor(
                [1500000000], dtype=torch.int64, device=x.device,
            )

        x = torch.zeros(1)
        eager = f(x)
        compiled = torch.compile(f, backend="inductor")(x)
        # Not asserted unconditionally true forever: if a future torch
        # release fixes pytorch/pytorch#183901 (e.g. by merging a
        # successor to the closed, unmerged #184492/#183740), this is
        # the record that the bug existed at the version noted in the
        # README/ledger -- update accordingly rather than treating a
        # future flip to "matches" as a regression.
        assert not torch.equal(eager, compiled), (
            "expected the native int64 arange-multiply truncation bug "
            "(pytorch/pytorch#183901) to reproduce on this torch "
            f"build ({torch.__version__}); if this now matches eager, "
            "the bug may be fixed upstream -- update the README/ledger "
            "accordingly"
        )

    def test_both_native_tensors_report_int64_dtype_despite_wrong_values(self):
        """The most dangerous part of this bug: a plain dtype check
        gives no signal that anything is wrong."""
        torch._dynamo.reset()

        def f(x):
            return torch.arange(0, 9, device=x.device, dtype=torch.int64) * torch.tensor(
                [1500000000], dtype=torch.int64, device=x.device,
            )

        x = torch.zeros(1)
        eager = f(x)
        compiled = torch.compile(f, backend="inductor")(x)
        assert eager.dtype == torch.int64
        assert compiled.dtype == torch.int64
        assert not torch.equal(eager, compiled)

    def test_no_overflow_case_does_not_diverge(self):
        """Control: a multiply whose products never leave 32-bit range
        must NOT diverge -- isolates the bug to genuine overflow, not
        a blanket inductor/eager mismatch."""
        torch._dynamo.reset()

        def f(x):
            return torch.arange(0, 9, device=x.device, dtype=torch.int64) * torch.tensor(
                [2], dtype=torch.int64, device=x.device,
            )

        x = torch.zeros(1)
        eager = f(x)
        compiled = torch.compile(f, backend="inductor")(x)
        assert torch.equal(eager, compiled)


class TestGuardIsNotACoincidentalNoOp:
    def test_guard_matches_eager_where_native_silently_diverges(self):
        torch._dynamo.reset()

        def f(x):
            return torch.arange(0, 9, device=x.device, dtype=torch.int64) * torch.tensor(
                [1500000000], dtype=torch.int64, device=x.device,
            )

        safe_fn = make_safe_int64_arange_mul(torch)

        def guarded_fn(x):
            return safe_fn(0, 9, torch.tensor([1500000000], dtype=torch.int64, device=x.device))

        x = torch.zeros(1)
        eager = f(x)
        native_compiled = torch.compile(f, backend="inductor")(x)
        torch._dynamo.reset()
        guarded_compiled = torch.compile(guarded_fn, backend="inductor")(x)

        # Establish the guard has something real to catch.
        assert not torch.equal(eager, native_compiled)
        # The guard must match eager on the exact same input.
        assert torch.equal(eager, guarded_compiled)

    def test_safe_int64_arange_mul_module_level_wrapper_matches_eager(self):
        torch._dynamo.reset()
        eager = torch.arange(0, 9, dtype=torch.int64) * 1500000000
        guarded = safe_int64_arange_mul(0, 9, torch.tensor(1500000000, dtype=torch.int64))
        assert torch.equal(eager, guarded)


class TestGuardIntOpGeneralPurposeFactory:
    def test_guard_int64_op_forces_eager_execution_under_compile(self):
        """guard_int64_op wraps an arbitrary callable; confirm the
        wrapped callable's compiled-region behavior matches its
        eager behavior when called from inside torch.compile, for a
        callable not covered by make_safe_int64_arange_mul's fixed
        signature. (Note: not every large multiplier/arange-size
        combination reproduces the underlying Inductor truncation on
        every host/torch build -- the guard's correctness claim here
        is that it matches eager, not that every input necessarily
        diverges without it; see TestNativeBugReproduction and
        TestDiagnose for the specific inputs confirmed to reproduce
        the bug on this host.)"""
        torch._dynamo.reset()

        def raw_fn(x):
            return torch.arange(0, 5, device=x.device, dtype=torch.int64) * torch.tensor(
                [3000000000], dtype=torch.int64, device=x.device,
            )

        guarded_raw = guard_int64_op(torch, raw_fn)

        def guarded_wrapper(x):
            return guarded_raw(x)

        x = torch.zeros(1)
        eager = raw_fn(x)
        guarded_compiled = torch.compile(guarded_wrapper, backend="inductor")(x)

        assert torch.equal(eager, guarded_compiled)


class TestDiagnose:
    def test_diagnose_runs_and_reports_torch_version(self):
        report = diagnose()
        assert report["torch_version"] == torch.__version__
        assert report["issue_url"] == "https://github.com/pytorch/pytorch/issues/183901"
        assert len(report["cases"]) == 3

    def test_any_native_diverges_flag_is_true_on_this_host(self):
        report = diagnose()
        assert report["any_native_diverges"] is True, (
            "expected the known upstream int64 arange-multiply "
            "truncation bug (pytorch/pytorch#183901) to reproduce on "
            f"torch {report['torch_version']}; if this now fails, the "
            "bug may have been fixed upstream -- verify against the "
            "issue tracker before assuming a test regression"
        )

    def test_guard_fully_correct_flag_is_true(self):
        report = diagnose()
        assert report["guard_fully_correct"] is True
        for c in report["cases"]:
            assert c["guard_matches_eager"], c["description"]

    def test_no_overflow_control_case_does_not_report_native_divergence(self):
        report = diagnose()
        control_case = next(
            c for c in report["cases"] if "no-overflow control" in c["description"]
        )
        assert control_case["native_diverges"] is False

    def test_overflowing_cases_report_native_divergence(self):
        report = diagnose()
        overflowing = [
            c for c in report["cases"] if "no-overflow control" not in c["description"]
        ]
        assert len(overflowing) == 2
        for c in overflowing:
            assert c["native_diverges"] is True, c["description"]


def test_torch_unavailable_error_is_distinct_type():
    from torch_correctness_guards.guards.int64_index_truncation import TorchUnavailableError

    assert issubclass(TorchUnavailableError, RuntimeError)
