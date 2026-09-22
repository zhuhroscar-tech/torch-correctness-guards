"""Regression tests for torch-inductor-duplicate-index-writeorder-guard.

These prove:
  1. The bug is real and reproducible from scratch on this host's
     installed torch build: v[idx] = v[idx] + delta under
     torch.compile(backend='inductor') diverges from eager when idx is
     a COMPUTED duplicate index (torch.arange(n) % 2).
  2. The literal-duplicate-index control case (torch.tensor([0,1,0,1]))
     does NOT diverge -- isolating the defect to computed index
     expressions specifically, per the upstream issue.
  3. safe_dup_index_assign is an independently-verified fix: with the
     guard applied, the compiled function's result matches eager on
     the triggering case.
  4. The guard is not a no-op that happens to match by accident: we
     assert the *native* (unguarded) path really diverges on this host
     before trusting the guarded path's match as meaningful.
"""
from __future__ import annotations

import pytest

torch = pytest.importorskip("torch")

from torch_correctness_guards.guards.duplicate_index_writeorder import (
    diagnose,
    make_safe_dup_index_assign,
    safe_dup_index_assign,
)


class TestNativeBugReproduction:
    def test_computed_duplicate_index_diverges_under_inductor(self):
        def native_fn(v):
            idx = torch.arange(4) % 2
            v[idx] = v[idx] + 1.0
            return v

        v_eager = torch.tensor([1.0, 2.0, 3.0, 4.0])
        eager_result = native_fn(v_eager.clone()).tolist()

        compiled = torch.compile(native_fn, backend="inductor")
        v_compiled = torch.tensor([1.0, 2.0, 3.0, 4.0])
        compiled_result = compiled(v_compiled.clone()).tolist()

        # Not asserted as an eternal truth: if a future Inductor
        # release fixes pytorch/pytorch#197582, this is the record
        # that the bug existed at the version noted in the
        # README/ledger -- update accordingly rather than treating a
        # future flip to "matches" as a regression.
        assert compiled_result != eager_result, (
            "expected the native Inductor write-order divergence on "
            f"this host; eager={eager_result} compiled={compiled_result} "
            "-- if these now match, pytorch/pytorch#197582 may be fixed "
            "upstream -- update the README/ledger accordingly"
        )
        assert eager_result == [2.0, 3.0, 3.0, 4.0]

    def test_matches_upstream_issue_reported_output(self):
        """Byte-exact match to pytorch/pytorch#197582's own reported
        eager vs compiled output, confirming this is the SAME defect,
        not a superficially similar one."""

        def fn(v):
            idx = torch.arange(4) % 2
            v[idx] = v[idx] + 1.0
            return v

        v_eager = torch.tensor([1.0, 2.0, 3.0, 4.0])
        eager_result = fn(v_eager.clone()).tolist()

        compiled = torch.compile(fn, backend="inductor")
        v_compiled = torch.tensor([1.0, 2.0, 3.0, 4.0])
        compiled_result = compiled(v_compiled.clone()).tolist()

        assert eager_result == [2.0, 3.0, 3.0, 4.0]
        assert compiled_result == [3.0, 4.0, 3.0, 4.0]


class TestLiteralIndexControlCaseIsolatesDefect:
    def test_literal_duplicate_index_does_not_diverge_under_inductor(self):
        """Control case named explicitly in the upstream issue: the
        SAME logical duplicate-index read-modify-write, but with a
        literal tensor instead of a computed expression, does NOT
        trigger the miscompilation."""

        def fn(v):
            idx = torch.tensor([0, 1, 0, 1])
            v[idx] = v[idx] + 1.0
            return v

        v_eager = torch.tensor([1.0, 2.0, 3.0, 4.0])
        eager_result = fn(v_eager.clone()).tolist()

        compiled = torch.compile(fn, backend="inductor")
        v_compiled = torch.tensor([1.0, 2.0, 3.0, 4.0])
        compiled_result = compiled(v_compiled.clone()).tolist()

        assert compiled_result == eager_result, (
            "expected the literal-index control case to match eager "
            f"on this host; eager={eager_result} compiled={compiled_result} "
            "-- if this now diverges too, the defect may have widened "
            "beyond computed indices -- update the README/ledger "
            "accordingly rather than treating this as unrelated"
        )


class TestGuardRestoresEagerSemantics:
    def test_safe_dup_index_assign_matches_eager_on_triggering_case(self):
        def eager_fn(v):
            idx = torch.arange(4) % 2
            v[idx] = v[idx] + 1.0
            return v

        def guarded_fn(v):
            idx = torch.arange(4) % 2
            safe_dup_index_assign(v, idx, 1.0)
            return v

        v_eager = torch.tensor([1.0, 2.0, 3.0, 4.0])
        eager_result = eager_fn(v_eager.clone()).tolist()

        compiled_guarded = torch.compile(guarded_fn, backend="inductor")
        v_guarded = torch.tensor([1.0, 2.0, 3.0, 4.0])
        guarded_result = compiled_guarded(v_guarded.clone()).tolist()

        assert guarded_result == eager_result

    def test_guard_is_not_a_coincidental_noop(self):
        """Confirms the native path really diverges (something real
        for the guard to fix) before trusting the guard's match."""

        def native_fn(v):
            idx = torch.arange(4) % 2
            v[idx] = v[idx] + 1.0
            return v

        def guarded_fn(v):
            idx = torch.arange(4) % 2
            safe_dup_index_assign(v, idx, 1.0)
            return v

        compiled_native = torch.compile(native_fn, backend="inductor")
        v_native = torch.tensor([1.0, 2.0, 3.0, 4.0])
        native_result = compiled_native(v_native.clone()).tolist()

        compiled_guarded = torch.compile(guarded_fn, backend="inductor")
        v_guarded = torch.tensor([1.0, 2.0, 3.0, 4.0])
        guarded_result = compiled_guarded(v_guarded.clone()).tolist()

        assert native_result == [3.0, 4.0, 3.0, 4.0]  # native: diverged
        assert guarded_result == [2.0, 3.0, 3.0, 4.0]  # guarded: correct


class TestMakeSafeFunctionReturnsCallableBoundToTorchModule:
    def test_make_safe_dup_index_assign_returns_callable(self):
        safe_fn = make_safe_dup_index_assign(torch)
        v = torch.tensor([1.0, 2.0, 3.0, 4.0])
        idx = torch.tensor([0, 1, 0, 1])
        result = safe_fn(v, idx, 1.0)
        assert result.tolist() == [2.0, 3.0, 3.0, 4.0]


class TestModuleLevelConvenienceWrapper:
    def test_safe_dup_index_assign_module_level_resolves_torch_lazily(self):
        v = torch.tensor([1.0, 2.0, 3.0, 4.0])
        idx = torch.tensor([0, 1, 0, 1])
        result = safe_dup_index_assign(v, idx, 1.0)
        assert result.tolist() == [2.0, 3.0, 3.0, 4.0]
        # Confirms it mutates and returns the SAME tensor object, matching
        # make_safe_dup_index_assign's contract.
        assert result is v


class TestDiagnose:
    def test_diagnose_default_runs_and_reports_consistent_structure(self):
        report = diagnose()
        assert isinstance(report["torch_version"], str)
        assert report["issue_url"] == "https://github.com/pytorch/pytorch/issues/197582"
        assert len(report["cases"]) == 2
        assert {c["kind"] for c in report["cases"]} == {
            "computed_duplicate_index",
            "literal_duplicate_index",
        }

    def test_bug_reproduced_flag_is_true_on_this_host(self):
        report = diagnose()
        assert report["bug_reproduced"] is True, (
            "expected the computed-duplicate-index write-order "
            f"divergence to reproduce on torch {report['torch_version']}; "
            "if this now fails, pytorch/pytorch#197582 may be fixed "
            "upstream -- update the README/ledger accordingly rather "
            "than treating this as a regression"
        )

    def test_isolation_confirmed_flag_is_true(self):
        report = diagnose()
        assert report["isolation_confirmed"] is True

    def test_guard_fully_correct_flag_is_true(self):
        report = diagnose()
        assert report["guard_fully_correct"] is True

    def test_every_case_has_a_verdict(self):
        report = diagnose()
        for case in report["cases"]:
            assert isinstance(case["guard_correct"], bool), case["description"]
