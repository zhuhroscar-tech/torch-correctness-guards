"""Regression tests for torch-take-along-dim-oob-guard.

These prove:
  1. The bug is real and reproducible from scratch on this host's
     installed torch build: torch.take_along_dim(input, indices,
     dim=<int>) silently wraps out-of-bounds indices modulo the
     dimension size instead of raising, in 1D and 2D, for large
     positive indices, huge positive indices, and negative indices
     past -size.
  2. Every sibling indexing op (gather, take, plain indexing,
     index_select) and take_along_dim's own dim=None path correctly
     raise on the same out-of-bounds index -- confirming this is a
     dim=<int>-path-specific defect, not a general indexing-contract
     ambiguity.
  3. make_safe_take_along_dim is an independently-verified fix: it
     raises IndexError on every genuinely out-of-bounds case (a
     bug-injection-style check -- the guard is not a no-op that
     happens to match by coincidence) and matches a NumPy oracle
     (np.take_along_axis) on every valid case, including valid
     negative indices, which must continue to work exactly as today.
  4. The already-correct dim=None path is delegated unchanged (no
     double-guarding / no regression on a path that never had a bug).
"""
from __future__ import annotations

import pytest

torch = pytest.importorskip("torch")

from torch_correctness_guards.guards.take_along_dim_oob import (
    diagnose,
    make_safe_take_along_dim,
    safe_take_along_dim,
)


class TestNativeBugReproduction:
    def test_native_silently_wraps_oob_positive_index(self):
        x = torch.tensor([10.0, 20.0, 30.0])
        idx = torch.tensor([9])
        # Not asserted unconditionally true forever: if a future torch
        # release fixes pytorch/pytorch#196106, this is the record
        # that the bug existed at the version noted in the
        # ledger/README -- update accordingly rather than treating a
        # future flip to "raises" as a regression.
        result = torch.take_along_dim(x, idx, dim=0)
        assert result.item() == pytest.approx(10.0), (
            "expected the native silent-wrap bug (x[9 % 3] == x[0] == 10.0); "
            "if this now raises, pytorch/pytorch#196106 may be fixed "
            "upstream -- update the README/ledger accordingly"
        )

    def test_native_silently_wraps_huge_positive_index(self):
        x = torch.tensor([10.0, 20.0, 30.0])
        idx = torch.tensor([2 ** 40])
        result = torch.take_along_dim(x, idx, dim=0)
        assert result.item() == pytest.approx(20.0)

    def test_native_silently_wraps_negative_index_past_minus_size(self):
        x = torch.tensor([10.0, 20.0, 30.0])
        idx = torch.tensor([-4])
        result = torch.take_along_dim(x, idx, dim=0)
        assert result.item() == pytest.approx(30.0)

    def test_native_valid_negative_index_is_correct(self):
        # Not a bug: -1..-size is documented, correct behavior and
        # must remain unaffected by any guard.
        x = torch.tensor([10.0, 20.0, 30.0])
        idx = torch.tensor([-1])
        result = torch.take_along_dim(x, idx, dim=0)
        assert result.item() == pytest.approx(30.0)

    def test_dim_none_path_correctly_raises(self):
        x = torch.tensor([10.0, 20.0, 30.0])
        idx = torch.tensor([9])
        with pytest.raises((IndexError, RuntimeError)):
            torch.take_along_dim(x, idx, dim=None)


class TestSiblingOpsCorrectlyRaiseOnSameIndex:
    """Confirms this is a dim=<int>-path-specific defect: every sibling
    op raises on the exact same out-of-bounds index."""

    @pytest.fixture
    def oob_setup(self):
        x = torch.tensor([10.0, 20.0, 30.0])
        idx = torch.tensor([9])
        return x, idx

    def test_gather_raises(self, oob_setup):
        x, idx = oob_setup
        with pytest.raises((IndexError, RuntimeError)):
            torch.gather(x, 0, idx)

    def test_take_raises(self, oob_setup):
        x, idx = oob_setup
        with pytest.raises((IndexError, RuntimeError)):
            torch.take(x, idx)

    def test_plain_indexing_raises(self, oob_setup):
        x, idx = oob_setup
        with pytest.raises((IndexError, RuntimeError)):
            x[idx]

    def test_index_select_raises(self, oob_setup):
        x, idx = oob_setup
        with pytest.raises((IndexError, RuntimeError)):
            torch.index_select(x, 0, idx)


class TestGuardIsNotACoincidentalNoOp:
    def test_guard_raises_where_native_silently_wraps(self):
        x = torch.tensor([10.0, 20.0, 30.0])
        idx = torch.tensor([9])
        # Confirm the native call does NOT raise (establishes the
        # guard has something real to catch).
        native_result = torch.take_along_dim(x, idx, dim=0)
        assert native_result.item() == pytest.approx(10.0)
        # The guard must raise on the exact same input.
        with pytest.raises(IndexError):
            safe_take_along_dim(x, idx, dim=0)

    def test_guard_raises_on_2d_oob(self):
        m = torch.arange(6.0).reshape(2, 3)
        idx = torch.tensor([[5], [7]])
        native_result = m.take_along_dim(idx, dim=1)
        assert native_result.tolist() == [[2.0], [4.0]]
        with pytest.raises(IndexError):
            safe_take_along_dim(m, idx, dim=1)


class TestGuardMatchesNumpyOracleOnValidInputs:
    def test_guard_matches_oracle_valid_negative_index(self):
        import numpy as np

        x = torch.tensor([10.0, 20.0, 30.0])
        idx = torch.tensor([-1])
        guarded = safe_take_along_dim(x, idx, dim=0)
        oracle = np.take_along_axis(x.numpy(), idx.numpy(), axis=0)
        assert guarded.tolist() == oracle.tolist()

    def test_guard_matches_oracle_valid_positive_index(self):
        import numpy as np

        x = torch.tensor([10.0, 20.0, 30.0])
        idx = torch.tensor([1])
        guarded = safe_take_along_dim(x, idx, dim=0)
        oracle = np.take_along_axis(x.numpy(), idx.numpy(), axis=0)
        assert guarded.tolist() == oracle.tolist()

    def test_guard_matches_oracle_2d_valid(self):
        import numpy as np

        m = torch.arange(6.0).reshape(2, 3)
        idx = torch.tensor([[0], [2]])
        guarded = safe_take_along_dim(m, idx, dim=1)
        oracle = np.take_along_axis(m.numpy(), idx.numpy(), axis=1)
        assert guarded.tolist() == oracle.tolist()

    def test_guard_delegates_dim_none_unchanged(self):
        x = torch.tensor([10.0, 20.0, 30.0])
        idx = torch.tensor([1])
        guarded = safe_take_along_dim(x, idx, dim=None)
        native = torch.take_along_dim(x, idx, dim=None)
        assert guarded.tolist() == native.tolist()

    def test_guard_dim_none_still_raises_on_oob(self):
        x = torch.tensor([10.0, 20.0, 30.0])
        idx = torch.tensor([9])
        with pytest.raises((IndexError, RuntimeError)):
            safe_take_along_dim(x, idx, dim=None)


class TestMakeSafeTakeAlongDimReturnsCallableBoundToTorchModule:
    def test_returns_callable(self):
        safe_fn = make_safe_take_along_dim(torch)
        x = torch.tensor([1.0, 2.0, 3.0])
        idx = torch.tensor([0])
        result = safe_fn(x, idx, dim=0)
        assert result.tolist() == [1.0]

    def test_negative_dim_resolves_correctly(self):
        m = torch.arange(6.0).reshape(2, 3)
        idx = torch.tensor([[0], [2]])
        safe_fn = make_safe_take_along_dim(torch)
        # dim=-1 is equivalent to dim=1 for a 2D tensor.
        result_neg = safe_fn(m, idx, dim=-1)
        result_pos = safe_fn(m, idx, dim=1)
        assert result_neg.tolist() == result_pos.tolist()

    def test_negative_dim_oob_raises(self):
        m = torch.arange(6.0).reshape(2, 3)
        idx = torch.tensor([[5], [7]])
        safe_fn = make_safe_take_along_dim(torch)
        with pytest.raises(IndexError):
            safe_fn(m, idx, dim=-1)


class TestDiagnose:
    def test_diagnose_default_runs_and_reports_consistent_structure(self):
        report = diagnose()
        assert isinstance(report["torch_version"], str)
        assert report["issue_url"] == "https://github.com/pytorch/pytorch/issues/196106"
        assert len(report["cases"]) == 9

    def test_any_native_silently_wrong_flag_is_true_on_this_host(self):
        report = diagnose()
        assert report["any_native_silently_wrong"] is True, (
            "expected the native silent OOB-wrap bug to reproduce on "
            f"torch {report['torch_version']}; if this now fails, "
            "pytorch/pytorch#196106 may be fixed upstream -- update "
            "the README/ledger accordingly rather than treating this "
            "as a regression"
        )

    def test_guard_fully_correct_flag_is_true(self):
        report = diagnose()
        assert report["guard_fully_correct"] is True

    def test_every_case_has_a_verdict(self):
        report = diagnose()
        for case in report["cases"]:
            assert case["guard_correct"] is True, case["description"]
