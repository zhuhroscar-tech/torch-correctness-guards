"""Regression tests for torch-compile-shuffle-sample-frozen-guard.

These prove:
  1. The bug is real and reproducible from scratch on this host's
     installed torch build: random.shuffle()/random.sample() inside a
     torch.compile(backend='eager')'d function return the exact same
     result on every call after the first, while eager code (same
     seed, no reseed between calls) legitimately varies call to call.
  2. safe_shuffle/safe_sample are independently-verified fixes: with
     the guard applied, the compiled function's sequence of results
     matches the eager sequence byte-for-byte across multiple calls --
     not just a coincidental single-call match.
  3. The guard is not a no-op that happens to match by accident: we
     assert the *native* (unguarded) path really is frozen on this
     host before trusting the guarded path's match as meaningful.
"""
from __future__ import annotations

import random

import pytest

torch = pytest.importorskip("torch")

from torch_correctness_guards.guards.shuffle_sample_frozen import (  # noqa: E402
    diagnose,
    make_safe_sample,
    make_safe_shuffle,
    safe_sample,
    safe_shuffle,
)


class TestNativeBugReproduction:
    def test_native_shuffle_is_frozen_after_first_compiled_call(self):
        # NOTE: the compiled function takes a tensor argument AND
        # performs a real tensor op on it, exactly like the upstream
        # issue's own reproducer -- without an actual tensor op,
        # Dynamo has nothing to build a graph around and does NOT
        # reproduce this defect on this host (confirmed by hand). See
        # core.py's _run_shuffle_case docstring note for the same
        # constraint.
        template = [1, 2, 3, 4, 5]

        def native_fn(x):
            lst = list(template)
            random.shuffle(lst)
            _ = x + 1
            return lst

        compiled = torch.compile(native_fn, backend="eager")
        zero = torch.zeros(1)
        random.seed(7)
        results = [compiled(zero) for _ in range(3)]
        # Not asserted unconditionally true forever: if a future
        # Dynamo release fixes pytorch/pytorch#197085, this is the
        # record that the bug existed at the version noted in the
        # README/ledger -- update accordingly rather than treating a
        # future flip to "varies" as a regression.
        assert results[1] == results[0] and results[2] == results[0], (
            "expected the native trace-time-freeze bug (all 3 calls "
            f"identical); got {results} -- if these now differ, "
            "pytorch/pytorch#197085 may be fixed upstream -- update "
            "the README/ledger accordingly"
        )

    def test_native_sample_is_frozen_after_first_compiled_call(self):
        population = list(range(100))

        def native_fn(x):
            s = random.sample(population, 2)
            _ = x + 1
            return s

        compiled = torch.compile(native_fn, backend="eager")
        zero = torch.zeros(1)
        random.seed(7)
        results = [compiled(zero) for _ in range(3)]
        assert results[1] == results[0] and results[2] == results[0]

    def test_eager_shuffle_genuinely_varies_across_calls(self):
        """Sanity check: eager must actually vary (no reseed between
        calls) so the frozen-compiled comparison above is meaningful,
        not an artifact of a template that happens to shuffle to
        itself."""
        template = [1, 2, 3, 4, 5]

        def eager_fn():
            lst = list(template)
            random.shuffle(lst)
            return lst

        random.seed(7)
        results = [eager_fn() for _ in range(3)]
        assert len({tuple(r) for r in results}) > 1

    def test_matches_upstream_issue_reported_output_at_seed_7(self):
        """Byte-exact match to pytorch/pytorch#197085's own reported
        output at seed=7, confirming this is the SAME defect, not a
        superficially similar one."""

        def shuffle_fn(x):
            lst = [1, 2, 3, 4, 5]
            random.shuffle(lst)
            return x + lst[0], lst

        def sample_fn(x):
            s = random.sample(range(100), 2)
            return x + s[0], s

        cf_shuffle = torch.compile(shuffle_fn, backend="eager")
        random.seed(7)
        eager_shuffle = [shuffle_fn(torch.zeros(1))[1] for _ in range(3)]
        random.seed(7)
        comp_shuffle = [cf_shuffle(torch.zeros(1))[1] for _ in range(3)]

        assert eager_shuffle == [[5, 1, 4, 2, 3], [3, 4, 2, 5, 1], [4, 3, 1, 2, 5]]
        assert comp_shuffle == [[5, 1, 4, 2, 3], [5, 1, 4, 2, 3], [5, 1, 4, 2, 3]]

        cf_sample = torch.compile(sample_fn, backend="eager")
        random.seed(7)
        eager_sample = [sample_fn(torch.zeros(1))[1] for _ in range(3)]
        random.seed(7)
        comp_sample = [cf_sample(torch.zeros(1))[1] for _ in range(3)]

        assert eager_sample == [[41, 19], [50, 83], [6, 9]]
        assert comp_sample == [[41, 19], [41, 19], [41, 19]]


class TestGuardRestoresEagerSemantics:
    def test_safe_shuffle_matches_eager_across_calls(self):
        # Tensor-argument compiled function with a real tensor op,
        # matching the constraint noted in core.py's _run_shuffle_case
        # (without an actual tensor op, Dynamo has nothing to build a
        # graph around and does not reproduce the freeze on this
        # host, so a test without it would prove nothing).
        template = [1, 2, 3, 4, 5]

        def eager_fn(x):
            lst = list(template)
            random.shuffle(lst)
            _ = x + 1
            return lst

        def guarded_fn(x):
            lst = list(template)
            safe_shuffle(lst)
            _ = x + 1
            return lst

        compiled_guarded = torch.compile(guarded_fn, backend="eager")
        zero = torch.zeros(1)

        random.seed(7)
        eager_results = [eager_fn(zero) for _ in range(3)]

        random.seed(7)
        guarded_results = [compiled_guarded(zero) for _ in range(3)]

        assert guarded_results == eager_results
        # Meaningfulness check: the eager sequence must actually vary,
        # otherwise a frozen guard could coincidentally "match".
        assert len({tuple(r) for r in eager_results}) > 1

    def test_safe_sample_matches_eager_across_calls(self):
        population = list(range(100))

        def eager_fn(x):
            s = random.sample(population, 2)
            _ = x + 1
            return s

        def guarded_fn(x):
            s = safe_sample(population, 2)
            _ = x + 1
            return s

        compiled_guarded = torch.compile(guarded_fn, backend="eager")
        zero = torch.zeros(1)

        random.seed(7)
        eager_results = [eager_fn(zero) for _ in range(3)]

        random.seed(7)
        guarded_results = [compiled_guarded(zero) for _ in range(3)]

        assert guarded_results == eager_results
        assert len({tuple(r) for r in eager_results}) > 1

    def test_guard_is_not_a_coincidental_noop(self):
        """Confirms the native path really is frozen (something real
        for the guard to fix) before trusting the guard's match. Uses
        a tensor-argument compiled function with a real tensor op,
        matching the constraint noted in core.py's _run_shuffle_case."""
        template = [1, 2, 3, 4, 5]

        def native_fn(x):
            lst = list(template)
            random.shuffle(lst)
            _ = x + 1
            return lst

        def guarded_fn(x):
            lst = list(template)
            safe_shuffle(lst)
            _ = x + 1
            return lst

        compiled_native = torch.compile(native_fn, backend="eager")
        compiled_guarded = torch.compile(guarded_fn, backend="eager")
        zero = torch.zeros(1)

        random.seed(7)
        native_results = [compiled_native(zero) for _ in range(3)]
        random.seed(7)
        guarded_results = [compiled_guarded(zero) for _ in range(3)]

        assert native_results[1] == native_results[0]  # native: frozen
        assert guarded_results[1] != guarded_results[0] or guarded_results[2] != guarded_results[0]


class TestMakeSafeFunctionsReturnCallableBoundToTorchModule:
    def test_make_safe_shuffle_returns_callable(self):
        safe_fn = make_safe_shuffle(torch)
        lst = [1, 2, 3]
        result = safe_fn(lst)
        assert sorted(result) == [1, 2, 3]

    def test_make_safe_sample_returns_callable(self):
        safe_fn = make_safe_sample(torch)
        result = safe_fn(range(10), 3)
        assert len(result) == 3
        assert len(set(result)) == 3


class TestDiagnose:
    def test_diagnose_default_runs_and_reports_consistent_structure(self):
        report = diagnose()
        assert isinstance(report["torch_version"], str)
        assert report["issue_url"] == "https://github.com/pytorch/pytorch/issues/197085"
        assert len(report["cases"]) == 2
        assert {c["kind"] for c in report["cases"]} == {"shuffle", "sample"}

    def test_any_native_frozen_flag_is_true_on_this_host(self):
        report = diagnose()
        assert report["any_native_frozen"] is True, (
            "expected the native trace-time-freeze bug to reproduce "
            f"on torch {report['torch_version']}; if this now fails, "
            "pytorch/pytorch#197085 may be fixed upstream -- update "
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
            assert case["eager_shows_real_variation"] is True, case["description"]

    def test_diagnose_accepts_custom_seed_and_calls(self):
        report = diagnose(seed=123, calls=4)
        assert report["seed"] == 123
        assert report["calls"] == 4
        for case in report["cases"]:
            assert len(case["eager_sequence"]) == 4
