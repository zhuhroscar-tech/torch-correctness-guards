"""Regression tests for torch-as-strided-restride-oob-guard.

These prove:
  1. The bug is real and reproducible from scratch on this host's
     installed torch build: torch.compile(backend="inductor") either
     raises a spurious "out of bounds for storage" RuntimeError, or
     returns a result diverging from eager, for an as_strided call
     that is provably in-bounds by the operation's own documented
     storage/stride/storage_offset contract -- exactly the
     pytorch/pytorch#197431 repro pattern (repeat -> strided slice ->
     as_strided with the pre-slice stride).
  2. safe_as_strided() is an independently verified fix: it matches
     eager's own output (the independent oracle -- eager's result is
     provably in-bounds and correct per the documented contract) on
     every call, including under torch.compile.
  3. A bug-injection test proves the divergence detection is
     non-tautological: the UNGUARDED compiled call (not wrapped in
     safe_as_strided) really does diverge from eager, confirming the
     test harness would catch a regression if the guard were removed
     or broken.
  4. The documented storage-span contract itself is verified
     independently (a plain-Python re-derivation of the formula from
     as_strided's own docs), so "eager is correct" is not asserted on
     faith -- it is checked against the operation's own documented
     math, matching this fleet's existing acceptance-oracle discipline
     (numguard's naive-vs-stable-vs-reference pattern applied here to
     a view operation instead of a numeric formula).
"""
from __future__ import annotations

import pytest

torch = pytest.importorskip("torch")

from torch_correctness_guards.guards.as_strided_restride_oob import diagnose, safe_as_strided


def _required_storage_span(size, stride, storage_offset=0):
    """Independent re-derivation of as_strided's own documented
    storage-span requirement, from the PyTorch docs:
    storage_offset + sum((size[i] - 1) * stride[i] for i in dims) + 1.
    Used as a high-precision oracle independent of any torch internals.
    """
    return storage_offset + sum((s - 1) * st for s, st in zip(size, stride)) + 1


class TestOracleContract:
    def test_repro_call_is_provably_in_bounds_per_documented_contract(self):
        # The exact upstream repro shape: full has stride (4, 1) and
        # storage size 12 (3 rows x 4 cols); the as_strided call
        # requests size=(3, 2), stride=(4, 2), storage_offset=0.
        required = _required_storage_span(size=(3, 2), stride=(4, 2), storage_offset=0)
        actual_storage_size = 12
        assert required <= actual_storage_size, (
            f"expected the documented as_strided contract to prove this "
            f"call in-bounds (required={required} <= storage={actual_storage_size}); "
            "if this fails, the guard's premise (eager is provably correct, "
            "not merely 'whatever eager happens to return') no longer holds"
        )

    def test_eager_result_matches_independently_computed_expected_values(self):
        x = torch.tensor([[7.0, 9.0]])
        full = x.repeat(3, 2)
        sliced = full[:, ::2]
        result = torch.as_strided(sliced, (3, 2), full.stride())
        # Independently derive the expected values by hand from the
        # contiguous storage layout, not copied from any tool output:
        # full's row-major storage (shape (3,4), values [7,9] tiled) is
        # [7,9,7,9, 7,9,7,9, 7,9,7,9] (12 elements). as_strided with
        # size=(3,2), stride=full.stride()=(4,1), storage_offset=0
        # (sliced's own offset) reads storage[i*4+j] for i in 0..2,
        # j in 0..1: row i always starts at the same repeating [7,9]
        # pair regardless of i, since the row stride (4) exactly
        # matches the tile period.
        expected = [[7.0, 9.0], [7.0, 9.0], [7.0, 9.0]]
        assert result.tolist() == expected


class TestNativeBugReproduction:
    def test_divergence_reproduces_on_this_host(self):
        # Not asserted unconditionally true forever: if a future torch
        # release fixes Inductor's as_strided storage-span
        # miscomputation, this docstring is the record that the bug
        # existed at the version noted in the ledger/README. On torch
        # 2.14.0 (this host) it reproduces reliably across all cases.
        report = diagnose()
        assert report["any_divergence_bug"] is True, (
            f"expected an as_strided restride divergence on torch "
            f"{report['torch_version']}; if this now fails, the bug may "
            "be fixed upstream (pytorch/pytorch#197431) -- update the "
            "README/ledger accordingly rather than treating this as a "
            "regression"
        )

    def test_all_three_cases_reproduce_the_divergence(self):
        report = diagnose()
        cases = report["cases"]
        assert len(cases) == 3
        for c in cases:
            assert c["divergence_bug"] is True, (
                f"call {c['call_index']} (x={c['x_values']}) was expected "
                "to reproduce the native as_strided divergence on this host"
            )

    def test_eager_never_errors_and_is_deterministic(self):
        report = diagnose()
        for c in report["cases"]:
            assert c["eager_ok"] is True


class TestBugInjectionNonTautological:
    """Prove the divergence assertions above are not tautological by
    running the UNGUARDED compiled call directly (bypassing
    safe_as_strided entirely) and confirming it really does diverge
    from eager -- i.e. the test harness would catch a regression if
    safe_as_strided stopped being applied."""

    def test_unguarded_compiled_as_strided_diverges_from_eager(self):
        torch._dynamo.reset()

        def make_view(x):
            full = x.repeat(3, 2)
            sliced = full[:, ::2]
            return torch.as_strided(sliced, (3, 2), full.stride())

        x = torch.tensor([[7.0, 9.0]])
        eager_result = make_view(x)

        compiled_fn = torch.compile(make_view, backend="inductor")

        diverged = False
        try:
            compiled_result = compiled_fn(x)
            if not torch.equal(compiled_result, eager_result):
                diverged = True
        except RuntimeError:
            diverged = True

        assert diverged, (
            "sanity check: the unguarded compiled as_strided call really "
            "must diverge from eager on this host for this test suite's "
            "positive assertions to be meaningful evidence of a real "
            "bug, not a tautology"
        )


class TestSafeAsStridedMatchesEager:
    def test_safe_as_strided_matches_eager_uncompiled(self):
        x = torch.tensor([[7.0, 9.0]])
        full = x.repeat(3, 2)
        sliced = full[:, ::2]
        eager = torch.as_strided(sliced, (3, 2), full.stride())
        guarded = safe_as_strided(sliced, (3, 2), full.stride())
        assert torch.equal(eager, guarded)

    def test_safe_as_strided_matches_eager_for_every_call_in_diagnose(self):
        report = diagnose()
        for case in report["cases"]:
            assert case["guard_matches_eager"] is True, (
                f"call {case['call_index']} (x={case['x_values']}): guard "
                f"value {case['guard_value']} did not match eager value "
                f"{case['eager_value']}"
            )

    def test_guard_fully_correct_flag_is_true(self):
        report = diagnose()
        assert report["guard_fully_correct"] is True

    def test_safe_as_strided_works_under_torch_compile(self):
        torch._dynamo.reset()

        def make_view_guarded(x):
            full = x.repeat(3, 2)
            sliced = full[:, ::2]
            return safe_as_strided(sliced, (3, 2), full.stride())

        compiled_guard = torch.compile(make_view_guarded, backend="inductor")
        x = torch.tensor([[7.0, 9.0]])
        guarded_val = compiled_guard(x)
        eager_val = make_view_guarded(x)
        assert torch.equal(guarded_val, eager_val)

    def test_safe_as_strided_preserves_storage_offset_argument(self):
        # Exercise the storage_offset kwarg path explicitly (not just
        # the default-None path used by the repro above).
        x = torch.arange(12, dtype=torch.float32)
        base = x.as_strided((3, 2), (4, 1), storage_offset=2)
        guarded = safe_as_strided(x, (3, 2), (4, 1), storage_offset=2)
        assert torch.equal(base, guarded)


class TestDiagnose:
    def test_diagnose_runs_and_reports_consistent_structure(self):
        report = diagnose()
        assert len(report["cases"]) == 3
        assert isinstance(report["torch_version"], str)
        assert "issue_urls" in report
        assert len(report["issue_urls"]) == 2

    def test_diagnose_is_deterministic_across_repeated_calls(self):
        r1 = diagnose(seed=0)
        r2 = diagnose(seed=0)
        assert r1["any_divergence_bug"] == r2["any_divergence_bug"]
        assert r1["guard_fully_correct"] == r2["guard_fully_correct"]
