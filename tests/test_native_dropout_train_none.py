from __future__ import annotations

import pytest

torch = pytest.importorskip("torch")

from torch_correctness_guards.guards.native_dropout_train_none import (  # noqa: E402
    diagnose,
    safe_native_dropout,
)


def test_diagnose_reproduces_train_none_divergence():
    """This is the core regression test: it MUST reproduce the real
    upstream defect (eager applies dropout for train=None, but
    torch.compile/inductor does not) against whatever torch build is
    currently installed. If a future torch release fixes this
    upstream (PR #197854 or similar), this assertion will start
    failing loudly -- that is a SIGNAL to re-check the upstream issue
    state and update this test, not a harness bug to silence."""
    report = diagnose()
    assert report["any_none_divergence_reproduced"] is True, (
        "expected to reproduce pytorch/pytorch#197846's train=None "
        "eager-vs-compiled divergence on the installed torch version "
        f"({report['torch_version']}) -- if this now fails, the "
        "upstream bug may have been fixed; re-verify issue #197846 "
        "and PR #197854 before treating this as a broken test"
    )


def test_guard_restores_agreement_for_all_none_cases():
    report = diagnose()
    assert report["guard_fully_restores_none_cases"] is True
    none_cases = [c for c in report["cases"] if c["train_arg"] is None]
    assert len(none_cases) == 5  # one per p value
    for c in none_cases:
        assert c["eager_mask_all_true"] == c["guarded_mask_all_true"], c


def test_explicit_train_args_never_diverge_unguarded():
    """Control: explicit train=True/False must already agree between
    eager and compiled -- isolating the defect to train=None only."""
    report = diagnose()
    assert report["explicit_cases_never_diverge_unguarded"] is True
    explicit_cases = [c for c in report["cases"] if c["train_arg"] is not None]
    assert len(explicit_cases) == 10  # 5 p values x {True, False}
    for c in explicit_cases:
        assert c["diverges_unguarded"] is False, c


def test_guard_is_a_noop_for_explicit_train_args():
    report = diagnose()
    assert report["explicit_cases_unaffected_by_guard"] is True


def test_safe_native_dropout_coerces_none_to_true():
    calls = []

    def fake_fn(input_tensor, p, train):
        calls.append((p, train))
        return "result"

    wrapped = safe_native_dropout(fake_fn)
    result = wrapped("x", 0.5, None)
    assert result == "result"
    assert calls == [(0.5, True)]


def test_safe_native_dropout_passes_through_explicit_values():
    calls = []

    def fake_fn(input_tensor, p, train):
        calls.append((p, train))
        return "result"

    wrapped = safe_native_dropout(fake_fn)
    wrapped("x", 0.3, True)
    wrapped("x", 0.7, False)
    assert calls == [(0.3, True), (0.7, False)]


def test_p_zero_boundary_case_never_diverges():
    """p=0.0 is a degenerate boundary: no dropout is possible regardless
    of train, so mask should be all-true everywhere and the guard
    should never need to change anything."""
    report = diagnose()
    zero_p_cases = [c for c in report["cases"] if c["p"] == 0.0]
    assert len(zero_p_cases) == 3  # None, True, False
    for c in zero_p_cases:
        assert c["eager_mask_all_true"] is True, c
        assert c["diverges_unguarded"] is False, c
