"""Regression tests for the expand()+fill_() bug and its guard.

Deliberately includes a bug-injection style test (verifying the native
compiled path DOES diverge from eager on this host's real torch build)
alongside the guard-correctness tests, so the test suite can never
silently pass merely because the underlying bug happened to disappear
on a newer torch version without anyone noticing -- see
test_native_bug_still_reproduces_on_this_torch_version.
"""
from __future__ import annotations

import pytest

torch = pytest.importorskip("torch")

from torch_correctness_guards.guards.expand_fill import (
    ExpandFillCase,
    TorchUnavailableError,
    diagnose,
    safe_fill_,
)


def test_diagnose_returns_expected_shape():
    report = diagnose()
    assert "torch_version" in report
    assert report["torch_version"] == torch.__version__
    assert "cases" in report
    assert len(report["cases"]) == 4
    assert "any_expand_fill_divergence" in report
    assert "guard_fully_correct" in report


def test_native_bug_still_reproduces_on_this_torch_version():
    """Bug-injection proof: the native (unguarded) compiled path must
    actually diverge from eager on THIS host's installed torch build.
    If this ever starts failing, it means the upstream bug has been
    fixed and this repo's README/scope claims need re-evaluating --
    NOT that the test should be weakened to tolerate either outcome."""
    report = diagnose()
    assert report["any_expand_fill_divergence"] is True
    assert any(c["native_diverges"] for c in report["cases"])


def test_guard_matches_eager_on_every_case():
    report = diagnose()
    assert report["guard_fully_correct"] is True
    for case in report["cases"]:
        assert case["guard_matches_eager"] is True
        assert case["compiled_guarded_result"] == case["eager_result"]


def test_safe_fill_matches_eager_under_compile_single_case():
    def f_guarded(base):
        v = base.expand(3, -1)
        safe_fill_(v, 2.0)
        return base

    def f_eager(base):
        v = base.expand(3, -1)
        v.fill_(2.0)
        return base

    eager_out = f_eager(torch.tensor([-1.0, -0.5, 0.0, 0.5]))

    torch._dynamo.reset()
    compiled_guarded = torch.compile(f_guarded, backend="inductor")
    guarded_out = compiled_guarded(torch.tensor([-1.0, -0.5, 0.0, 0.5]))

    assert torch.equal(eager_out, guarded_out)
    assert torch.equal(eager_out, torch.tensor([2.0, 2.0, 2.0, 2.0]))


def test_unguarded_compiled_fill_diverges_from_eager_single_case():
    """Companion bug-injection test at unit granularity (not just via
    diagnose()'s aggregate fixtures): proves the native bug is real for
    a hand-written minimal case, independent of the fixture list."""

    def f_native(base):
        v = base.expand(3, -1)
        v.fill_(2.0)
        return base

    eager_out = f_native(torch.tensor([-1.0, -0.5, 0.0, 0.5]))

    torch._dynamo.reset()
    compiled_native = torch.compile(f_native, backend="inductor")
    native_out = compiled_native(torch.tensor([-1.0, -0.5, 0.0, 0.5]))

    assert not torch.equal(eager_out, native_out)


def test_diagnose_reports_issue_url():
    report = diagnose()
    assert "https://github.com/pytorch/pytorch/issues/197448" in report["issue_urls"]


def test_expand_fill_case_dataclass_fields():
    case = ExpandFillCase(
        expand_rows=3,
        base_values=[1.0, 2.0],
        fill_value=0.0,
        eager_result=[0.0, 0.0],
        compiled_native_result=[9.0, 9.0],
        compiled_guarded_result=[0.0, 0.0],
        native_diverges=True,
        guard_matches_eager=True,
    )
    assert case.expand_rows == 3
    assert case.native_diverges is True
    assert case.guard_matches_eager is True


def test_torch_unavailable_error_is_runtime_error():
    assert issubclass(TorchUnavailableError, RuntimeError)
