"""Tests for torch-memory-budget-rng-guard. Requires the 'torch' extra
(skipped otherwise).

Design mirrors this fleet's established discipline: every guard claim
is backed by a real reproduction against the currently installed torch
build, not an assumption, and at least one test proves the test suite
itself is not tautological (a direct bug-injection reproduction, plus
exception-path coverage of the restore-on-exit contract).
"""
from __future__ import annotations

import types

import pytest

torch = pytest.importorskip("torch")

from torch_correctness_guards.guards.memory_budget_rng import (  # noqa: E402
    TorchUnavailableError,
    _forced_budget,
    diagnose,
    safe_compile,
)


@pytest.fixture(scope="module")
def report():
    return diagnose()


def test_diagnose_reports_torch_version(report):
    assert report["torch_version"] == torch.__version__
    assert len(report["unguarded_cases"]) == 4
    assert len(report["guarded_cases"]) == 3


def test_unguarded_rng_recompute_bug_reproduced_on_this_host(report):
    """This is the core evidentiary claim for this tool: prove the
    activation_memory_budget RNG-recompute gradient bug is real on the
    CURRENTLY installed torch build, not merely cited from the issue
    tracker (pytorch/pytorch#190758). If torch fixes this upstream,
    this assertion should start failing -- news the tool should
    surface (via any_unguarded_rng_recompute_bug), not silently pass."""
    assert report["any_unguarded_rng_recompute_bug"] is True, (
        "Expected the known upstream torch.compile activation_memory_"
        "budget RNG-recompute gradient bug (pytorch/pytorch#190758) to "
        f"reproduce on torch {torch.__version__} at budget<1.0; if this "
        "now fails, the bug may have been fixed upstream -- verify "
        "against the issue tracker before assuming a test regression."
    )


def test_unguarded_budget_1_is_correct(report):
    """budget=1.0 (no forced recompute at all) is the one value the
    upstream issue's own ablation table reports as NOT buggy -- confirm
    that control case independently on this host."""
    case = next(c for c in report["unguarded_cases"] if c["budget"] == 1.0)
    assert case["matches"] is True, (
        "Expected budget=1.0 (the default, no partitioner-forced "
        "recompute) to be correct on this host; if this fails, either "
        "the reproduction harness itself is broken, or torch's default "
        "behavior at budget=1.0 has changed."
    )


def test_guard_fully_correct_across_all_tested_budgets(report):
    assert report["guard_fully_correct"] is True


def test_exact_upstream_reproducer_bug_injection():
    """Direct reproduction of pytorch/pytorch#190758's own minimal
    repro, run with NO guard and NO helper functions from this
    package at all -- proves the bug is real independent of this
    package's own diagnose() logic, so this suite is not tautological
    (a bug in diagnose() itself could otherwise make every assertion
    above pass for the wrong reason)."""
    import torch.nn.functional as F
    import torch._functorch.config as fc

    prior = getattr(fc, "activation_memory_budget", 1.0)
    try:
        fc.activation_memory_budget = 0.0
        w = torch.arange(1.0, 26.0).view(5, 5)

        def f(x):
            return F.dropout(x @ w, 0.5, True)

        compiled = torch.compile(f, backend="inductor")
        torch.manual_seed(42)
        x = torch.ones(2, 5, requires_grad=True)
        out = compiled(x)
        out.sum().backward()

        mask = (out.detach() != 0).float()
        expected = 2.0 * mask @ w.t()
        assert not torch.equal(x.grad, expected), (
            "Expected the unguarded upstream minimal reproducer to show "
            "a mismatched gradient (pytorch/pytorch#190758); if this now "
            "passes, the bug may have been fixed upstream -- verify "
            "against the issue tracker before assuming a regression."
        )
    finally:
        fc.activation_memory_budget = prior


def test_forced_budget_context_manager_restores_prior_value_on_success():
    fake_config = types.SimpleNamespace(activation_memory_budget=0.5)
    with _forced_budget(fake_config, 1.0):
        assert fake_config.activation_memory_budget == 1.0
    assert fake_config.activation_memory_budget == 0.5


def test_forced_budget_context_manager_restores_prior_value_on_exception():
    """Proves the restore-on-exit contract holds even when the wrapped
    call raises -- a guard that only restored on the happy path would
    leave the caller's global torch config silently corrupted after
    any exception, which is exactly the kind of silent-corruption bug
    this whole package exists to avoid introducing."""
    fake_config = types.SimpleNamespace(activation_memory_budget=0.3)
    with pytest.raises(ValueError):
        with _forced_budget(fake_config, 1.0):
            assert fake_config.activation_memory_budget == 1.0
            raise ValueError("boom")
    assert fake_config.activation_memory_budget == 0.3, (
        "activation_memory_budget must be restored even when the "
        "wrapped call raises."
    )


def test_safe_compile_returns_callable_wrapping_original_fn():
    def f(x):
        return x * 2

    guarded = safe_compile(f)
    assert callable(guarded)
    assert hasattr(guarded, "__wrapped_compiled__")


def test_safe_compile_restores_callers_budget_setting_after_call():
    """End-to-end (real torch, real compile): the caller's globally
    configured budget must be exactly as they left it after a guarded
    call returns, even though safe_compile forced it to 1.0 internally
    during the call."""
    import torch._functorch.config as fc

    def f(x):
        return x * 2

    guarded = safe_compile(f)
    fc.activation_memory_budget = 0.42
    guarded(torch.ones(3))
    assert fc.activation_memory_budget == 0.42, (
        "safe_compile must restore the caller's requested budget after "
        "the guarded call returns, not leave it permanently at 1.0."
    )
    fc.activation_memory_budget = 1.0  # leave the shared module-level config clean


def test_torch_unavailable_error_is_distinct_type():
    """Sanity check the error type exists and is a RuntimeError
    subclass, independent of whether torch is actually installed in
    this env."""
    assert issubclass(TorchUnavailableError, RuntimeError)
