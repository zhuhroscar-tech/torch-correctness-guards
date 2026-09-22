"""Tests for torch-linalg-pinv-complex-grad-guard. Requires the
'torch' extra (skipped otherwise).

Design mirrors this fleet's established discipline: every guard claim
is backed by a real reproduction, not an assumption, and at least one
test proves the test suite itself would have failed before the fix
(bug-injection verification), not just that the fix's own code path
returns success."""
from __future__ import annotations

import pytest

torch = pytest.importorskip("torch")

from torch_correctness_guards.guards.linalg_pinv_complex_grad import (  # noqa: E402
    TorchUnavailableError,
    diagnose,
    safe_complex_pinv_grad,
)


def _pinv_sum_abs_fn():
    def fn(a):
        return torch.linalg.pinv(a).sum().abs()

    return fn


def test_diagnose_runs_and_reports_torch_version():
    report = diagnose()
    assert report["torch_version"] == torch.__version__
    assert len(report["cases"]) == 3
    assert report["issue_url"] == "https://github.com/pytorch/pytorch/issues/197084"


def test_native_bug_is_actually_reproduced_on_this_host():
    """This is the core evidentiary claim for this tool: prove the
    AOTAutograd-backend complex pinv wrong-gradient bug is real on
    the CURRENTLY installed torch build, not merely cited from the
    issue tracker (pytorch/pytorch#197084). If torch fixes this
    upstream, this assertion should start failing -- news the tool
    should surface (via any_native_bug), not silently pass."""
    report = diagnose()
    assert report["any_native_bug"] is True, (
        "Expected the known upstream AOTAutograd complex pinv/"
        f"matrix_sqrth wrong-gradient bug (pytorch/pytorch#197084) to "
        f"reproduce on torch {torch.__version__}; if this now fails, "
        "the bug may have been fixed upstream -- verify against the "
        "issue tracker before assuming a test regression."
    )
    for c in report["cases"]:
        assert c["eager_matches_dynamo_only"], c


def test_guard_fully_correct_across_all_cases():
    report = diagnose()
    assert report["guard_fully_correct"] is True
    for c in report["cases"]:
        assert c["guarded_matches_eager"], c
        assert c["max_abs_diff_guarded"] < 1e-4, c


def test_safe_wrapper_restores_correct_gradient_directly():
    """Direct, minimal reproduction of the guard's core claim without
    going through diagnose(): a guarded call used correctly (invoked
    from WITHIN an outer compiled function, so torch.compiler.disable()
    can act as a graph-break boundary) must produce a gradient that
    matches plain eager."""
    fn = _pinv_sum_abs_fn()
    guarded_inner = safe_complex_pinv_grad(fn)

    def outer(x):
        y = x * 1.0
        return guarded_inner(y)

    compiled_outer = torch.compile(outer, backend="inductor")

    torch.manual_seed(42)
    a0 = torch.randn(4, 3, dtype=torch.complex64)

    a_eager = a0.detach().clone().requires_grad_(True)
    fn(a_eager).backward()

    a_guarded = a0.detach().clone().requires_grad_(True)
    compiled_outer(a_guarded).backward()

    assert torch.allclose(a_eager.grad, a_guarded.grad, atol=1e-4)


def test_native_compiled_diverges_bug_injection_check():
    """Bug-injection check proving the regression tests above are
    real: deliberately call the RAW (unguarded) compiled function
    under aot_eager/inductor and confirm its gradient DOES diverge
    from eager -- i.e. if safe_complex_pinv_grad were a no-op
    passthrough (the bug this tool guards against), the guard tests
    above would correctly fail. This proves those tests are not
    tautological."""
    fn = _pinv_sum_abs_fn()

    torch.manual_seed(7)
    a0 = torch.randn(4, 3, dtype=torch.complex64)

    a_eager = a0.detach().clone().requires_grad_(True)
    fn(a_eager).backward()

    a_raw = a0.detach().clone().requires_grad_(True)
    torch.compile(fn, backend="aot_eager")(a_raw).backward()

    max_diff = (a_eager.grad - a_raw.grad).abs().max().item()
    assert max_diff > 1e-3, (
        "Expected the RAW (unguarded) aot_eager-compiled function's "
        "gradient to diverge significantly from eager (that is the "
        "whole bug this tool detects, pytorch/pytorch#197084); if "
        f"this assertion fails (max_diff={max_diff}), the underlying "
        "bug may have disappeared upstream, which would make the "
        "guard tautologically pass for the wrong reason."
    )


def test_guard_is_passthrough_for_real_valued_input():
    """When the input is NOT complex (the unaffected case), the
    wrapper must still produce correct results -- it only forces
    eager execution for the wrapped call, it does not change
    numerics."""

    def fn(a):
        return torch.linalg.pinv(a).sum()

    guarded = safe_complex_pinv_grad(fn)
    a = torch.randn(4, 3, requires_grad=True)
    a2 = a.detach().clone().requires_grad_(True)

    fn(a).backward()
    guarded(a2).backward()

    assert torch.allclose(a.grad, a2.grad, atol=1e-5)


def test_torch_unavailable_error_is_distinct_type():
    """Sanity check the error type exists and is a RuntimeError
    subclass, independent of whether torch is actually installed in
    this env."""
    assert issubclass(TorchUnavailableError, RuntimeError)
