"""Tests for torch-compile-normal-dtype-promotion-guard. Requires the
'torch' extra (skipped otherwise).

Design mirrors this fleet's established discipline: every guard claim
is backed by a real reproduction, not an assumption, and at least one
test proves the test suite itself would have failed before the fix
(bug-injection verification), not just that the fix's own code path
returns success.
"""
from __future__ import annotations

import pytest

torch = pytest.importorskip("torch")

from torch_correctness_guards.guards.normal_dtype_promotion import (  # noqa: E402
    TorchUnavailableError,
    _normal_sample_fn,
    diagnose,
    safe_compiled_normal_sample,
)


def test_diagnose_runs_and_reports_torch_version():
    report = diagnose()
    assert report["torch_version"] == torch.__version__
    assert len(report["cases"]) == 4
    assert report["issue_url"] == "https://github.com/pytorch/pytorch/issues/194547"


def test_native_dtype_promotion_bug_is_actually_reproduced_on_this_host():
    """This is the core evidentiary claim for this tool: prove the
    torch.compile Normal.sample() dtype-promotion divergence is real
    on the CURRENTLY installed torch build, not merely cited from the
    issue tracker (pytorch/pytorch#194547). If torch fixes this
    upstream, this assertion should start failing -- news the tool
    should surface (via any_native_divergence), not silently pass."""
    report = diagnose()
    assert report["any_native_divergence"] is True, (
        "Expected the known upstream torch.compile Normal.sample() "
        "dtype-promotion bug (pytorch/pytorch#194547) to reproduce on "
        f"torch {torch.__version__}; if this now fails, the bug may "
        "have been fixed upstream -- verify against the issue tracker "
        "before assuming a test regression."
    )
    diverging = [c for c in report["cases"] if c["dtype_diverges"]]
    assert len(diverging) >= 1
    for c in diverging:
        # The divergence is specifically lower-precision loc promoted
        # to match a higher-precision scale.
        assert c["eager_dtype"] != c["compiled_dtype"]


def test_guard_fully_correct_across_all_cases():
    report = diagnose()
    assert report["guard_fully_correct"] is True
    for c in report["cases"]:
        assert c["guarded_matches_eager"], c


def test_safe_wrapper_restores_correct_dtype_directly():
    """Direct, minimal reproduction of the guard's core claim without
    going through diagnose(): the wrapped compiled function's return
    dtype must match eager's dtype (loc's dtype preserved)."""
    fn = _normal_sample_fn(torch)
    torch._dynamo.reset()
    compiled = torch.compile(fn, fullgraph=True)
    guarded = safe_compiled_normal_sample(compiled, fn)

    loc = torch.tensor([0.0, 1.0], dtype=torch.float16)
    scale = torch.tensor([1.0, 2.0], dtype=torch.float32)

    out = guarded(loc, scale)
    assert out.dtype == torch.float16, "guarded output must preserve loc's dtype"


def test_native_compiled_diverges_bug_injection_check():
    """Bug-injection check proving the regression tests above are
    real: deliberately call the RAW (unguarded) compiled function and
    confirm it DOES promote the dtype away from loc's float16 -- i.e.
    if safe_compiled_normal_sample were a no-op passthrough (the bug
    this tool guards against), the guard tests above would correctly
    fail. This proves those tests are not tautological."""
    fn = _normal_sample_fn(torch)
    torch._dynamo.reset()
    compiled = torch.compile(fn, fullgraph=True)

    loc = torch.tensor([0.0, 1.0], dtype=torch.float16)
    scale = torch.tensor([1.0, 2.0], dtype=torch.float32)

    out = compiled(loc, scale)
    assert out.dtype != torch.float16, (
        "Expected the RAW (unguarded) compiled function to promote "
        "the dtype away from loc's float16 (that is the whole "
        "dtype-promotion bug this tool detects, "
        "pytorch/pytorch#194547); if this assertion fails, the "
        "underlying bug may have disappeared upstream, which would "
        "make the guard tautologically pass for the wrong reason."
    )


def test_guard_preserves_correct_dtype_when_no_bug_present():
    """When compiled and eager already agree (e.g. loc/scale both
    float32, where there is nothing to promote), the guard must still
    return a dtype matching eager -- proving it is not merely
    coincidentally correct only in the presence of the bug."""
    fn = _normal_sample_fn(torch)
    torch._dynamo.reset()
    compiled = torch.compile(fn, fullgraph=True)
    guarded = safe_compiled_normal_sample(compiled, fn)

    loc = torch.tensor([0.0, 1.0], dtype=torch.float32)
    scale = torch.tensor([1.0, 2.0], dtype=torch.float32)

    out = guarded(loc, scale)
    assert out.dtype == torch.float32


def test_torch_unavailable_error_is_distinct_type():
    """Sanity check the error type exists and is a RuntimeError
    subclass, independent of whether torch is actually installed in
    this env."""
    assert issubclass(TorchUnavailableError, RuntimeError)
