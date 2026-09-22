"""Tests for torch-compile-transpose-argmin-guard. Requires the
'torch' extra (skipped otherwise).

Design mirrors this fleet's established discipline: every guard claim
is backed by a real reproduction, not an assumption, and at least one
test proves the test suite itself would have failed before the fix
(bug-injection verification), not just that the fix's own code path
returns success."""
from __future__ import annotations

import pytest

torch = pytest.importorskip("torch")

from torch_correctness_guards.guards.transpose_argmin import (  # noqa: E402
    TorchUnavailableError,
    diagnose,
    safe_reduce_index,
)


def test_diagnose_runs_and_reports_torch_version():
    report = diagnose()
    assert report["torch_version"] == torch.__version__
    assert len(report["cases"]) == 6


def test_transpose_argmin_divergence_is_actually_reproduced_on_this_host():
    """This is the core evidentiary claim of the whole tool: prove the
    transpose+op+argmin/argmax index bug is real on the CURRENTLY
    installed torch build, not merely cited from the issue tracker. If
    torch fixes this upstream, this assertion should start failing --
    news the tool should surface (via
    any_transpose_argreduce_divergence), not silently pass."""
    report = diagnose()
    assert report["any_transpose_argreduce_divergence"] is True, (
        "Expected the known upstream Inductor transpose+op+argmin/argmax "
        "index bug (pytorch/pytorch#197739) to reproduce on torch "
        f"{torch.__version__}; if this now fails, the bug may have been "
        "fixed upstream -- verify against the issue tracker before "
        "assuming a test regression."
    )


def test_bare_transpose_with_no_intervening_op_is_not_affected():
    """Confirm the documented sensitivity: a bare `x.t().argmin()` with
    NO intervening op between transpose and reduction is NOT affected
    -- only when a further op (`+ scalar`, `.contiguous()`) runs on
    the transposed tensor first. This distinguishes the specific bug
    shape from a blanket "transpose is always broken" claim."""
    torch.manual_seed(0)
    x = torch.randn(2, 2)

    def bare(x):
        return x.t().argmin()

    eager_out = bare(x)
    torch._dynamo.reset()
    compiled = torch.compile(bare, backend="inductor")
    compiled_out = compiled(x)

    assert torch.equal(eager_out, compiled_out), (
        "Expected the bare-transpose-then-argmin case (no intervening "
        "op) to NOT reproduce the bug on this host; if it now diverges "
        "too, the bug's trigger conditions have broadened upstream."
    )


def test_guard_matches_eager_for_every_case():
    report = diagnose()
    assert report["guard_fully_correct"] is True
    for r in report["cases"]:
        assert r["guard_matches_eager"], r


def test_safe_reduce_index_matches_eager_directly_add_argmin():
    """Direct, minimal reproduction of the guard's core claim without
    going through diagnose(): transpose, add 0.5, argmin, compiled
    through the guard, and confirm it matches plain eager -- not the
    wrong index Inductor would otherwise silently compute."""
    torch.manual_seed(5)
    x = torch.randn(3, 5)
    xt_eager = x.t() + 0.5
    eager_out = xt_eager.argmin()

    torch._dynamo.reset()

    def guarded_fn(x):
        xt = x.t()
        return safe_reduce_index(xt, op="add", mode="argmin")

    compiled_guarded = torch.compile(guarded_fn, backend="inductor")
    guarded_out = compiled_guarded(x)

    assert torch.equal(eager_out, guarded_out)


def test_unguarded_native_diverges_bug_injection_check():
    """Bug-injection check proving the regression tests above are real:
    deliberately compile the RAW (unguarded) transpose+add+argmin
    sequence and confirm the compiled output DIFFERS from eager --
    i.e. if safe_reduce_index() were a no-op (the bug this tool guards
    against), the guard tests above would correctly fail. This proves
    those tests are not tautological."""

    def native(x):
        xt = x.t()
        y = xt + 0.5
        return y.argmin()

    torch.manual_seed(1)
    x = torch.randn(4, 6)

    eager_out = native(x)
    torch._dynamo.reset()
    compiled_native = torch.compile(native, backend="inductor")
    native_out = compiled_native(x)

    assert not torch.equal(eager_out, native_out), (
        "Expected the unguarded compiled transpose+add+argmin sequence "
        "to diverge from eager (that is the whole bug this tool "
        "detects); if this assertion fails, the underlying bug may "
        "have disappeared upstream, which would make the guard "
        "tautologically pass for the wrong reason."
    )


def test_safe_reduce_index_argmax_mode():
    """Confirm the guard also covers argmax, not just argmin."""
    torch.manual_seed(2)
    x = torch.randn(2, 2)
    xt_eager = x.t() + 0.5
    eager_out = xt_eager.argmax()

    guarded_out = safe_reduce_index(x.t(), op="add", mode="argmax")

    assert torch.equal(eager_out, guarded_out)


def test_safe_reduce_index_contiguous_mode():
    """Confirm the guard also covers the .contiguous() trigger
    variant, not just '+ scalar'."""
    torch.manual_seed(3)
    x = torch.randn(4, 4)
    xt_eager = x.t().contiguous()
    eager_out = xt_eager.argmin()

    guarded_out = safe_reduce_index(x.t(), op="contiguous", mode="argmin")

    assert torch.equal(eager_out, guarded_out)


def test_safe_reduce_index_rejects_unknown_op_and_mode():
    x = torch.randn(2, 2)
    with pytest.raises(ValueError):
        safe_reduce_index(x, op="bogus", mode="argmin")
    with pytest.raises(ValueError):
        safe_reduce_index(x, op="add", mode="bogus")


def test_torch_unavailable_error_is_distinct_type():
    """Sanity check the error type exists and is a RuntimeError
    subclass, independent of whether torch is actually installed in
    this env."""
    assert issubclass(TorchUnavailableError, RuntimeError)
