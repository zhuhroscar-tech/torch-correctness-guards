"""Tests for torch-inductor-softmax-dim-guard. Requires the 'torch'
extra (skipped otherwise).

Design mirrors this fleet's established discipline: every guard claim
is backed by a real reproduction, not an assumption, and at least one
test proves the test suite itself would have failed before the fix
(bug-injection verification), not just that the fix's own code path
returns success."""
from __future__ import annotations

import pytest

torch = pytest.importorskip("torch")

from torch_correctness_guards.guards.softmax_dim import (  # noqa: E402
    TorchUnavailableError,
    _is_last_axis,
    diagnose,
    safe_softmax_attention,
)


def test_is_last_axis_positive_and_negative_dims():
    assert _is_last_axis(3, 4) is True
    assert _is_last_axis(-1, 4) is True
    assert _is_last_axis(0, 4) is False
    assert _is_last_axis(-2, 4) is False
    assert _is_last_axis(2, 3) is True


def test_diagnose_runs_and_reports_torch_version():
    report = diagnose()
    assert report["torch_version"] == torch.__version__
    assert len(report["cases"]) == 4


def test_softmax_dim_pattern_rewrite_is_actually_reproduced_on_this_host():
    """This is the core evidentiary claim of the whole tool: prove the
    non-last-axis softmax pattern-rewrite bug is real on the CURRENTLY
    installed torch build, not merely cited from the issue tracker. If
    torch fixes this upstream, this assertion should start failing --
    news the tool should surface (via any_softmax_dim_divergence), not
    silently pass."""
    report = diagnose()
    assert report["any_softmax_dim_divergence"] is True, (
        "Expected the known upstream Inductor attention pattern-rewrite "
        "bug (pytorch/pytorch#196468) to reproduce on torch "
        f"{torch.__version__}; if this now fails, the bug may have been "
        "fixed upstream -- verify against the issue tracker before "
        "assuming a test regression."
    )
    # Also confirm the specific FAILURE MODE described in the issue: the
    # compiled output isn't just "wrong", it specifically matches the
    # dim=-1 answer -- proving this is the documented pattern-rewrite
    # substitution, not some unrelated numerical drift.
    diverging_cases = [c for c in report["cases"] if c["native_diverges"]]
    assert diverging_cases, "expected at least one diverging case"
    assert all(c["native_matches_wrong_lastaxis"] for c in diverging_cases), (
        "Expected every diverging case's compiled output to match the "
        "WRONG dim=-1 answer (the documented pattern-rewrite "
        "substitution), not some other unrelated divergence."
    )


def test_guard_matches_eager_for_every_case():
    report = diagnose()
    assert report["guard_fully_correct"] is True
    for r in report["cases"]:
        assert r["guard_matches_eager"], r


def test_safe_softmax_attention_matches_eager_directly_dim0():
    """Direct, minimal reproduction of the guard's core claim without
    going through diagnose(): build a matmul-softmax-matmul block with
    dim=0, run it compiled through the guard, and confirm it matches a
    plain eager computation of the SAME dim -- not the wrong dim=-1
    answer Inductor's pattern-matcher would otherwise silently
    substitute."""
    torch.manual_seed(11)
    q = torch.randn(3, 4, 4, 6)
    k = torch.randn(3, 4, 4, 6)
    v = torch.randn(3, 4, 4, 6)

    scale = q.shape[-1] ** -0.5
    eager_dim0 = torch.nn.functional.softmax(torch.matmul(q, k.transpose(-2, -1)) * scale, dim=0) @ v

    torch._dynamo.reset()

    def guarded_fn(q, k, v):
        return safe_softmax_attention(q, k, v, dim=0)

    compiled_guarded = torch.compile(guarded_fn, backend="inductor")
    guarded_out = compiled_guarded(q, k, v)

    assert torch.equal(eager_dim0, guarded_out)


def test_unguarded_native_diverges_from_requested_dim_bug_injection_check():
    """Bug-injection check proving the regression tests above are real:
    deliberately compile the RAW (unguarded) attention block with
    dim=1 and confirm the compiled output DIFFERS from the eager
    dim=1 answer -- i.e. if safe_softmax_attention() were a no-op (the
    bug this tool guards against), the guard tests above would
    correctly fail. This proves those tests are not tautological."""

    def attn_native(q, k, v, dim):
        scale = q.shape[-1] ** -0.5
        scores = torch.matmul(q, k.transpose(-2, -1)) * scale
        return torch.nn.functional.softmax(scores, dim=dim) @ v

    torch.manual_seed(22)
    q = torch.randn(2, 3, 5, 4)
    k = torch.randn(2, 3, 5, 4)
    v = torch.randn(2, 3, 5, 4)

    eager_dim1 = attn_native(q, k, v, 1)

    torch._dynamo.reset()
    compiled_native = torch.compile(attn_native, backend="inductor")
    native_dim1 = compiled_native(q, k, v, 1)

    assert not torch.equal(eager_dim1, native_dim1), (
        "Expected the unguarded compiled attention block to diverge from "
        "eager for a non-last-axis softmax dim (that is the whole bug "
        "this tool detects); if this assertion fails, the underlying "
        "pattern-rewrite bug may have disappeared upstream, which would "
        "make the guard tautologically pass for the wrong reason."
    )


def test_safe_softmax_attention_last_axis_still_matches_eager():
    """The known bug only affects non-last-axis dims; confirm the guard
    also matches eager for the ORDINARY last-axis case (dim=-1), which
    is not itself affected by the pattern-rewrite bug but must not be
    broken by routing it through the guard."""
    torch.manual_seed(33)
    q = torch.randn(2, 3, 4, 5)
    k = torch.randn(2, 3, 4, 5)
    v = torch.randn(2, 3, 4, 5)

    scale = q.shape[-1] ** -0.5
    eager_lastaxis = torch.nn.functional.softmax(torch.matmul(q, k.transpose(-2, -1)) * scale, dim=-1) @ v

    guarded_out = safe_softmax_attention(q, k, v, dim=-1)

    assert torch.equal(eager_lastaxis, guarded_out)


def test_default_scale_matches_explicit_scale():
    """safe_softmax_attention's default scale (1/sqrt(head_dim)) must
    match the same computation done with an explicitly passed scale."""
    torch.manual_seed(44)
    q = torch.randn(1, 2, 3, 8)
    k = torch.randn(1, 2, 3, 8)
    v = torch.randn(1, 2, 3, 8)

    out_default = safe_softmax_attention(q, k, v, dim=0)
    out_explicit = safe_softmax_attention(q, k, v, dim=0, scale=8 ** -0.5)

    assert torch.equal(out_default, out_explicit)


def test_torch_unavailable_error_is_distinct_type():
    """Sanity check the error type exists and is a RuntimeError
    subclass, independent of whether torch is actually installed in
    this env."""
    assert issubclass(TorchUnavailableError, RuntimeError)
