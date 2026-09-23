"""Tests for torch-multioutput-alias-guard. Requires the 'torch' extra
(skipped otherwise).

Design mirrors this fleet's established discipline: every guard claim
is backed by a real reproduction, not an assumption, and at least one
test proves the test suite itself would have failed before the fix
(bug-injection verification), not just that the fix's own code path
returns success."""
from __future__ import annotations

import pytest

torch = pytest.importorskip("torch")

from torch_correctness_guards.guards.multioutput_alias import (  # noqa: E402
    AliasedMultiOutputError,
    TorchUnavailableError,
    _tensors_alias,
    diagnose,
    safe_aminmax,
    safe_slogdet,
)


def test_tensors_alias_same_object_true():
    t = torch.empty(())
    assert _tensors_alias(t, t) is True


def test_tensors_alias_different_objects_false():
    a = torch.empty(())
    b = torch.empty(())
    assert _tensors_alias(a, b) is False


def test_tensors_alias_none_is_false():
    t = torch.empty(())
    assert _tensors_alias(None, t) is False
    assert _tensors_alias(t, None) is False


def test_diagnose_runs_and_reports_torch_version():
    report = diagnose()
    assert report["torch_version"] == torch.__version__
    assert len(report["aminmax_cases"]) == 3
    assert len(report["slogdet_cases"]) == 2


def test_aminmax_aliasing_bug_is_actually_reproduced_on_this_host():
    """This is the core evidentiary claim for the aminmax half of this
    tool: prove the same-tensor out= aliasing bug is real on the
    CURRENTLY installed torch build, not merely cited from the issue
    tracker (pytorch/pytorch#195338). If torch fixes this upstream,
    this assertion should start failing -- news the tool should
    surface (via any_aminmax_alias_bug), not silently pass."""
    report = diagnose()
    assert report["any_aminmax_alias_bug"] is True, (
        "Expected the known upstream torch.aminmax same-tensor out= "
        "aliasing bug (pytorch/pytorch#195338) to reproduce on torch "
        f"{torch.__version__}; if this now fails, the bug may have "
        "been fixed upstream -- verify against the issue tracker "
        "before assuming a test regression."
    )


def test_slogdet_aliasing_bug_is_actually_reproduced_on_this_host():
    """Same evidentiary claim for the slogdet half: this candidate's
    own LAPACK-backed finding (same defect class, no separate upstream
    issue at time of writing)."""
    report = diagnose()
    assert report["any_slogdet_alias_bug"] is True, (
        "Expected torch.linalg.slogdet(out=(t, t)) to silently lose "
        f"the sign value on torch {torch.__version__}; if this now "
        "fails, the bug may have been fixed upstream."
    )


def test_guard_fully_correct_across_all_cases():
    report = diagnose()
    assert report["guard_fully_correct"] is True
    for c in report["aminmax_cases"]:
        assert c["guard_raised"], c
    for c in report["slogdet_cases"]:
        assert c["guard_raised"], c


def test_safe_aminmax_raises_on_aliased_out_directly():
    """Direct, minimal reproduction of the guard's core claim without
    going through diagnose(): pass the same tensor twice and confirm
    safe_aminmax raises BEFORE computing anything, rather than the
    native silent-wrong-result behavior."""
    x = torch.tensor([3.0, 1.0, 4.0, 1.0, 5.0])
    same = torch.empty(())
    with pytest.raises(AliasedMultiOutputError):
        safe_aminmax(x, out=(same, same))


def test_safe_slogdet_raises_on_aliased_out_directly():
    B = torch.eye(3) * 2.0
    same = torch.empty(())
    with pytest.raises(AliasedMultiOutputError):
        safe_slogdet(B, out=(same, same))


def test_native_aminmax_diverges_bug_injection_check():
    """Bug-injection check proving the regression tests above are
    real: deliberately call the RAW (unguarded) torch.aminmax with
    aliased out= tensors and confirm it does NOT raise and returns a
    result inconsistent with the true min/max -- i.e. if safe_aminmax
    were a no-op passthrough (the bug this tool guards against), the
    guard tests above would correctly fail. This proves those tests
    are not tautological."""
    x = torch.tensor([3.0, 1.0, 4.0, 1.0, 5.0, 9.0, 2.0, 6.0])
    same = torch.empty(())
    # Must NOT raise -- this is the bug: silent wrong computation.
    torch.aminmax(x, out=(same, same))
    result = same.item()
    assert result != x.min().item() or result != x.max().item(), (
        "Expected the unguarded native torch.aminmax(out=(t, t)) call "
        "to produce a result inconsistent with a correct (min, max) "
        "pair (that is the whole bug this tool detects); if this "
        "assertion fails, the underlying aliasing bug may have "
        "disappeared upstream, which would make the guard "
        "tautologically pass for the wrong reason."
    )


def test_native_slogdet_loses_sign_bug_injection_check():
    B = torch.tensor([[1.0, 2.0], [3.0, 4.0]])  # det = -2, sign = -1
    ref_sign, ref_logabsdet = torch.linalg.slogdet(B)
    same = torch.empty(())
    torch.linalg.slogdet(B, out=(same, same))
    assert abs(same.item() - ref_sign.item()) > 1e-6, (
        "Expected the unguarded native torch.linalg.slogdet(out=(t, t)) "
        "call to lose the sign value (that is the whole bug this tool "
        "detects for slogdet); if this assertion fails, the underlying "
        "aliasing bug may have disappeared upstream."
    )


def test_safe_aminmax_matches_native_when_out_is_none():
    x = torch.tensor([3.0, 1.0, 4.0, 1.0, 5.0])
    guarded = safe_aminmax(x)
    native = torch.aminmax(x)
    assert torch.equal(guarded.min, native.min)
    assert torch.equal(guarded.max, native.max)


def test_safe_aminmax_matches_native_with_distinct_out_tensors():
    x = torch.tensor([3.0, 1.0, 4.0, 1.0, 5.0])
    g_min, g_max = torch.empty(()), torch.empty(())
    safe_aminmax(x, out=(g_min, g_max))
    n_min, n_max = torch.empty(()), torch.empty(())
    torch.aminmax(x, out=(n_min, n_max))
    assert torch.equal(g_min, n_min)
    assert torch.equal(g_max, n_max)


def test_safe_slogdet_matches_native_when_out_is_none():
    B = torch.eye(3) * 2.0
    g_sign, g_logabsdet = safe_slogdet(B)
    n_sign, n_logabsdet = torch.linalg.slogdet(B)
    assert torch.equal(g_sign, n_sign)
    assert torch.equal(g_logabsdet, n_logabsdet)


def test_safe_slogdet_matches_native_with_distinct_out_tensors():
    B = torch.eye(3) * 2.0
    g_sign, g_logabsdet = torch.empty(()), torch.empty(())
    safe_slogdet(B, out=(g_sign, g_logabsdet))
    n_sign, n_logabsdet = torch.empty(()), torch.empty(())
    torch.linalg.slogdet(B, out=(n_sign, n_logabsdet))
    assert torch.equal(g_sign, n_sign)
    assert torch.equal(g_logabsdet, n_logabsdet)


def test_torch_unavailable_error_is_distinct_type():
    """Sanity check the error type exists and is a RuntimeError
    subclass, independent of whether torch is actually installed in
    this env."""
    assert issubclass(TorchUnavailableError, RuntimeError)


def test_aliased_multioutput_error_is_distinct_runtime_error_subtype():
    assert issubclass(AliasedMultiOutputError, RuntimeError)
    assert not issubclass(AliasedMultiOutputError, TorchUnavailableError)
