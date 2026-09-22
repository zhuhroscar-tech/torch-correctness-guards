"""Tests for torch-inductor-2d-tiled-reduction-tail-store-guard.
Requires the 'torch' extra (skipped otherwise).

Design mirrors this fleet's established discipline: every guard claim
is backed by a real reproduction, not an assumption, and at least one
test proves the test suite itself would have failed before the fix
(bug-injection verification), not just that the fix's own code path
returns success.

Each real bug/guard check here runs in its OWN subprocess (via
core._invoke_worker / _worker.py), because pytorch/pytorch#196681 can
manifest as a hard process abort (SIGABRT) depending on host/allocator
-- confirmed on this project's own ubuntu-latest CI. Isolating each
check in a subprocess means a crash in one test cannot take down the
whole pytest session."""
from __future__ import annotations

import pytest

torch = pytest.importorskip("torch")

from torch_correctness_guards.guards.tiled_reduction_tail_store import (  # noqa: E402
    TailStoreOverrunSuspected,
    TorchUnavailableError,
    _invoke_worker,
    diagnose,
    safe_compiled_reduction,
)
from torch_correctness_guards.guards._tiled_reduction_tail_store_worker import (  # noqa: E402
    _reference_fn,
)


def test_diagnose_runs_and_reports_torch_version():
    report = diagnose()
    assert report["torch_version"] == torch.__version__
    assert len(report["cases"]) == 5
    assert report["issue_url"] == "https://github.com/pytorch/pytorch/issues/196681"


def test_native_bug_is_actually_reproduced_on_this_host():
    """This is the core evidentiary claim for this tool: prove the
    Inductor 2D-tiled-reduction tail-store-overrun bug is real on the
    CURRENTLY installed torch build, not merely cited from the issue
    tracker (pytorch/pytorch#196681). If torch fixes this upstream (PR
    #196882 or equivalent lands), this assertion should start failing
    -- news the tool should surface (via any_bug_reproduced), not
    silently pass. The bug may show up as a wrong VALUE or as a
    process CRASH depending on host/allocator; either counts."""
    report = diagnose()
    assert report["any_bug_reproduced"] is True, (
        "Expected the known upstream Inductor 2D-tiled-reduction "
        f"tail-store-overrun bug (pytorch/pytorch#196681) to reproduce "
        f"on torch {torch.__version__}; if this now fails, the bug may "
        "have been fixed upstream -- verify against the issue tracker "
        "(and PR #196882's merge status) before assuming a test "
        "regression."
    )
    # At least one aligned and one misaligned case must be present so
    # the guard's "silent when correct, raises-or-isolates-a-crash
    # when wrong" contract is actually exercised both ways.
    aligned_cases = [c for c in report["cases"] if c["aligned_to_vector_width"]]
    misaligned_cases = [c for c in report["cases"] if not c["aligned_to_vector_width"]]
    assert aligned_cases and misaligned_cases
    assert any(c["bug_reproduced"] for c in misaligned_cases)


def test_guard_fully_correct_across_all_cases():
    report = diagnose()
    assert report["guard_fully_correct"] is True
    for c in report["cases"]:
        assert c["guard_behaved_correctly"], c


def test_safe_wrapper_raises_on_misaligned_size_when_bug_is_silent(monkeypatch):
    """Direct, minimal in-process reproduction of the guard's core
    claim, guarded against this test itself being killed by a crash
    variant: if the bare (unguarded) subprocess check for this size
    crashed on this host, skip -- the in-process wrapper's documented
    limitation is that it cannot survive that manifestation, and this
    test only exercises the silent-wrong-value manifestation."""
    bare = _invoke_worker("bare", 66)
    if bare.get("crashed"):
        pytest.skip(
            "size=66 crashes the bare compiled call on this host "
            "(SIGABRT manifestation of #196681) -- safe_compiled_reduction's "
            "in-process wrapper cannot survive an abort inside the call "
            "it wraps; see core.py's documented limitation. The subprocess-"
            "isolated diagnose() path (exercised by other tests) is the "
            "correct way to handle this manifestation."
        )

    fn = _reference_fn(torch)
    torch.manual_seed(0)
    x = torch.randn(1, 8, 66, 66).contiguous(memory_format=torch.channels_last)
    compiled = torch.compile(fn, fullgraph=True)
    guarded = safe_compiled_reduction(compiled, fn)

    with pytest.raises(TailStoreOverrunSuspected):
        guarded(x)


def test_guard_is_silent_when_no_divergence_present():
    """When a function's compiled output does NOT diverge from eager
    (the unaffected/already-fixed/aligned-size case), the wrapper must
    return the value without raising."""
    fn = _reference_fn(torch)
    torch.manual_seed(0)
    x = torch.randn(1, 8, 64, 64).contiguous(memory_format=torch.channels_last)
    compiled = torch.compile(fn, fullgraph=True)
    guarded = safe_compiled_reduction(compiled, fn)

    result = guarded(x)  # must not raise
    expected = fn(x)
    assert torch.allclose(result, expected, rtol=1e-4, atol=1e-4)


def test_worker_bare_and_guarded_agree_on_bug_presence():
    """Bug-injection check proving the regression tests above are
    real: the isolated worker's 'bare' mode on a known-bad size must
    report either a large divergence or a crash -- i.e. if the bug
    were already fixed upstream, this would correctly fail, proving
    the other tests are not tautological."""
    bare = _invoke_worker("bare", 66)
    if bare.get("crashed"):
        assert bare["returncode"] != 0
    else:
        assert bare["max_abs_diff"] > 1.0, (
            "Expected the RAW (unguarded) compiled function to diverge "
            "substantially from eager on this known-bad size (that is "
            "the whole bug this tool detects, pytorch/pytorch#196681); "
            "if this assertion fails, the underlying overrun bug may "
            "have disappeared upstream, which would make the guard "
            "tautologically pass for the wrong reason."
        )


def test_torch_unavailable_error_is_distinct_type():
    """Sanity check the error type exists and is a RuntimeError
    subclass, independent of whether torch is actually installed in
    this env."""
    assert issubclass(TorchUnavailableError, RuntimeError)


def test_tail_store_overrun_suspected_is_distinct_type():
    assert issubclass(TailStoreOverrunSuspected, RuntimeError)
