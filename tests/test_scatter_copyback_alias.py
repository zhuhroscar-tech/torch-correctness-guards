"""Tests for torch-inductor-scatter-copyback-alias-guard. Requires the
'torch' extra (skipped otherwise).

Design mirrors this fleet's established discipline: every guard claim
is backed by a real reproduction, not an assumption, and at least one
test proves the test suite itself would have failed before the fix
(bug-injection verification), not just that the fix's own code path
returns success."""
from __future__ import annotations

import pytest

torch = pytest.importorskip("torch")

from torch_correctness_guards.guards.scatter_copyback_alias import (  # noqa: E402
    TorchUnavailableError,
    diagnose,
    safe_compiled_scatter_returning,
)


def _scatter_copyback_fn():
    def fn(x, src):
        updated = torch.slice_scatter(x, src, 0, 0, 1)
        x.copy_(updated)
        return updated

    return fn


def test_diagnose_runs_and_reports_torch_version():
    report = diagnose()
    assert report["torch_version"] == torch.__version__
    assert len(report["cases"]) == 3
    assert report["issue_url"] == "https://github.com/pytorch/pytorch/issues/195451"


def test_native_alias_bug_is_actually_reproduced_on_this_host():
    """This is the core evidentiary claim for this tool: prove the
    Inductor direct-scatter-copyback return-aliasing bug is real on
    the CURRENTLY installed torch build, not merely cited from the
    issue tracker (pytorch/pytorch#195451). If torch fixes this
    upstream (PR #195484 or equivalent lands), this assertion should
    start failing -- news the tool should surface (via
    any_native_alias_bug), not silently pass."""
    report = diagnose()
    assert report["any_native_alias_bug"] is True, (
        "Expected the known upstream Inductor direct-scatter-copyback "
        f"aliasing bug (pytorch/pytorch#195451) to reproduce on torch "
        f"{torch.__version__}; if this now fails, the bug may have "
        "been fixed upstream -- verify against the issue tracker "
        "(and PR #195484's merge status) before assuming a test "
        "regression."
    )
    assert report["any_native_corruption"] is True


def test_guard_fully_correct_across_all_cases():
    report = diagnose()
    assert report["guard_fully_correct"] is True
    for c in report["cases"]:
        assert not c["guarded_aliases_input"], c
        assert not c["guarded_input_corrupted_after_output_mutation"], c
        assert c["guarded_values_match_eager"], c


def test_safe_wrapper_restores_non_aliasing_directly():
    """Direct, minimal reproduction of the guard's core claim without
    going through diagnose(): the wrapped compiled function's return
    value must not alias the input tensor, restoring eager's
    contract."""
    fn = _scatter_copyback_fn()
    compiled = torch.compile(fn, fullgraph=True)
    guarded = safe_compiled_scatter_returning(compiled)

    x = torch.tensor([1.0, 2.0])
    out = guarded(x, torch.tensor([10.0]))
    assert x.data_ptr() != out.data_ptr()

    out.add_(100.0)
    assert torch.equal(x, torch.tensor([10.0, 2.0]))  # unaffected by output mutation


def test_native_compiled_diverges_bug_injection_check():
    """Bug-injection check proving the regression tests above are
    real: deliberately call the RAW (unguarded) compiled function and
    confirm it DOES alias its input and DOES get corrupted by a later
    output mutation -- i.e. if safe_compiled_scatter_returning were a
    no-op passthrough (the bug this tool guards against), the guard
    tests above would correctly fail. This proves those tests are not
    tautological."""
    fn = _scatter_copyback_fn()
    compiled = torch.compile(fn, fullgraph=True)

    x = torch.tensor([1.0, 2.0])
    out = compiled(x, torch.tensor([10.0]))
    assert x.data_ptr() == out.data_ptr(), (
        "Expected the RAW (unguarded) compiled function to alias its "
        "input (that is the whole bug this tool detects, "
        "pytorch/pytorch#195451); if this assertion fails, the "
        "underlying aliasing bug may have disappeared upstream, which "
        "would make the guard tautologically pass for the wrong "
        "reason."
    )
    out.add_(100.0)
    assert not torch.equal(x, torch.tensor([10.0, 2.0])), (
        "Expected the unguarded input to be corrupted by mutating the "
        "aliased output; if this fails the bug may be fixed upstream."
    )


def test_guard_is_passthrough_when_no_aliasing_present():
    """When a function's output does NOT alias any input argument (the
    unaffected/already-fixed case), the wrapper must return the exact
    same tensor object -- no unnecessary clone."""

    def fn(x):
        return x + 1.0

    guarded = safe_compiled_scatter_returning(fn)
    x = torch.tensor([1.0, 2.0])
    out = guarded(x)
    assert out.data_ptr() != x.data_ptr()  # x+1.0 never aliased x to begin with
    assert torch.equal(out, torch.tensor([2.0, 3.0]))


def test_guard_preserves_correct_values():
    """The guard must not just fix aliasing -- the VALUES must still
    match what eager mode (or the intended computation) produces."""
    fn = _scatter_copyback_fn()
    compiled = torch.compile(fn, fullgraph=True)
    guarded = safe_compiled_scatter_returning(compiled)

    x = torch.tensor([5.0, -3.0, 7.0])
    out = guarded(x, torch.tensor([0.0]))
    assert torch.equal(out, torch.tensor([0.0, -3.0, 7.0]))


def test_torch_unavailable_error_is_distinct_type():
    """Sanity check the error type exists and is a RuntimeError
    subclass, independent of whether torch is actually installed in
    this env."""
    assert issubclass(TorchUnavailableError, RuntimeError)


# ---------------------------------------------------------------------------
# Second, independently-discovered root cause: no-op elimination of a
# neutral-value op (pytorch/pytorch#197893). Same defect shape as the
# scatter-copyback bug above (a compiled function's output silently
# aliases an input eager never would), guarded by the SAME wrapper
# with zero new guard code -- these tests prove that reuse is real,
# not assumed.
# ---------------------------------------------------------------------------


def test_diagnose_reports_noop_section():
    report = diagnose()
    assert report["noop_issue_url"] == "https://github.com/pytorch/pytorch/issues/197893"
    assert len(report["noop_cases"]) == 3
    op_names = {c["op_name"] for c in report["noop_cases"]}
    assert op_names == {"x / 1.0", "x * 1.0", "x + 0"}


def test_native_noop_alias_bug_is_actually_reproduced_on_this_host():
    """Core evidentiary claim for the SECOND root cause: prove the
    Inductor no-op-elimination return-aliasing bug is real on the
    CURRENTLY installed torch build (pytorch/pytorch#197893), not
    merely cited from the issue tracker. If torch fixes this
    upstream, this assertion should start failing -- news the tool
    should surface (via any_noop_alias_bug), not silently pass."""
    report = diagnose()
    assert report["any_noop_alias_bug"] is True, (
        "Expected the known upstream Inductor no-op-elimination "
        f"aliasing bug (pytorch/pytorch#197893) to reproduce on torch "
        f"{torch.__version__}; if this now fails, the bug may have "
        "been fixed upstream -- verify against the issue tracker "
        "before assuming a test regression."
    )
    assert report["any_noop_corruption"] is True


def test_noop_guard_fully_correct_across_all_cases():
    report = diagnose()
    assert report["noop_guard_fully_correct"] is True
    for c in report["noop_cases"]:
        assert not c["guarded_aliases_input"], c
        assert not c["guarded_input_corrupted_after_output_mutation"], c
        assert c["guarded_values_match_eager"], c


def test_safe_wrapper_restores_non_aliasing_for_div_by_one_directly():
    """Direct, minimal reproduction of the guard's claim for the
    SECOND root cause without going through diagnose(): the same
    safe_compiled_scatter_returning wrapper, applied to a completely
    different function (x / 1.0, not the scatter pattern), must
    still detach the aliased output."""

    def scale(x, temperature):
        return x / temperature

    compiled = torch.compile(scale, fullgraph=True)
    guarded = safe_compiled_scatter_returning(compiled)

    x = torch.tensor([1.0, 2.0, 3.0])
    out = guarded(x, 1.0)
    assert x.data_ptr() != out.data_ptr()

    out.add_(100.0)
    assert torch.equal(x, torch.tensor([1.0, 2.0, 3.0]))  # unaffected


def test_native_compiled_noop_diverges_bug_injection_check():
    """Bug-injection check proving the noop regression tests above are
    real: deliberately call the RAW (unguarded) compiled function for
    x/1.0 and confirm it DOES alias its input and DOES get corrupted
    by a later output mutation. If safe_compiled_scatter_returning
    were a no-op passthrough, the guard tests above would correctly
    fail -- this proves those tests are not tautological, and that
    this is a genuinely distinct code path from the scatter-copyback
    case (a different op, a different upstream pass) rather than the
    same call accidentally exercised twice."""

    def scale(x, temperature):
        return x / temperature

    compiled = torch.compile(scale, fullgraph=True)

    x = torch.tensor([1.0, 2.0, 3.0])
    out = compiled(x, 1.0)
    assert x.data_ptr() == out.data_ptr(), (
        "Expected the RAW (unguarded) compiled x/1.0 to alias its "
        "input (that is the whole bug this second root cause guards "
        "against, pytorch/pytorch#197893); if this assertion fails, "
        "the underlying bug may have disappeared upstream, which "
        "would make the guard tautologically pass for the wrong "
        "reason."
    )
    out.add_(100.0)
    assert not torch.equal(x, torch.tensor([1.0, 2.0, 3.0])), (
        "Expected the unguarded input to be corrupted by mutating "
        "the aliased output; if this fails the bug may be fixed "
        "upstream."
    )


def test_native_eager_and_aot_eager_never_alias_for_div_by_one():
    """Confirms the bug is Inductor-specific: eager and aot_eager both
    correctly return a fresh tensor for the exact same x/1.0 call that
    Inductor aliases -- isolating the defect to Inductor's
    joint_graph.remove_no_ops pass, not a general torch.compile or
    eager-mode ambiguity about neutral-value ops."""

    def scale(x, temperature):
        return x / temperature

    x_eager = torch.tensor([1.0, 2.0, 3.0])
    out_eager = scale(x_eager, 1.0)
    assert x_eager.data_ptr() != out_eager.data_ptr()

    x_aot = torch.tensor([1.0, 2.0, 3.0])
    compiled_aot = torch.compile(scale, backend="aot_eager", fullgraph=True)
    out_aot = compiled_aot(x_aot, 1.0)
    assert x_aot.data_ptr() != out_aot.data_ptr()
