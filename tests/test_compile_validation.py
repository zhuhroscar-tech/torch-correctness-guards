"""Tests for the consolidated compile-validation guard. Requires the 'torch'
extra (skipped otherwise).

Design mirrors this fleet's established discipline: every guard claim
is backed by a real reproduction (via an isolated subprocess per
fixture, see core._run_fixture_via_subprocess / _worker.py), and at
least one test proves the test suite itself would have failed before
the fix (bug-injection verification), not just that the fix's own
code path returns success.

NOTE ON SUBPROCESS ISOLATION: this project's own harness discovered
(while being built) that tracing a torch.distributions-based function
through torch.compile flips a PROCESS-GLOBAL validate-args flag,
silently corrupting a LATER fixture's eager baseline if measured in
the same process. Every diagnose()/worker call below therefore
spawns real subprocesses; these tests are slower (each `diagnose()`
call takes ~15-40s, spawning 10 python+torch subprocesses) but are
the only way to get an honest per-fixture measurement.
"""
from __future__ import annotations

import pytest

torch = pytest.importorskip("torch")

from torch_correctness_guards.guards.compile_validation import (  # noqa: E402
    TorchUnavailableError,
    _fixture_specs,
    _run_fixture_via_subprocess,
    _validate_bernoulli_p,
    _validate_bce_target_scalar,
    _validate_normal_std,
    diagnose,
    safe_compiled_bernoulli,
    safe_compiled_binary_cross_entropy,
    safe_compiled_categorical_sample,
    safe_compiled_normal,
    safe_compiled_upsample_bilinear2d,
)


def test_diagnose_runs_and_reports_torch_version():
    report = diagnose()
    assert report["torch_version"] == torch.__version__
    assert len(report["cases"]) == 10
    assert len(report["issue_urls"]) == 5


def test_all_five_validation_bypasses_are_actually_reproduced_on_this_host():
    """This is the core evidentiary claim for this tool: prove each of
    the 5 upstream torch.compile input-validation-bypass bugs is real
    on the CURRENTLY installed torch build, not merely cited from the
    issue trackers (pytorch/pytorch#185246, #185248, #193757, #194548,
    #193811). If torch fixes any of these upstream, the corresponding
    assertion should start failing -- news the tool should surface via
    any_validation_bypassed / per-case validation_bypassed, not
    silently pass."""
    report = diagnose()
    assert report["any_validation_bypassed"] is True

    invalid_cases = [c for c in report["cases"] if not c["is_boundary_case"]]
    assert len(invalid_cases) == 5
    bypassed_names = {c["name"] for c in invalid_cases if c["validation_bypassed"]}
    assert bypassed_names == {
        "bernoulli_p_out_of_range",
        "normal_negative_std",
        "bce_target_out_of_range",
        "categorical_zero_batch",
        "upsample_bilinear2d_int_dtype",
    }, (
        "Expected all 5 known upstream validation-bypass bugs to "
        f"reproduce on torch {torch.__version__}; if any is missing "
        "here, it may have been fixed upstream -- verify against the "
        "issue tracker before assuming a test regression."
    )
    for c in invalid_cases:
        assert c["eager_raised"] is True, c
        assert c["compiled_raised"] is False, c


def test_guard_fully_correct_across_all_cases():
    report = diagnose()
    assert report["guard_fully_correct"] is True
    for c in report["cases"]:
        assert c["guarded_raised"] == c["eager_raised"], c


def test_boundary_cases_never_spuriously_flagged():
    """The valid boundary inputs (p=1.0, std=0.0, target=1.0,
    non-empty batch, float dtype) must never be rejected by eager,
    compiled, OR the guard -- proving the guard's validators are not
    simply "reject everything"."""
    report = diagnose()
    assert report["boundary_cases_not_spuriously_flagged"] is True
    boundary_cases = [c for c in report["cases"] if c["is_boundary_case"]]
    assert len(boundary_cases) == 5
    for c in boundary_cases:
        assert c["eager_raised"] is False, c
        assert c["compiled_raised"] is False, c
        assert c["guarded_raised"] is False, c


@pytest.mark.parametrize(
    "fixture_name",
    [
        "bernoulli_p_out_of_range",
        "normal_negative_std",
        "bce_target_out_of_range",
        "categorical_zero_batch",
        "upsample_bilinear2d_int_dtype",
    ],
)
def test_individual_invalid_fixture_via_subprocess(fixture_name):
    """Directly exercise the subprocess-isolation path (not just
    diagnose()'s aggregate) for each invalid-input fixture."""
    result = _run_fixture_via_subprocess(fixture_name)
    assert result.name == fixture_name
    assert result.eager_raised is True
    assert result.compiled_raised is False
    assert result.validation_bypassed is True
    assert result.guarded_raised is True
    assert result.guard_restores_validation is True


def test_fixture_specs_cover_all_five_issues():
    specs = _fixture_specs(torch)
    issue_urls = {spec["issue_url"] for spec in specs}
    assert issue_urls == {
        "https://github.com/pytorch/pytorch/issues/185246",
        "https://github.com/pytorch/pytorch/issues/185248",
        "https://github.com/pytorch/pytorch/issues/193757",
        "https://github.com/pytorch/pytorch/issues/194548",
        "https://github.com/pytorch/pytorch/issues/193811",
    }
    assert len(specs) == 10  # 5 invalid + 5 boundary


# --- Direct validator unit tests (no torch.compile needed) ----------------


def test_validate_bernoulli_p_rejects_out_of_range():
    with pytest.raises(RuntimeError):
        _validate_bernoulli_p(2.0)
    with pytest.raises(RuntimeError):
        _validate_bernoulli_p(-0.1)


def test_validate_bernoulli_p_accepts_boundary():
    _validate_bernoulli_p(0.0)
    _validate_bernoulli_p(1.0)
    _validate_bernoulli_p(0.5)


def test_validate_normal_std_rejects_negative():
    with pytest.raises(RuntimeError):
        _validate_normal_std(-1.0)


def test_validate_normal_std_accepts_boundary():
    _validate_normal_std(0.0)
    _validate_normal_std(2.5)


def test_validate_bce_target_rejects_out_of_range():
    with pytest.raises(RuntimeError):
        _validate_bce_target_scalar(1.5)
    with pytest.raises(RuntimeError):
        _validate_bce_target_scalar(-0.1)


def test_validate_bce_target_accepts_boundary():
    _validate_bce_target_scalar(0.0)
    _validate_bce_target_scalar(1.0)


# --- Guard wrapper bug-injection checks -------------------------------------


def test_raw_compiled_bernoulli_bypasses_validation_bug_injection_check():
    """Bug-injection check proving the regression tests above are
    real: deliberately call the RAW (unguarded) compiled function and
    confirm it does NOT raise for an out-of-domain p -- i.e. if
    safe_compiled_bernoulli were a no-op passthrough (the bug this
    tool guards against), the guard tests above would correctly fail.
    This proves those tests are not tautological."""

    def fn(p_tensor):
        return torch.bernoulli(p_tensor.expand(4).clone())

    torch._dynamo.reset()
    compiled = torch.compile(fn, fullgraph=True)
    # Must NOT raise -- this is the bug.
    out = compiled(torch.tensor(2.0))
    assert out is not None


def test_guarded_bernoulli_raises_where_raw_compiled_does_not():
    def fn(p_tensor):
        return torch.bernoulli(p_tensor.expand(4).clone())

    torch._dynamo.reset()
    compiled = torch.compile(fn, fullgraph=True)
    guarded = safe_compiled_bernoulli(compiled)
    with pytest.raises(RuntimeError):
        guarded(torch.tensor(2.0))


def test_guarded_bernoulli_does_not_raise_for_valid_p():
    def fn(p_tensor):
        return torch.bernoulli(p_tensor.expand(4).clone())

    torch._dynamo.reset()
    compiled = torch.compile(fn, fullgraph=True)
    guarded = safe_compiled_bernoulli(compiled)
    out = guarded(torch.tensor(0.5))
    assert out.shape == (4,)


def test_torch_unavailable_error_is_distinct_type():
    assert issubclass(TorchUnavailableError, RuntimeError)


def test_worker_subprocess_crash_is_reported_not_swallowed(monkeypatch):
    """If the worker subprocess exits non-zero (crash, uncaught
    exception unrelated to the validation check itself), diagnose()
    must raise, not silently report success."""
    import subprocess

    class _FakeCompletedProcess:
        returncode = 1
        stdout = ""
        stderr = "boom: something unrelated crashed"

    def _fake_run(*args, **kwargs):
        return _FakeCompletedProcess()

    monkeypatch.setattr(subprocess, "run", _fake_run)
    with pytest.raises(RuntimeError, match="worker subprocess exited"):
        _run_fixture_via_subprocess("bernoulli_p_out_of_range")
