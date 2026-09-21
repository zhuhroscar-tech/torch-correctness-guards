"""Regression tests for torch-compile-dynamic-clamp-guard.

These prove:
  1. The bug is real and reproducible from scratch on this host's
     installed torch build: torch.compile(backend="inductor") silently
     reuses a stale Python-float clamp bound across a sequence of
     calls where a later call shares the same tensor
     shape/dtype/requires_grad signature as an earlier call but passes
     a DIFFERENT runtime float value -- exactly the invocation order
     from pytorch/pytorch#194976's own repro.
  2. safe_clamp() is an independently verified fix: it matches eager's
     own output (the independent oracle -- eager always uses the
     correct, current Python float) on every call in the sequence,
     including under torch.compile.
  3. A bug-injection test proves the stale-reuse detection is
     non-tautological: an UNGUARDED compiled clamp (not wrapped in
     safe_clamp) really does diverge from eager on the third call,
     confirming the test harness would catch a regression if the
     guard were removed or broken.
"""
from __future__ import annotations

import pytest

torch = pytest.importorskip("torch")

from torch_correctness_guards.guards.dynamic_clamp import diagnose, safe_clamp


class TestNativeBugReproduction:
    def test_stale_reuse_reproduces_on_this_host(self):
        # Not asserted unconditionally true forever: if a future torch
        # release fixes Inductor's stale-value reuse for automatically
        # dynamic floats, this docstring is the record that the bug
        # existed at the version noted in the ledger/README. On torch
        # 2.14.0 (this host) it reproduces reliably.
        report = diagnose()
        assert report["any_stale_reuse_bug"] is True, (
            f"expected a stale dynamic-float clamp reuse divergence on "
            f"torch {report['torch_version']}; if this now fails, the "
            "bug may be fixed upstream (pytorch/pytorch#194976) -- "
            "update the README/ledger accordingly rather than treating "
            "this as a regression"
        )

    def test_third_call_is_the_one_that_diverges(self):
        # The upstream repro's own invocation order: only the THIRD
        # call (same shape/requires_grad as the second, different
        # limit) exhibits the staleness; the first two calls are
        # correct because they establish fresh compiled state.
        report = diagnose()
        cases = report["cases"]
        assert len(cases) == 3
        assert cases[0]["stale_reuse_bug"] is False
        assert cases[1]["stale_reuse_bug"] is False
        assert cases[2]["stale_reuse_bug"] is True, (
            "expected the third call (shape=(1,4), requires_grad=True, "
            "limit=1.0, immediately following an identical-signature "
            "call with limit=0.5) to reproduce the stale-value reuse"
        )

    def test_diverging_call_compiled_value_matches_prior_calls_limit(self):
        # Directly demonstrate the "stale" characterization: the
        # compiled result for call 2 (limit=1.0) numerically matches
        # what eager WOULD have produced for the earlier limit=0.5,
        # not an arbitrary wrong number.
        report = diagnose()
        case = report["cases"][2]
        assert case["limit"] == 1.0
        x = torch.full((1, 4), 0.75, dtype=torch.float32)
        eager_with_stale_limit = torch.clamp(x, max=0.5).abs().sum().item()
        assert abs(case["compiled_value"] - eager_with_stale_limit) < 1e-6, (
            "expected the compiled (buggy) value for limit=1.0 to "
            "numerically match eager's result for the PRIOR call's "
            "limit=0.5, confirming genuine stale-value reuse rather "
            "than an unrelated wrong answer"
        )


class TestBugInjectionNonTautological:
    """Prove the stale-reuse assertions above are not tautological by
    running the UNGUARDED compiled sequence directly (bypassing
    safe_clamp entirely) and confirming it really does diverge from
    eager on the third call -- i.e. the test harness would catch a
    regression if safe_clamp stopped being applied."""

    def test_unguarded_compiled_clamp_diverges_from_eager(self):
        torch._dynamo.reset()

        def eager_fn(x, limit):
            return torch.clamp(x, max=limit)

        compiled_fn = torch.compile(eager_fn, backend="inductor")

        with torch._dynamo.config.patch(specialize_float=False):
            x0 = torch.full((1, 2), 0.75, dtype=torch.float32)
            compiled_fn(x0, 1.0)

            x1 = torch.full((1, 4), 0.75, dtype=torch.float32, requires_grad=True)
            compiled_fn(x1, 0.5)

            x2 = torch.full((1, 4), 0.75, dtype=torch.float32, requires_grad=True)
            eager_val = eager_fn(x2.detach().clone(), 1.0).abs().sum().item()
            compiled_val = compiled_fn(x2, 1.0).abs().sum().item()

        assert abs(compiled_val - eager_val) > 1e-6, (
            "sanity check: the unguarded compiled clamp really must "
            "diverge from eager on this host for this test suite's "
            "positive assertions to be meaningful evidence of a real "
            "bug, not a tautology"
        )

    def test_unguarded_compiled_clamp_min_bound_also_diverges(self):
        # The README/module docstring use `max` as the concrete repro,
        # but `min` shares the exact same Inductor scalar-tensorization
        # code path and was never independently verified to reproduce
        # the native bug under torch.compile -- only exercised against
        # eager (see test_safe_clamp_min_bound_also_matches_eager
        # below, added before this test, which never compiled at all).
        # Confirms the native stale-reuse bug is not max-bound-specific
        # before trusting that safe_clamp's min-bound guard below is
        # fixing a real divergence rather than a no-op.
        torch._dynamo.reset()

        def eager_fn(x, limit):
            return torch.clamp(x, min=limit)

        compiled_fn = torch.compile(eager_fn, backend="inductor")

        with torch._dynamo.config.patch(specialize_float=False):
            x0 = torch.full((1, 2), -0.75, dtype=torch.float32)
            compiled_fn(x0, -1.0)

            x1 = torch.full((1, 4), -0.75, dtype=torch.float32, requires_grad=True)
            compiled_fn(x1, -0.5)

            x2 = torch.full((1, 4), -0.75, dtype=torch.float32, requires_grad=True)
            eager_val = eager_fn(x2.detach().clone(), -1.0).abs().sum().item()
            compiled_val = compiled_fn(x2, -1.0).abs().sum().item()

        assert abs(compiled_val - eager_val) > 1e-6, (
            "expected the native bug to also reproduce for clamp's "
            "min bound under torch.compile, using the identical "
            "invocation-order trigger as the max-bound repro; if this "
            "now fails, min may behave differently from max on this "
            "torch version -- update README's scope section rather "
            "than treating this as a hard regression"
        )


class TestSafeClampMatchesEager:
    def test_safe_clamp_matches_eager_uncompiled(self):
        x = torch.full((1, 4), 0.75, dtype=torch.float32)
        eager = torch.clamp(x, max=0.5)
        guarded = safe_clamp(x, max=0.5)
        assert torch.equal(eager, guarded)

    def test_safe_clamp_matches_eager_for_every_call_in_stale_sequence(self):
        report = diagnose()
        for case in report["cases"]:
            assert case["guard_matches_eager"] is True, (
                f"call {case['call_index']} (shape={case['shape']}, "
                f"limit={case['limit']}): guard value "
                f"{case['guard_value']} did not match eager value "
                f"{case['eager_value']}"
            )

    def test_guard_fully_correct_flag_is_true(self):
        report = diagnose()
        assert report["guard_fully_correct"] is True

    def test_safe_clamp_preserves_gradient(self):
        x = torch.full((1, 4), 0.75, dtype=torch.float32, requires_grad=True)
        x_ref = x.detach().clone().requires_grad_(True)

        eager = torch.clamp(x_ref, max=1.0)
        eager.sum().backward()

        guarded = safe_clamp(x, max=1.0)
        guarded.sum().backward()

        assert torch.allclose(x.grad, x_ref.grad)

    def test_safe_clamp_min_bound_also_matches_eager(self):
        x = torch.full((1, 4), -0.75, dtype=torch.float32)
        eager = torch.clamp(x, min=-0.5)
        guarded = safe_clamp(x, min=-0.5)
        assert torch.equal(eager, guarded)

    def test_safe_clamp_works_under_torch_compile(self):
        torch._dynamo.reset()
        compiled_guard = torch.compile(
            lambda x, limit: safe_clamp(x, max=limit), backend="inductor"
        )
        with torch._dynamo.config.patch(specialize_float=False):
            x0 = torch.full((1, 2), 0.75, dtype=torch.float32)
            compiled_guard(x0, 1.0)
            x1 = torch.full((1, 4), 0.75, dtype=torch.float32, requires_grad=True)
            compiled_guard(x1, 0.5)
            x2 = torch.full((1, 4), 0.75, dtype=torch.float32, requires_grad=True)
            guarded_val = compiled_guard(x2, 1.0).abs().sum().item()
            eager_val = torch.clamp(x2.detach().clone(), max=1.0).abs().sum().item()
        assert abs(guarded_val - eager_val) < 1e-6

    def test_safe_clamp_min_bound_works_under_torch_compile(self):
        # Companion to test_safe_clamp_works_under_torch_compile above:
        # that test only exercises the `max` bound under compile. The
        # existing test_safe_clamp_min_bound_also_matches_eager (above)
        # only exercises `min` in plain eager mode, which can never
        # demonstrate the guard fixing anything (eager has no staleness
        # to guard against). This test closes that gap by running
        # `min` through the same stale-reuse-triggering invocation
        # sequence under torch.compile that
        # test_unguarded_compiled_clamp_min_bound_also_diverges just
        # proved really does diverge natively -- confirming safe_clamp
        # fixes a REAL min-bound divergence, not an already-correct
        # no-op.
        torch._dynamo.reset()
        compiled_guard = torch.compile(
            lambda x, limit: safe_clamp(x, min=limit), backend="inductor"
        )
        with torch._dynamo.config.patch(specialize_float=False):
            x0 = torch.full((1, 2), -0.75, dtype=torch.float32)
            compiled_guard(x0, -1.0)
            x1 = torch.full((1, 4), -0.75, dtype=torch.float32, requires_grad=True)
            compiled_guard(x1, -0.5)
            x2 = torch.full((1, 4), -0.75, dtype=torch.float32, requires_grad=True)
            guarded_val = compiled_guard(x2, -1.0).abs().sum().item()
            eager_val = torch.clamp(x2.detach().clone(), min=-1.0).abs().sum().item()
        assert abs(guarded_val - eager_val) < 1e-6, (
            "safe_clamp's min-bound guard should match eager exactly "
            "under torch.compile, even though the native (unguarded) "
            "path diverges here"
        )


class TestDiagnose:
    def test_diagnose_runs_and_reports_consistent_structure(self):
        report = diagnose()
        assert len(report["cases"]) == 3
        assert isinstance(report["torch_version"], str)
        assert "issue_urls" in report
        assert len(report["issue_urls"]) == 2

    def test_diagnose_is_deterministic_across_repeated_calls(self):
        r1 = diagnose(seed=0)
        r2 = diagnose(seed=0)
        assert r1["any_stale_reuse_bug"] == r2["any_stale_reuse_bug"]
        assert r1["guard_fully_correct"] == r2["guard_fully_correct"]
