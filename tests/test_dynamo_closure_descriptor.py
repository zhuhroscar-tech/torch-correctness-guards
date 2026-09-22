"""Regression tests for torch-dynamo-closure-descriptor-guard.

These prove:
  1. The bug is real and reproducible from scratch on this host's
     installed torch build: torch.compile silently reuses a compiled
     graph for a closure that captures a DIFFERENT Tensor method
     descriptor than the one that first populated the cache for that
     shared code object (pytorch/pytorch#197811).
  2. safe_call() is an independently verified fix: it matches eager's
     own output (the independent oracle -- eager always resolves the
     currently-captured descriptor correctly) for every case, whether
     called directly or under torch.compile.
  3. A bug-injection test proves the stale-reuse detection is
     non-tautological: an UNGUARDED compiled closure sequence built
     the same way really does diverge from eager, confirming the test
     harness would catch a regression if the guard were removed.
  4. The bug is specific to capturing a Tensor method DESCRIPTOR, not
     a general closure/free-variable defect: a plain Python function
     captured the identical way does NOT reproduce it.
"""
from __future__ import annotations

import pytest

torch = pytest.importorskip("torch")

from torch_correctness_guards.guards.dynamo_closure_descriptor import (
    diagnose,
    diagnose_bound_method_guard,
    diagnose_module_shadow_guard,
    safe_call,
    safe_bound_method_call,
    safe_module_forward,
    _make_closure,
    _make_guarded_closure,
    _make_bound_method_model,
    _make_module_shadow_model,
)


class TestNativeBugReproduction:
    def test_stale_reuse_reproduces_on_this_host(self):
        # Not asserted unconditionally true forever: if a future torch
        # release fixes Dynamo's missing guard on captured method
        # descriptors, this docstring is the record that the bug
        # existed at the version noted in the ledger/README. On torch
        # 2.14.0 (this host) it reproduces reliably.
        report = diagnose()
        assert report["any_stale_reuse_bug"] is True, (
            f"expected a closure-descriptor graph-reuse divergence on "
            f"torch {report['torch_version']}; if this now fails, the "
            "bug may be fixed upstream (pytorch/pytorch#197811) -- "
            "update the README/ledger accordingly rather than treating "
            "this as a regression"
        )

    def test_second_closure_is_the_one_that_diverges(self):
        # The first closure compiled from a fresh factory always
        # populates the cache correctly (nothing to reuse yet); only
        # the SECOND closure sharing the same code object but a
        # different captured descriptor exhibits the reuse bug.
        report = diagnose()
        cases = report["cases"]
        assert len(cases) == 2
        assert cases[0]["op_name"] == "__add__"
        assert cases[0]["stale_reuse_bug"] is False
        assert cases[1]["op_name"] == "__mul__"
        assert cases[1]["stale_reuse_bug"] is True, (
            "expected the second closure (capturing __mul__, sharing "
            "the first closure's code object) to reuse the first "
            "closure's (__add__'s) compiled graph"
        )

    def test_diverging_call_compiled_value_matches_first_ops_result(self):
        # Directly demonstrate the "stale reuse" characterization: the
        # second (mul) closure's compiled result numerically matches
        # what the FIRST closure's op (add) would have produced on the
        # same inputs, not an arbitrary wrong number.
        report = diagnose(a_val=6.0, b_val=3.0)
        add_case, mul_case = report["cases"]
        assert add_case["op_name"] == "__add__"
        assert mul_case["op_name"] == "__mul__"
        expected_add_result = 6.0 + 3.0
        assert abs(mul_case["compiled_result"] - expected_add_result) < 1e-6, (
            "expected the mul closure's compiled (buggy) value to "
            "numerically match the add op's result on the same "
            "inputs, confirming genuine graph reuse rather than an "
            "unrelated wrong answer"
        )


class TestBugInjectionNonTautological:
    """Prove the stale-reuse assertions above are not tautological by
    running the UNGUARDED compiled sequence directly (bypassing
    safe_call entirely) and confirming it really does diverge from
    eager -- i.e. the test harness would catch a regression if
    safe_call stopped being applied."""

    def test_unguarded_compiled_closures_diverge(self):
        torch._dynamo.reset()

        add_fn = _make_closure(torch.Tensor.__add__)
        mul_fn = _make_closure(torch.Tensor.__mul__)

        a = torch.tensor(6.0)
        b = torch.tensor(3.0)

        eager_mul = float(mul_fn(a, b).item())

        compiled_add = torch.compile(add_fn, dynamic=True)
        compiled_add(a, b)  # populate the cache with add's graph

        compiled_mul = torch.compile(mul_fn, dynamic=True)
        compiled_mul_val = float(compiled_mul(a, b).item())

        assert abs(compiled_mul_val - eager_mul) > 1e-6, (
            "sanity check: the unguarded compiled mul closure really "
            "must diverge from eager on this host for this test "
            "suite's positive assertions to be meaningful evidence of "
            "a real bug, not a tautology"
        )

    def test_plain_python_function_closure_does_not_reproduce_the_bug(self):
        # Scope-narrowing check: this bug is specific to capturing a
        # Tensor method DESCRIPTOR (a C-implemented dunder), not a
        # general closure/free-variable caching defect. A user-defined
        # Python function captured the identical way should NOT
        # reproduce it, because Dynamo traces the function's body
        # rather than treating it as an opaque cached descriptor.
        torch._dynamo.reset()

        def add(a, b):
            return a + b

        def mul(a, b):
            return a * b

        def make(op):
            def f(a, b):
                return op(a, b)

            return f

        a = torch.tensor(6.0)
        b = torch.tensor(3.0)

        eager_mul = float(make(mul)(a, b).item())

        compiled_add = torch.compile(make(add), dynamic=True)
        compiled_add(a, b)

        compiled_mul = torch.compile(make(mul), dynamic=True)
        compiled_mul_val = float(compiled_mul(a, b).item())

        assert abs(compiled_mul_val - eager_mul) < 1e-6, (
            "expected a plain Python function closure to NOT exhibit "
            "the graph-reuse bug (Dynamo traces the function body, "
            "not an opaque descriptor) -- if this now fails, the "
            "bug's scope has widened and the README's scope section "
            "should be updated"
        )


class TestSafeCallMatchesEager:
    def test_safe_call_matches_eager_uncompiled(self):
        a = torch.tensor(6.0)
        b = torch.tensor(3.0)
        eager = torch.Tensor.__mul__(a, b)
        guarded = safe_call(torch.Tensor.__mul__, a, b)
        assert torch.equal(eager, guarded)

    def test_safe_call_matches_eager_for_every_case(self):
        report = diagnose()
        for case in report["cases"]:
            assert case["guard_matches_eager"] is True, (
                f"op {case['op_name']}: guard value {case['guard_result']} "
                f"did not match eager value {case['eager_result']}"
            )

    def test_guard_fully_correct_flag_is_true(self):
        report = diagnose()
        assert report["guard_fully_correct"] is True

    def test_safe_call_works_under_torch_compile_and_fixes_reuse(self):
        torch._dynamo.reset()

        guarded_add = torch.compile(
            _make_guarded_closure(torch, torch.Tensor.__add__), dynamic=True
        )
        a = torch.tensor(6.0)
        b = torch.tensor(3.0)
        guarded_add(a, b)  # populate cache with a guarded add call

        guarded_mul = torch.compile(
            _make_guarded_closure(torch, torch.Tensor.__mul__), dynamic=True
        )
        guarded_val = float(guarded_mul(a, b).item())
        eager_val = float(torch.Tensor.__mul__(a, b).item())
        assert abs(guarded_val - eager_val) < 1e-6, (
            "safe_call's guard should match eager exactly under "
            "torch.compile, even though the native (unguarded) path "
            "reuses a stale graph here"
        )

    def test_safe_call_preserves_gradient(self):
        a = torch.tensor(6.0, requires_grad=True)
        b = torch.tensor(3.0, requires_grad=True)
        a_ref = a.detach().clone().requires_grad_(True)
        b_ref = b.detach().clone().requires_grad_(True)

        eager = torch.Tensor.__mul__(a_ref, b_ref)
        eager.backward()

        guarded = safe_call(torch.Tensor.__mul__, a, b)
        guarded.backward()

        assert torch.allclose(a.grad, a_ref.grad)
        assert torch.allclose(b.grad, b_ref.grad)


class TestDiagnose:
    def test_diagnose_runs_and_reports_consistent_structure(self):
        report = diagnose()
        assert len(report["cases"]) == 2
        assert isinstance(report["torch_version"], str)
        assert report["issue_urls"] == ["https://github.com/pytorch/pytorch/issues/197811"]

    def test_diagnose_is_deterministic_across_repeated_calls(self):
        r1 = diagnose(a_val=6.0, b_val=3.0)
        r2 = diagnose(a_val=6.0, b_val=3.0)
        assert r1["any_stale_reuse_bug"] == r2["any_stale_reuse_bug"]
        assert r1["guard_fully_correct"] == r2["guard_fully_correct"]

    def test_diagnose_with_different_operands(self):
        report = diagnose(a_val=2.0, b_val=5.0)
        add_case, mul_case = report["cases"]
        assert abs(add_case["eager_result"] - 7.0) < 1e-6
        assert abs(mul_case["eager_result"] - 10.0) < 1e-6
        assert report["guard_fully_correct"] is True


class TestBoundMethodCodeMutationGuard:
    """Regression tests for the second guard family: pytorch/pytorch#197860
    (Dynamo does not guard the __code__ object of an inlined bound method,
    so a compiled callable silently keeps running stale bytecode after
    ``SomeClass.method.__code__`` is replaced at runtime)."""

    def test_stale_reuse_reproduces_on_this_host(self):
        report = diagnose_bound_method_guard()
        assert report["stale_reuse_bug"] is True, (
            f"expected a bound-method __code__ mutation staleness bug on "
            f"torch {report['torch_version']}; if this now fails, the bug "
            "may be fixed upstream (pytorch/pytorch#197860) -- update the "
            "README/ledger accordingly rather than treating this as a "
            "regression"
        )

    def test_guard_matches_eager_after_mutation(self):
        report = diagnose_bound_method_guard()
        assert report["guard_matches_eager"] is True
        case = report["case"]
        assert case["guard_after_mutation"] == case["eager_after_mutation"]

    def test_unguarded_compiled_call_diverges_from_eager(self):
        # Bug-injection / non-tautological check: prove the compiled call
        # (bypassing safe_bound_method_call) really does diverge from eager
        # after the mutation, so the positive assertions above are
        # meaningful evidence of a real bug, not a tautology.
        report = diagnose_bound_method_guard()
        case = report["case"]
        assert case["compiled_after_mutation"] != case["eager_after_mutation"]
        assert case["compiled_after_mutation"] == case["before_mutation"]

    def test_safe_bound_method_call_matches_eager_uncompiled(self):
        model, first, second, Model = _make_bound_method_model(torch)
        x = torch.arange(1.0, 4.0)
        eager = model.m(x)
        guarded = safe_bound_method_call(model, "m", x)
        assert torch.equal(eager, guarded)

    def test_diagnose_bound_method_guard_structure(self):
        report = diagnose_bound_method_guard()
        assert isinstance(report["torch_version"], str)
        assert report["issue_urls"] == ["https://github.com/pytorch/pytorch/issues/197860"]
        for key in (
            "before_mutation",
            "eager_after_mutation",
            "compiled_after_mutation",
            "guard_after_mutation",
        ):
            assert len(report["case"][key]) == 3


class TestModuleDictShadowGuard:
    """Regression tests for the third guard family: pytorch/pytorch#197859
    (Dynamo does not guard nn.Module instance-__dict__ attribute shadowing
    of a registered submodule, so a compiled callable silently keeps using
    the old submodule after ``object.__setattr__`` swaps it in)."""

    def test_stale_reuse_reproduces_on_this_host(self):
        report = diagnose_module_shadow_guard()
        assert report["stale_reuse_bug"] is True, (
            f"expected an nn.Module __dict__-shadowing staleness bug on "
            f"torch {report['torch_version']}; if this now fails, the bug "
            "may be fixed upstream (pytorch/pytorch#197859) -- update the "
            "README/ledger accordingly rather than treating this as a "
            "regression"
        )

    def test_guard_matches_eager_after_shadow(self):
        report = diagnose_module_shadow_guard()
        assert report["guard_matches_eager"] is True
        case = report["case"]
        assert case["guard_after_shadow"] == case["eager_after_shadow"]

    def test_unguarded_compiled_call_diverges_from_eager(self):
        # Bug-injection / non-tautological check: prove the compiled call
        # (bypassing safe_module_forward) really does diverge from eager
        # after the shadow, so the positive assertions above are meaningful
        # evidence of a real bug, not a tautology.
        report = diagnose_module_shadow_guard()
        case = report["case"]
        assert case["compiled_after_shadow"] != case["eager_after_shadow"]
        assert case["compiled_after_shadow"] == case["before_shadow"]

    def test_safe_module_forward_matches_eager_uncompiled(self):
        model, replacement = _make_module_shadow_model(torch)
        x = torch.arange(4.0)
        eager = model(x)
        guarded = safe_module_forward(model, x)
        assert torch.equal(eager, guarded)

    def test_diagnose_module_shadow_guard_structure(self):
        report = diagnose_module_shadow_guard()
        assert isinstance(report["torch_version"], str)
        assert report["issue_urls"] == ["https://github.com/pytorch/pytorch/issues/197859"]
        for key in (
            "before_shadow",
            "eager_after_shadow",
            "compiled_after_shadow",
            "guard_after_shadow",
        ):
            assert len(report["case"][key]) == 4
