"""torch-dynamo-closure-descriptor-guard core: detect and guard a real
``torch.compile`` (Dynamo) correctness bug where a closure that
captures a Tensor method DESCRIPTOR (e.g. ``torch.Tensor.__add__`` vs
``torch.Tensor.__mul__``) as a free variable can silently reuse a
PREVIOUSLY COMPILED graph belonging to a *different* descriptor,
because Dynamo's guard system does not add a guard on the captured
descriptor's identity itself.

Reproduced from scratch on this host (torch 2.14.0, CPU, no GPU/MPS
needed -- this is a pure Dynamo guard-source/graph-cache defect, not a
kernel-codegen one; see README for the exact commands):

    def make(op):
        def f(a, b):
            return op(a, b)
        return f

    add_fn = make(torch.Tensor.__add__)
    mul_fn = make(torch.Tensor.__mul__)

    add_fn(t6, t3)                         # eager: 9.0  (6+3)
    mul_fn(t6, t3)                         # eager: 18.0 (6*3)

    torch.compile(add_fn, dynamic=True)(t6, t3)   # inductor: 9.0  (correct)
    torch.compile(mul_fn, dynamic=True)(t6, t3)   # inductor: 9.0  (WRONG! should be 18.0)

``make(op)`` returns a code object shared across both closures --
``add_fn`` and ``mul_fn`` differ only in the captured free variable
``op``. Dynamo's guard system installs guards on the TENSOR inputs
(``a``, ``b``) but never on the captured method-descriptor free
variable itself, so ``torch.compile()`` on ``mul_fn`` sees a cache hit
against ``add_fn``'s already-compiled graph (same code object, same
tensor shapes/dtypes) and silently reuses ADD's graph, returning add's
result for a multiply call. No exception, no warning, no graph break,
no non-finite marker -- just a silently wrong number for an entirely
ordinary functional-programming pattern (a factory function that binds
a different operator per closure instance).

Plain Python functions captured the same way (not Tensor method
descriptors, e.g. a user-defined ``def add(a, b): return a + b``) do
NOT reproduce this -- Dynamo traces a user function's *body* rather
than treating it as an opaque descriptor object, so the miscompare is
specific to capturing one of Tensor's C-implemented dunder method
descriptors (verified here for ``__add__``/``__mul__``; the same
underlying guard-cache-key omission plausibly extends to other
Tensor dunder descriptors captured the same way, though only these two
are exercised by this guard's regression tests).

Upstream reference: pytorch/pytorch#197811 ("[dynamo] torch.compile
silently reuses a graph when closures capture different Tensor method
descriptors"), open as of this writing, filed against a 2026-09-19
nightly build.

This module's guard function, ``safe_call``, forces the actual
operator-descriptor call outside the compiled graph via
``torch.compiler.disable`` (a deliberate, narrow Dynamo graph break at
this single call site), so the real eager dispatch -- which always
resolves the CURRENT descriptor, not a cached one -- runs regardless
of whether the surrounding function is itself under ``torch.compile``.
This is the same call-site-workaround shape already established in
this fleet (torch-compile-dynamic-clamp-guard's ``safe_clamp``,
torch-inductor-full-dtype-guard's ``safe_full``): no ATen/Dynamo patch
access needed, just a targeted graph break at the one call site where
staleness/reuse can occur.
"""
from __future__ import annotations

import dataclasses
from typing import Any, Callable, Dict, List


class TorchUnavailableError(RuntimeError):
    """Raised when torch cannot be imported. Kept as a distinct type so
    callers can distinguish "torch isn't installed" from an actual
    diagnostic failure."""


def _import_torch():
    try:
        import torch  # noqa: F401
    except Exception as exc:  # pragma: no cover - exercised only without torch
        raise TorchUnavailableError(
            "torch is required for diagnosis and guarding; install the "
            "'torch' extra."
        ) from exc
    return torch


def _make_safe_call(torch_module):
    """Build the guard function bound to a specific torch module (so
    the same core logic works against whatever torch build is actually
    installed, without importing torch at module load time)."""

    @torch_module.compiler.disable
    def _eager_apply(op, a, b):
        return op(a, b)

    def safe_call(op, a, b):
        """Drop-in guard for calling a captured Tensor method
        descriptor (or any binary callable) at a call site where the
        SAME enclosing code object may be reused across closures that
        capture DIFFERENT descriptors as a free variable under
        ``torch.compile``. Forces the actual call to run in eager mode
        via a Dynamo graph break, so the real, currently-captured
        descriptor is always dispatched -- matching eager exactly
        instead of silently reusing a differently-captured closure's
        cached graph."""
        return _eager_apply(op, a, b)

    return safe_call


def safe_call(op: Callable, a, b):
    """Module-level convenience wrapper: resolves torch on first call
    and delegates to the bound guard. See ``_make_safe_call`` for the
    full rationale."""
    torch_module = _import_torch()
    return _make_safe_call(torch_module)(op, a, b)


def _make_closure(op):
    """The exact factory-function shape from the upstream repro: a
    single code object shared across every closure this factory
    produces, differing only in the captured free variable ``op``."""

    def f(a, b):
        return op(a, b)

    return f


def _make_guarded_closure(torch_module, op):
    safe = _make_safe_call(torch_module)

    def f(a, b):
        return safe(op, a, b)

    return f


@dataclasses.dataclass
class ClosureDescriptorCase:
    op_name: str
    a: float
    b: float
    eager_result: float
    compiled_result: float
    guard_result: float
    stale_reuse_bug: bool  # compiled disagrees with eager on THIS call
    guard_matches_eager: bool


def _run_closure_descriptor_sequence(
    torch_module, a_val: float, b_val: float
) -> List[ClosureDescriptorCase]:
    """Reproduce the exact invocation-order-dependent repro from
    pytorch/pytorch#197811: compile a closure capturing
    ``Tensor.__add__`` first (establishing cached graph state for this
    factory's code object), then compile a SECOND, freshly-made
    closure from the same factory that instead captures
    ``Tensor.__mul__`` -- and check whether the second call silently
    reuses the first closure's graph.
    """
    ops = [
        ("__add__", torch_module.Tensor.__add__),
        ("__mul__", torch_module.Tensor.__mul__),
    ]

    results: List[ClosureDescriptorCase] = []
    for op_name, op in ops:
        a = torch_module.tensor(a_val)
        b = torch_module.tensor(b_val)

        eager_fn = _make_closure(op)
        eager_val = float(eager_fn(a, b).item())

        compiled_fn = torch_module.compile(_make_closure(op), dynamic=True)
        compiled_val = float(compiled_fn(a, b).item())

        guard_fn = torch_module.compile(
            _make_guarded_closure(torch_module, op), dynamic=True
        )
        guard_val = float(guard_fn(a, b).item())

        results.append(
            ClosureDescriptorCase(
                op_name=op_name,
                a=a_val,
                b=b_val,
                eager_result=eager_val,
                compiled_result=compiled_val,
                guard_result=guard_val,
                stale_reuse_bug=abs(compiled_val - eager_val) > 1e-6,
                guard_matches_eager=abs(guard_val - eager_val) <= 1e-6,
            )
        )
    return results


def diagnose(a_val: float = 6.0, b_val: float = 3.0) -> Dict[str, Any]:
    """Reproduce the eager-vs-Dynamo closure-descriptor graph-reuse
    divergence from scratch against the currently installed torch
    build, and verify ``safe_call`` matches eager under compilation.
    Never trusts a cached/prior result -- every call re-runs the
    actual repro sequence (a fresh ``torch._dynamo.reset()`` per case
    would defeat the very cache-reuse this diagnostic is checking for,
    so resets happen only between independent ``diagnose()`` calls,
    matching the upstream issue's own single-process repro).
    """
    torch_module = _import_torch()
    torch_module._dynamo.reset()

    cases = _run_closure_descriptor_sequence(torch_module, a_val, b_val)

    any_stale_reuse_bug = any(c.stale_reuse_bug for c in cases)
    guard_fully_correct = all(c.guard_matches_eager for c in cases)

    return {
        "torch_version": torch_module.__version__,
        "issue_urls": ["https://github.com/pytorch/pytorch/issues/197811"],
        "cases": [dataclasses.asdict(c) for c in cases],
        "any_stale_reuse_bug": any_stale_reuse_bug,
        "guard_fully_correct": guard_fully_correct,
    }


# ---------------------------------------------------------------------------
# Second guard family: bound-method __code__ mutation staleness
# (pytorch/pytorch#197860). Distinct root cause from the descriptor-capture
# bug above (that one is about WHICH descriptor a closure captures; this one
# is about Dynamo never guarding the __code__ object underlying an inlined
# bound method), but the same fleet-established shape: Dynamo's guard tree
# omits a check that would force recompilation, so a compiled callable
# silently keeps running stale bytecode after `SomeClass.method.__code__` is
# swapped at runtime -- no exception, no warning, no graph break.
#
# Reproduced from scratch on this host (torch 2.14.0, CPU):
#
#   def first(self, x):  return x * 2.0
#   def second(self, x): return x * 9.0
#   class Model:
#       m = first
#   model = Model()
#   compiled = torch.compile(lambda x: model.m(x))
#   compiled(x)                       # [2, 4, 6]      (correct)
#   Model.m.__code__ = second.__code__
#   model.m(x)                        # [9, 18, 27]    (eager: correct)
#   compiled(x)                       # [2, 4, 6]      (WRONG! stale)
#
# `TORCH_LOGS=guards` (per the issue's own report) shows a type guard for
# `model` and an absence guard for `model.__dict__["m"]`, but no guard on
# `Model.m.__code__` itself -- so the inlined bound-method path reuses the
# old graph after the underlying function's bytecode changes in place.
#
# Upstream reference: pytorch/pytorch#197860 ("[dynamo] torch.compile does
# not guard the code object of an inlined bound method"), open as of this
# writing, filed against a 2026-09-19 nightly build alongside #197811 and
# #197859 by the same reporter.
#
# The guard here uses the identical call-site-workaround shape as
# `safe_call` above: force the actual bound-method dispatch to happen
# outside the compiled graph via `torch.compiler.disable`, so every call
# re-resolves the CURRENT `__code__` in real eager Python instead of
# reusing whatever bytecode was inlined at trace time.


@dataclasses.dataclass
class BoundMethodMutationCase:
    before_mutation: List[float]
    eager_after_mutation: List[float]
    compiled_after_mutation: List[float]
    guard_after_mutation: List[float]
    stale_reuse_bug: bool  # compiled kept the pre-mutation result after eager changed
    guard_matches_eager: bool


def _make_bound_method_model(torch_module):
    """Build the exact Model/first/second shape from the upstream
    #197860 repro, isolated in its own factory so each diagnose() call
    gets a fresh class/instance (mutating `Model.m.__code__` is
    process-global-ish via the shared function object, so reusing a
    class across calls would contaminate later diagnose() calls)."""

    def first(self, x):
        return x * 2.0

    def second(self, x):
        return x * 9.0

    class Model:
        m = first

    model = Model()
    return model, first, second, Model


def _make_safe_bound_method_call(torch_module):
    """Build the guard function bound to a specific torch module: force
    the actual bound-method dispatch (attribute lookup AND call) to run
    in eager mode via a Dynamo graph break, so a live __code__ mutation
    on the underlying function is always observed."""

    @torch_module.compiler.disable
    def _eager_bound_call(obj, attr_name, x):
        method = getattr(obj, attr_name)
        return method(x)

    def safe_bound_method_call(obj, attr_name, x):
        return _eager_bound_call(obj, attr_name, x)

    return safe_bound_method_call


def safe_bound_method_call(obj, attr_name: str, x):
    """Module-level convenience wrapper: resolves torch on first call
    and delegates to the bound guard. Drop-in guard for calling
    ``getattr(obj, attr_name)(x)`` at a call site that may run under
    ``torch.compile`` while the underlying method's ``__code__`` can be
    replaced at runtime (e.g. hot-patching, monkeypatch-based test
    fixtures, some serialization/versioning shims)."""
    torch_module = _import_torch()
    return _make_safe_bound_method_call(torch_module)(obj, attr_name, x)


def _run_bound_method_mutation_case(torch_module) -> BoundMethodMutationCase:
    model, first, second, Model = _make_bound_method_model(torch_module)
    x = torch_module.arange(1.0, 4.0)

    compiled_call = torch_module.compile(lambda inp: model.m(inp), dynamic=True)
    before = compiled_call(x).tolist()

    safe = _make_safe_bound_method_call(torch_module)
    guarded_call = torch_module.compile(lambda inp: safe(model, "m", inp), dynamic=True)
    _ = guarded_call(x)  # populate the guarded callable's cache before mutation too

    Model.m.__code__ = second.__code__

    eager_after = model.m(x).tolist()
    compiled_after = compiled_call(x).tolist()
    guard_after = guarded_call(x).tolist()

    stale_reuse_bug = (compiled_after == before) and (eager_after != before)
    guard_matches_eager = all(
        abs(a - b) < 1e-6 for a, b in zip(guard_after, eager_after)
    )

    return BoundMethodMutationCase(
        before_mutation=before,
        eager_after_mutation=eager_after,
        compiled_after_mutation=compiled_after,
        guard_after_mutation=guard_after,
        stale_reuse_bug=stale_reuse_bug,
        guard_matches_eager=guard_matches_eager,
    )


def diagnose_bound_method_guard() -> Dict[str, Any]:
    """Reproduce the eager-vs-Dynamo bound-method __code__ mutation
    divergence (pytorch/pytorch#197860) from scratch against the
    currently installed torch build, and verify
    ``safe_bound_method_call`` matches eager under compilation even
    after the underlying method's bytecode is replaced at runtime."""
    torch_module = _import_torch()
    torch_module._dynamo.reset()

    case = _run_bound_method_mutation_case(torch_module)

    return {
        "torch_version": torch_module.__version__,
        "issue_urls": ["https://github.com/pytorch/pytorch/issues/197860"],
        "case": dataclasses.asdict(case),
        "stale_reuse_bug": case.stale_reuse_bug,
        "guard_matches_eager": case.guard_matches_eager,
    }


# ---------------------------------------------------------------------------
# Third guard family: nn.Module instance __dict__ attribute-shadowing
# staleness (pytorch/pytorch#197859). Same sibling-issue batch as #197860
# (same reporter, same nightly build, same "module: correctness (silent)" +
# "module: dynamo"/"module: guards" labels) and the identical underlying
# defect SHAPE as both guards above: Dynamo's guard tree omits one specific
# absence/identity check, so a compiled callable silently reuses a stale
# graph after a runtime mutation that eager handles correctly.
#
# Reproduced from scratch on this host (torch 2.14.0, CPU):
#
#   class Wrapper(nn.Module):
#       def __init__(self):
#           super().__init__()
#           self.sub = nn.Linear(4, 4, bias=False)   # registered submodule
#       def forward(self, x):
#           return self.sub(x)
#   model = Wrapper(); compiled = torch.compile(model)
#   compiled(x)                                       # correct
#   object.__setattr__(model, "sub", replacement)      # shadows model._modules["sub"]
#                                                       # via instance __dict__
#   model(x)                                           # eager: correct (new weights)
#   compiled(x)                                        # WRONG! stale (old weights)
#
# Per the issue's own report: Python attribute lookup finds the new
# `model.__dict__["sub"]` entry before falling back to `nn.Module.__getattr__`
# and `_modules`. Dynamo guards `model._modules["sub"]` but never adds an
# absence guard for `"sub"` staying out of `model.__dict__` -- so all
# existing guards still pass and the stale graph is reused.
#
# Upstream reference: pytorch/pytorch#197859 ("[dynamo] torch.compile
# misses nn.Module __dict__ shadowing and returns stale results"), open as
# of this writing.
#
# The guard here uses the same call-site-workaround shape as the two guards
# above: force submodule attribute resolution to happen outside the
# compiled graph via `torch.compiler.disable`, so a live instance-__dict__
# shadow is always observed instead of the compiled graph's stale
# `_modules` reference.


@dataclasses.dataclass
class ModuleDictShadowCase:
    before_shadow: List[float]
    eager_after_shadow: List[float]
    compiled_after_shadow: List[float]
    guard_after_shadow: List[float]
    stale_reuse_bug: bool  # compiled kept the pre-shadow result after eager changed
    guard_matches_eager: bool


def _make_module_shadow_model(torch_module):
    """Build the exact Wrapper/nn.Linear shape from the upstream
    #197859 repro, isolated in its own factory so each diagnose() call
    gets a fresh module instance."""
    import torch.nn as nn  # local import: only needed for this guard family

    class Wrapper(nn.Module):
        def __init__(self):
            super().__init__()
            self.sub = nn.Linear(4, 4, bias=False)
            with torch_module.no_grad():
                self.sub.weight.copy_(torch_module.eye(4))

        def forward(self, x):
            return self.sub(x)

    model = Wrapper()
    replacement = nn.Linear(4, 4, bias=False)
    with torch_module.no_grad():
        replacement.weight.fill_(5.0)
    return model, replacement


def _make_safe_module_forward(torch_module):
    """Build the guard function bound to a specific torch module: force
    the actual forward-through-submodule dispatch to run in eager mode
    via a Dynamo graph break, so a live instance-__dict__ submodule
    shadow is always observed."""

    @torch_module.compiler.disable
    def _eager_forward(module, x):
        return module(x)

    def safe_module_forward(module, x):
        return _eager_forward(module, x)

    return safe_module_forward


def safe_module_forward(module, x):
    """Module-level convenience wrapper: resolves torch on first call
    and delegates to the bound guard. Drop-in guard for calling
    ``module(x)`` at a call site that may run under ``torch.compile``
    while a submodule/parameter/buffer attribute can be shadowed at
    runtime via ``object.__setattr__`` (bypassing ``nn.Module``'s own
    ``__setattr__`` registration bookkeeping -- e.g. some
    serialization, hot-swap, or test-mocking code paths)."""
    torch_module = _import_torch()
    return _make_safe_module_forward(torch_module)(module, x)


def _run_module_shadow_case(torch_module) -> ModuleDictShadowCase:
    model, replacement = _make_module_shadow_model(torch_module)
    x = torch_module.arange(4.0)

    compiled_call = torch_module.compile(model, dynamic=True)
    before = compiled_call(x).tolist()

    safe = _make_safe_module_forward(torch_module)
    guarded_call = torch_module.compile(lambda inp: safe(model, inp), dynamic=True)
    _ = guarded_call(x)  # populate the guarded callable's cache before shadowing too

    object.__setattr__(model, "sub", replacement)

    eager_after = model(x).tolist()
    compiled_after = compiled_call(x).tolist()
    guard_after = guarded_call(x).tolist()

    stale_reuse_bug = (compiled_after == before) and (eager_after != before)
    guard_matches_eager = all(
        abs(a - b) < 1e-6 for a, b in zip(guard_after, eager_after)
    )

    return ModuleDictShadowCase(
        before_shadow=before,
        eager_after_shadow=eager_after,
        compiled_after_shadow=compiled_after,
        guard_after_shadow=guard_after,
        stale_reuse_bug=stale_reuse_bug,
        guard_matches_eager=guard_matches_eager,
    )


def diagnose_module_shadow_guard() -> Dict[str, Any]:
    """Reproduce the eager-vs-Dynamo nn.Module instance-__dict__
    submodule-shadowing divergence (pytorch/pytorch#197859) from
    scratch against the currently installed torch build, and verify
    ``safe_module_forward`` matches eager under compilation even after
    a submodule attribute is shadowed via ``object.__setattr__``."""
    torch_module = _import_torch()
    torch_module._dynamo.reset()

    case = _run_module_shadow_case(torch_module)

    return {
        "torch_version": torch_module.__version__,
        "issue_urls": ["https://github.com/pytorch/pytorch/issues/197859"],
        "case": dataclasses.asdict(case),
        "stale_reuse_bug": case.stale_reuse_bug,
        "guard_matches_eager": case.guard_matches_eager,
    }
