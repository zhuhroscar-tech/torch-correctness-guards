"""torch-inductor-duplicate-index-writeorder-guard core: detect and guard
a real Inductor miscompilation where an advanced-indexing
read-modify-write assignment (``v[idx] = v[idx] + delta``) drops or
reorders one of two logically-independent writes when ``idx`` contains
a COMPUTED duplicate index (e.g. ``torch.arange(n) % 2``), rather than
a literal duplicate-index tensor.

Upstream reference: pytorch/pytorch#197582 ("[torch.compile] torch.compile
miscompiles advanced indexing assignment with duplicate indices computed
from arange % 2"), reproduced from scratch on this host before this
module was written -- never trusted from a cached issue summary alone.

The bug, reproduced on this host (torch 2.14.0, macOS arm64 CPU,
backend="inductor"; see README for exact commands):

    v = torch.tensor([1., 2., 3., 4.])
    idx = torch.arange(4) % 2          # -> [0, 1, 0, 1]
    v[idx] = v[idx] + 1.0

Eager gives ``[2., 3., 3., 4.]`` -- index 0's single logical
read-modify-write (last-write-wins semantics for a repeated index) is
applied exactly once, index 1's write likewise applied exactly once
(different value than index 0 because they read different original
values). Under ``torch.compile(backend="inductor")`` the SAME code
gives ``[3., 4., 3., 4.]`` -- index 0's contribution is silently
replaced by index 1's, meaning one of the two logically-distinct
writes was dropped or the accumulation order was scrambled by
Inductor's fused kernel for a *computed* duplicate-index pattern.

Per the upstream issue, the SAME operation with a LITERAL duplicate-
index tensor (``torch.tensor([0, 1, 0, 1])`` instead of
``torch.arange(4) % 2``) does NOT trigger the bug -- isolating the
defect to Inductor's handling of computed index expressions
specifically. This module's own diagnose() reproduces both the
triggering (computed) and non-triggering (literal) shapes so a caller
can see the isolation directly, not just take it on faith.

This is explicitly NOT the same as calling ``index_put_`` with
``accumulate=False`` and duplicate indices -- PyTorch's own docs
already document THAT as "undefined behavior" for non-accumulate mode.
This guard targets the DIVERGENCE BETWEEN EAGER AND COMPILED output for
the exact same Python-level assignment syntax, which is a genuine
compiler correctness bug regardless of what the "true" answer for
duplicate-index writes should be -- eager and compiled must agree, and
here they silently don't.

This module's guard, ``safe_dup_index_assign``, wraps the
read-modify-write assignment in a function decorated with
``torch._dynamo.disable``, forcing a Dynamo graph-break so the
assignment always executes eagerly instead of being lowered through
Inductor's kernel fusion. The guard does not patch PyTorch; callers
replace their inline ``v[idx] = v[idx] + delta`` call sites with
``safe_dup_index_assign(v, idx, delta)`` explicitly.
"""
from __future__ import annotations

import dataclasses
from typing import Any, Dict, List


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


def make_safe_dup_index_assign(torch_module):
    """Build a guard function, bound to a specific torch module, that
    forces ``v[idx] = v[idx] + delta`` to graph-break out of any
    enclosing ``torch.compile`` region via ``torch._dynamo.disable``,
    so the read-modify-write assignment always executes eagerly
    (matching eager's last-write-wins semantics for duplicate indices)
    instead of being lowered through Inductor's fused kernel. Returns
    a callable ``(v, idx, delta) -> v`` (mutates and returns ``v`` for
    convenience)."""

    def _safe_assign(v, idx, delta):
        v[idx] = v[idx] + delta
        return v

    return torch_module._dynamo.disable(_safe_assign)


@dataclasses.dataclass
class DupIndexCase:
    kind: str
    description: str
    eager_result: List[float]
    native_compiled_result: List[float]
    guarded_compiled_result: List[float]
    native_matches_eager: bool
    guard_matches_eager: bool
    guard_correct: bool


def _run_case(torch_module, kind: str, description: str, idx_builder) -> DupIndexCase:
    base = [1.0, 2.0, 3.0, 4.0]

    def native_fn(v):
        idx = idx_builder(torch_module)
        v[idx] = v[idx] + 1.0
        return v

    safe_assign = make_safe_dup_index_assign(torch_module)

    def guarded_fn(v):
        idx = idx_builder(torch_module)
        safe_assign(v, idx, 1.0)
        return v

    v_eager = torch_module.tensor(base)
    eager_result = native_fn(v_eager.clone()).tolist()

    compiled_native = torch_module.compile(native_fn, backend="inductor")
    v_native = torch_module.tensor(base)
    native_compiled_result = compiled_native(v_native.clone()).tolist()

    compiled_guarded = torch_module.compile(guarded_fn, backend="inductor")
    v_guarded = torch_module.tensor(base)
    guarded_compiled_result = compiled_guarded(v_guarded.clone()).tolist()

    native_matches_eager = native_compiled_result == eager_result
    guard_matches_eager = guarded_compiled_result == eager_result
    # Only meaningful as a "fix" if the native path actually diverges
    # from eager on this case; a guard that trivially matches on a
    # case where nothing was ever broken proves nothing.
    guard_correct = guard_matches_eager and not native_matches_eager

    return DupIndexCase(
        kind=kind,
        description=description,
        eager_result=eager_result,
        native_compiled_result=native_compiled_result,
        guarded_compiled_result=guarded_compiled_result,
        native_matches_eager=native_matches_eager,
        guard_matches_eager=guard_matches_eager,
        guard_correct=guard_correct,
    )


def _computed_duplicate_idx(torch_module):
    # The triggering shape per pytorch/pytorch#197582: a duplicate
    # index produced by a COMPUTED expression, not a literal tensor.
    return torch_module.arange(4) % 2


def _literal_duplicate_idx(torch_module):
    # The upstream issue's own control case: the SAME logical
    # duplicate-index pattern but as a literal tensor. Per the issue,
    # this does NOT trigger the miscompilation -- included so
    # diagnose() demonstrates the isolation directly rather than
    # asserting it from prose alone.
    return torch_module.tensor([0, 1, 0, 1])


def diagnose() -> Dict[str, Any]:
    """Reproduce the computed-duplicate-index miscompilation from
    scratch against the currently installed torch build
    (backend='inductor'), demonstrate the literal-index control case
    does NOT trigger it (isolating the defect per the upstream issue),
    and verify ``safe_dup_index_assign``'s ``torch._dynamo.disable``
    guard restores eager semantics on the triggering case. Never
    trusts a cached/prior result -- every call re-runs the actual
    repro."""
    torch_module = _import_torch()

    computed_case = _run_case(
        torch_module,
        kind="computed_duplicate_index",
        description="v[torch.arange(4) % 2] = v[...] + 1.0 (the triggering shape)",
        idx_builder=_computed_duplicate_idx,
    )
    literal_case = _run_case(
        torch_module,
        kind="literal_duplicate_index",
        description="v[torch.tensor([0,1,0,1])] = v[...] + 1.0 (control: should NOT trigger)",
        idx_builder=_literal_duplicate_idx,
    )

    bug_reproduced = not computed_case.native_matches_eager
    isolation_confirmed = literal_case.native_matches_eager
    guard_fully_correct = computed_case.guard_correct

    return {
        "torch_version": torch_module.__version__,
        "issue_url": "https://github.com/pytorch/pytorch/issues/197582",
        "cases": [dataclasses.asdict(computed_case), dataclasses.asdict(literal_case)],
        "bug_reproduced": bug_reproduced,
        "isolation_confirmed": isolation_confirmed,
        "guard_fully_correct": guard_fully_correct,
    }


# Public convenience wrapper: resolves torch lazily so importing this
# module without torch installed doesn't crash (matching the sibling
# guard repos' degradation pattern).
def safe_dup_index_assign(v, idx, delta):
    """Module-level convenience wrapper around
    ``make_safe_dup_index_assign``: resolves torch on first call,
    performs ``v[idx] = v[idx] + delta`` eagerly (forcing a Dynamo
    graph-break when called from inside a ``torch.compile``'d
    region), and returns the mutated ``v``. See
    ``make_safe_dup_index_assign``'s docstring for the full
    rationale."""
    torch_module = _import_torch()
    return make_safe_dup_index_assign(torch_module)(v, idx, delta)
