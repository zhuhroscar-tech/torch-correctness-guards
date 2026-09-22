"""torch-inductor-scatter-copyback-alias-guard core: guards a real
torch.compile (Inductor) correctness bug CLASS where the compiled
function's RETURN-VALUE ALIASING contract silently changes compared
to eager mode, even though the returned VALUES are correct. Two
independently-filed, independently-reproduced upstream root causes
share this exact defect shape and are both closed by the SAME generic
guard below (no per-root-cause guard code needed):

1. Direct generalized-scatter copy-back -- pytorch/pytorch#195451
   ("[Inductor] Direct generalized_scatter copy-back changes returned
   output aliasing"). Root cause per the issue's own diagnosis and a
   maintainer-confirmed comment: functionalization lowers a pattern
   like::

       updated = torch.slice_scatter(x, src, dim, start, end)
       x.copy_(updated)
       return updated

   into ``generalized_scatter(x, ...) -> copy_(x, scatter) -> return
   scatter``. Inductor's ``should_reinplace_scatter()`` treats the
   direct copy-back as profitable and reinplaces the scatter onto
   ``x``'s own buffer -- valid ONLY when the functional scatter result
   is dead after the copy-back. Here it is returned, so reinplacing
   silently turns a non-aliasing functional result into an alias of
   the input. Upstream issue OPEN, proposed fix PR #195484 OPEN/
   UNMERGED as of this repo's creation (2026-09-20) -- confirmed via
   `gh api`, not cached.

2. No-op elimination of a neutral-value op -- pytorch/pytorch#197893
   ("Inductor's no-op removal makes x/1.0, buf*momentum(=1.0), and
   x+0 return the input object itself, unlike eager/aot_eager").
   Root cause per the issue's own diagnosis: Inductor's
   ``joint_graph.remove_no_ops`` replaces a graph node with its
   operand directly whenever the op is algebraically a no-op for the
   given constant (dividing by 1.0, multiplying by 1.0, adding 0),
   without the "don't introduce new aliasing between inputs and
   outputs" check that the separate ``post_grad.remove_noop_ops`` pass
   already has. A caller whose code is correct for every OTHER
   constant value (e.g. ``temperature=0.9``) silently gets a live
   alias of its own input for the one value that makes the op a
   no-op (``temperature=1.0``), and a subsequent in-place update of
   the "output" corrupts the input with no error. Independently
   reproduced on this host (torch 2.14.0 CPU): eager and
   ``aot_eager`` both return a fresh tensor (``out is x`` is False);
   ``backend="inductor"`` returns the SAME object as the input
   (``out.data_ptr() == x.data_ptr()`` is True), including when the
   input ``requires_grad`` (a different Tensor wrapper aliasing the
   same storage). Upstream issue OPEN as of this repo's last live
   check -- confirmed via `gh api`, not cached.

Both defects are the SAME underlying contract violation (a compiled
function's output silently shares storage with an input when eager
never would, for a specific "profitable rewrite" case Inductor's own
aliasing-safety check doesn't cover) -- not two unrelated bugs. This
tool guards both with one shared aliasing-detach wrapper rather than
shipping a near-duplicate repo per root cause, matching the pattern
already established for torch-multioutput-alias-guard (multi-output
out= aliasing) and torch-inductor-duplicate-index-writeorder-guard
(duplicate-index write order) -- each a distinct code path guarded
under one existing tool rather than forking a new one.

``safe_compiled_scatter_returning(fn)`` wraps a compiled callable and,
immediately after each call, DETACHES any returned tensor from an
accidental alias by cloning it whenever it is found to alias one of
the call's own tensor arguments -- restoring eager's non-aliasing
contract without requiring the caller to know which root cause
applies, or to remember an extra `.clone()` (which Inductor's own
lowering can eliminate INSIDE the compiled region, so the clone must
happen outside it, at the wrapper boundary, exactly as this function
already does for root cause 1).
"""
from __future__ import annotations

import dataclasses
import functools
from typing import Any, Callable, Dict, List, Sequence, Tuple


class TorchUnavailableError(RuntimeError):
    """Raised when torch cannot be imported."""


def _import_torch():
    try:
        import torch  # noqa: F401
    except Exception as exc:  # pragma: no cover - exercised only without torch
        raise TorchUnavailableError(
            "torch is required for diagnosis and guarding; install the "
            "'torch' extra."
        ) from exc
    return torch


def _tensor_args(args: Sequence[Any], kwargs: Dict[str, Any], torch_module) -> List[Any]:
    """Collect every torch.Tensor found among positional/keyword call
    arguments (shallow only -- matches the shape of the upstream bug's
    own minimal reproducer, which passes tensors directly, not nested
    inside containers)."""
    out = []
    for a in list(args) + list(kwargs.values()):
        if isinstance(a, torch_module.Tensor):
            out.append(a)
    return out


def safe_compiled_scatter_returning(compiled_fn: Callable) -> Callable:
    """Wrap a ``torch.compile``-produced callable that may exhibit the
    #195451 direct-scatter-copyback aliasing bug: any tensor in the
    return value that aliases one of the CALL's own input tensor
    arguments is replaced with an independent clone, restoring eager's
    non-aliasing contract. When the compiled function's output does
    not alias any input (the correct/unaffected case, or already-fixed
    upstream builds), this is a no-op passthrough -- no clone is made
    and the original tensor object is returned unchanged, so this
    wrapper never adds overhead beyond one ``data_ptr()`` comparison
    per output tensor once the bug is fixed upstream.
    """
    torch_module = _import_torch()

    @functools.wraps(compiled_fn)
    def wrapper(*args, **kwargs):
        inputs = _tensor_args(args, kwargs, torch_module)
        input_ptrs = {t.data_ptr() for t in inputs}
        result = compiled_fn(*args, **kwargs)

        def _fix(value):
            if isinstance(value, torch_module.Tensor) and value.data_ptr() in input_ptrs:
                return value.clone()
            return value

        if isinstance(result, tuple):
            return tuple(_fix(v) for v in result)
        if isinstance(result, list):
            return [_fix(v) for v in result]
        return _fix(result)

    return wrapper


@dataclasses.dataclass
class ScatterCopybackCase:
    x0: List[float]
    src: List[float]
    eager_aliases_input: bool
    compiled_aliases_input: bool
    compiled_input_corrupted_after_output_mutation: bool
    guarded_aliases_input: bool
    guarded_input_corrupted_after_output_mutation: bool
    guarded_values_match_eager: bool


def _scatter_copyback_fn(torch_module):
    def fn(x, src):
        updated = torch_module.slice_scatter(x, src, 0, 0, 1)
        x.copy_(updated)
        return updated

    return fn


def _run_case(torch_module, x0: Sequence[float], src: Sequence[float]) -> ScatterCopybackCase:
    fn = _scatter_copyback_fn(torch_module)

    def run(callable_, xs, ss):
        x = torch_module.tensor(list(xs), dtype=torch_module.float64)
        out = callable_(x, torch_module.tensor(list(ss), dtype=torch_module.float64))
        return x, out

    # Eager reference: the correct, non-aliasing contract.
    eager_x, eager_out = run(fn, x0, src)
    eager_aliases_input = eager_x.data_ptr() == eager_out.data_ptr()
    eager_out_before_mutation = eager_out.clone()
    eager_out.add_(100.0)
    eager_values_after_mutation = eager_x.clone()

    # Native compiled: reproduce the #195451 bug.
    compiled_fn = torch_module.compile(fn, fullgraph=True)
    compiled_x, compiled_out = run(compiled_fn, x0, src)
    compiled_aliases_input = compiled_x.data_ptr() == compiled_out.data_ptr()
    compiled_x_before_mutation = compiled_x.clone()
    compiled_out.add_(100.0)
    compiled_input_corrupted = not torch_module.equal(compiled_x, compiled_x_before_mutation)

    # Guarded compiled: the fix under test.
    guarded_fn = safe_compiled_scatter_returning(torch_module.compile(fn, fullgraph=True))
    guarded_x, guarded_out = run(guarded_fn, x0, src)
    guarded_aliases_input = guarded_x.data_ptr() == guarded_out.data_ptr()
    guarded_x_before_mutation = guarded_x.clone()
    guarded_out.add_(100.0)
    guarded_input_corrupted = not torch_module.equal(guarded_x, guarded_x_before_mutation)
    guarded_values_match_eager = torch_module.equal(
        guarded_x_before_mutation, eager_out_before_mutation
    )

    return ScatterCopybackCase(
        x0=list(x0),
        src=list(src),
        eager_aliases_input=eager_aliases_input,
        compiled_aliases_input=compiled_aliases_input,
        compiled_input_corrupted_after_output_mutation=compiled_input_corrupted,
        guarded_aliases_input=guarded_aliases_input,
        guarded_input_corrupted_after_output_mutation=guarded_input_corrupted,
        guarded_values_match_eager=guarded_values_match_eager,
    )


@dataclasses.dataclass
class NoOpAliasCase:
    op_name: str
    neutral_arg: float
    eager_aliases_input: bool
    compiled_aliases_input: bool
    compiled_input_corrupted_after_output_mutation: bool
    guarded_aliases_input: bool
    guarded_input_corrupted_after_output_mutation: bool
    guarded_values_match_eager: bool


def _noop_elimination_fns(torch_module):
    """Three ops whose Inductor lowering hits joint_graph.remove_no_ops
    for a specific "neutral" constant argument (pytorch/pytorch#197893):
    dividing by 1.0, multiplying by 1.0, and adding 0. Each is paired
    with its own neutral arg value used in the case below."""

    def div_by_one(x, s):
        return x / s

    def mul_by_one(x, s):
        return x * s

    def add_zero(x, s):
        return x + s

    return {
        "x / 1.0": (div_by_one, 1.0),
        "x * 1.0": (mul_by_one, 1.0),
        "x + 0": (add_zero, 0.0),
    }


def _run_noop_case(torch_module, op_name: str, fn, neutral_arg: float) -> NoOpAliasCase:
    def run(callable_):
        x = torch_module.arange(1.0, 4.0)
        out = callable_(x, neutral_arg)
        return x, out

    # Eager reference: the correct, non-aliasing contract.
    eager_x, eager_out = run(fn)
    eager_aliases_input = eager_x.data_ptr() == eager_out.data_ptr()

    # Native compiled: reproduce the #197893 bug (fresh compile per
    # case -- avoids any risk of a stale cached graph masking it).
    compiled_fn = torch_module.compile(fn, fullgraph=True)
    compiled_x, compiled_out = run(compiled_fn)
    compiled_aliases_input = compiled_x.data_ptr() == compiled_out.data_ptr()
    compiled_x_before_mutation = compiled_x.clone()
    compiled_out.add_(100.0)
    compiled_input_corrupted = not torch_module.equal(compiled_x, compiled_x_before_mutation)

    # Guarded compiled: the SAME existing wrapper, unmodified, applied
    # to this second, independently-discovered root cause.
    guarded_fn = safe_compiled_scatter_returning(torch_module.compile(fn, fullgraph=True))
    guarded_x, guarded_out = run(guarded_fn)
    guarded_aliases_input = guarded_x.data_ptr() == guarded_out.data_ptr()
    guarded_x_before_mutation = guarded_x.clone()
    guarded_out.add_(100.0)
    guarded_input_corrupted = not torch_module.equal(guarded_x, guarded_x_before_mutation)
    guarded_values_match_eager = torch_module.equal(guarded_x_before_mutation, eager_x)

    return NoOpAliasCase(
        op_name=op_name,
        neutral_arg=neutral_arg,
        eager_aliases_input=eager_aliases_input,
        compiled_aliases_input=compiled_aliases_input,
        compiled_input_corrupted_after_output_mutation=compiled_input_corrupted,
        guarded_aliases_input=guarded_aliases_input,
        guarded_input_corrupted_after_output_mutation=guarded_input_corrupted,
        guarded_values_match_eager=guarded_values_match_eager,
    )


def diagnose(
    cases: Sequence[Tuple[Sequence[float], Sequence[float]]] = (
        ([1.0, 2.0], [10.0]),
        ([1.0, 2.0, 3.0, 4.0], [99.0]),
        ([5.0, -3.0, 7.0], [0.0]),
    ),
) -> Dict[str, Any]:
    """Reproduce BOTH independently-filed aliasing-contract divergences
    from scratch against the currently installed torch build -- the
    direct-scatter-copyback pattern (#195451) and the no-op-elimination
    pattern (#197893) -- and verify the SAME shared guard wrapper
    restores eager's non-aliasing contract for both. Never trusts a
    cached/prior result -- every call re-runs the actual repro,
    including a fresh ``torch.compile`` for each case (avoids any risk
    of a stale cached graph masking either bug)."""
    torch_module = _import_torch()
    results = [_run_case(torch_module, x0, src) for (x0, src) in cases]
    noop_results = [
        _run_noop_case(torch_module, op_name, fn, neutral_arg)
        for op_name, (fn, neutral_arg) in _noop_elimination_fns(torch_module).items()
    ]

    any_native_alias_bug = any(
        (not c.eager_aliases_input) and c.compiled_aliases_input for c in results
    )
    any_native_corruption = any(c.compiled_input_corrupted_after_output_mutation for c in results)
    guard_fully_correct = all(
        (not c.guarded_aliases_input)
        and (not c.guarded_input_corrupted_after_output_mutation)
        and c.guarded_values_match_eager
        for c in results
    )

    any_noop_alias_bug = any(
        (not c.eager_aliases_input) and c.compiled_aliases_input for c in noop_results
    )
    any_noop_corruption = any(
        c.compiled_input_corrupted_after_output_mutation for c in noop_results
    )
    noop_guard_fully_correct = all(
        (not c.guarded_aliases_input)
        and (not c.guarded_input_corrupted_after_output_mutation)
        and c.guarded_values_match_eager
        for c in noop_results
    )

    return {
        "torch_version": torch_module.__version__,
        "issue_url": "https://github.com/pytorch/pytorch/issues/195451",
        "fix_pr_url": "https://github.com/pytorch/pytorch/pull/195484",
        "cases": [dataclasses.asdict(c) for c in results],
        "any_native_alias_bug": any_native_alias_bug,
        "any_native_corruption": any_native_corruption,
        "guard_fully_correct": guard_fully_correct,
        "noop_issue_url": "https://github.com/pytorch/pytorch/issues/197893",
        "noop_cases": [dataclasses.asdict(c) for c in noop_results],
        "any_noop_alias_bug": any_noop_alias_bug,
        "any_noop_corruption": any_noop_corruption,
        "noop_guard_fully_correct": noop_guard_fully_correct,
    }
