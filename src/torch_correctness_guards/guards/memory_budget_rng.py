"""torch-memory-budget-rng-guard core: detect and guard a real PyTorch
correctness bug (pytorch/pytorch#190758, filed 2026-09-19, open at
time of writing): torch.compile's AOTAutograd partitioner, when
``torch._functorch.config.activation_memory_budget < 1.0``, can force
an RNG-containing op (``F.dropout``, ``randn_like``, ...) to be
RECOMPUTED in the backward graph -- and that recompute draws FRESH
randomness instead of replaying the forward pass's own draw. The
backward then applies a different dropout mask than the forward
actually used, silently corrupting gradients. No error, no warning.

Root cause (per the upstream issue's own trace, confirmed independently
on this host): the ``budget == 0`` shortcut skips the knapsack
partitioner's "random op" recompute ban entirely, and at
``0 < budget < 1`` the knapsack path can still re-allow recompute of
banned RNG ops because ``get_recomputable_banned_nodes`` has no RNG
filter. The bug does NOT reproduce at ``budget == 1.0`` (the default,
no forced recompute at all) -- confirmed both by the upstream issue's
own ablation table and by this module's own ``diagnose()``.

Independently reproduced on this host (torch 2.14.0, CPU, Inductor
backend) using the issue author's own minimal reproducer: comparing
the actual backward gradient against the gradient IMPLIED by the mask
the compiled forward itself applied (inferred from which output
elements are exactly zero) -- a mismatch proves the backward redrew a
different mask than the forward used, without needing internal
PyTorch instrumentation.

This module's guard (``safe_compile``) does not and cannot patch
PyTorch's partitioner from userspace. Instead it provides a
call-boundary mitigation: temporarily force
``activation_memory_budget = 1.0`` (the one value confirmed NOT to
trigger the bug) for the duration of a guarded compiled call, then
restore whatever budget the caller had configured. This sacrifices
budget<1.0's memory savings for that call in exchange for correctness
-- an explicit, documented trade-off, not a silent one.
"""
from __future__ import annotations

import contextlib
import dataclasses
from typing import Any, Dict, List, Sequence


class TorchUnavailableError(RuntimeError):
    """Raised when torch cannot be imported. Kept as a distinct type so
    callers can distinguish "torch isn't installed" from an actual
    diagnostic failure."""


def _import_torch():
    try:
        import torch  # noqa: F401
        import torch._dynamo  # noqa: F401
        import torch.nn.functional as F  # noqa: F401
        import torch._functorch.config as functorch_config  # noqa: F401
    except Exception as exc:  # pragma: no cover - exercised only without torch
        raise TorchUnavailableError(
            "torch is required for diagnosis and guarding; install the "
            "'torch' extra."
        ) from exc
    import torch
    import torch._dynamo
    import torch.nn.functional as F
    import torch._functorch.config as functorch_config

    return torch, F, functorch_config


@contextlib.contextmanager
def _forced_budget(functorch_config, budget: float):
    """Temporarily set activation_memory_budget to ``budget`` for the
    duration of the ``with`` block, restoring the prior value
    afterward -- even if the wrapped call raises."""
    prior = getattr(functorch_config, "activation_memory_budget", 1.0)
    functorch_config.activation_memory_budget = budget
    try:
        yield
    finally:
        functorch_config.activation_memory_budget = prior


def safe_compile(fn, backend: str = "inductor"):
    """Wrap ``fn`` so that every call to the returned callable compiles
    and runs with ``activation_memory_budget`` forced to 1.0 -- the one
    value confirmed not to trigger pytorch/pytorch#190758's RNG-recompute
    gradient corruption -- regardless of whatever budget the caller has
    configured globally, and restores the caller's prior budget setting
    immediately after the call returns (or raises).

    This is a call-boundary mitigation, not a fix to PyTorch's
    partitioner: it trades away budget<1.0's memory savings for THIS
    call in exchange for gradient correctness. Callers who need the
    memory savings AND have independently verified (e.g. via
    ``diagnose()`` against their own installed torch build) that their
    specific graph contains no RNG ops may bypass this guard and call
    ``torch.compile`` directly with their chosen budget.

    Calls ``torch._dynamo.reset()`` before its first compiled call to
    avoid a real caching pitfall discovered while building this guard:
    ``torch.compile`` does not key its compiled-graph cache on
    ``activation_memory_budget``, so a process that compiled the SAME
    function under a different budget earlier can silently reuse that
    stale compiled graph instead of recompiling under the forced
    budget=1.0 -- masking the very guard this function exists to
    provide. Guarded functions are always compiled fresh.
    """
    torch, _F, functorch_config = _import_torch()
    torch._dynamo.reset()
    compiled = torch.compile(fn, backend=backend)

    def wrapped(*args, **kwargs):
        with _forced_budget(functorch_config, 1.0):
            return compiled(*args, **kwargs)

    wrapped.__wrapped_compiled__ = compiled  # exposed for tests/introspection
    return wrapped


@dataclasses.dataclass
class BudgetRngCase:
    budget: float
    x_grad: List[List[float]]
    expected_grad_from_forward_mask: List[List[float]]
    matches: bool  # True == correct (no bug at this budget)


def _run_case(torch_module, functorch_config, F, budget: float) -> BudgetRngCase:
    """Reproduce pytorch/pytorch#190758's own minimal reproducer at a
    given budget: compile ``dropout(x @ w, 0.5, True)``, run forward +
    backward, then compare the ACTUAL backward gradient against the
    gradient analytically implied by the mask the forward itself
    applied (inferred from the forward output's own zero pattern).
    A mismatch means the backward redrew a different mask than the
    forward used -- the bug. Mirrors the upstream issue's exact
    reproducer so results are directly comparable to the filed report.

    Calls ``torch._dynamo.reset()`` first: ``torch.compile`` does not
    key its cache on ``activation_memory_budget``, so running several
    budgets in sequence in one process without resetting would let a
    later budget silently reuse an earlier budget's stale compiled
    graph (discovered while validating this diagnosis -- see
    ``safe_compile``'s docstring for the same pitfall).
    """
    torch_module._dynamo.reset()
    w = torch_module.arange(1.0, 26.0).view(5, 5)

    def f(x):
        return F.dropout(x @ w, 0.5, True)

    functorch_config.activation_memory_budget = budget
    compiled = torch_module.compile(f, backend="inductor")

    torch_module.manual_seed(42)
    x = torch_module.ones(2, 5, requires_grad=True)
    out = compiled(x)
    out.sum().backward()

    mask = (out.detach() != 0).to(torch_module.float64)
    expected = 2.0 * mask @ w.to(torch_module.float64).t()
    actual = x.grad.to(torch_module.float64)
    matches = bool(torch_module.equal(actual, expected))

    return BudgetRngCase(
        budget=budget,
        x_grad=actual.tolist(),
        expected_grad_from_forward_mask=expected.tolist(),
        matches=matches,
    )


def _run_guarded_case(torch_module, functorch_config, F, requested_budget: float) -> BudgetRngCase:
    """Same reproduction as ``_run_case`` but through ``safe_compile``
    with the caller requesting ``requested_budget`` (which safe_compile
    overrides to 1.0 internally) -- proves the guard prevents the
    divergence regardless of what the caller asked for."""
    w = torch_module.arange(1.0, 26.0).view(5, 5)

    def f(x):
        return F.dropout(x @ w, 0.5, True)

    functorch_config.activation_memory_budget = requested_budget
    guarded = safe_compile(f, backend="inductor")

    torch_module.manual_seed(42)
    x = torch_module.ones(2, 5, requires_grad=True)
    out = guarded(x)
    out.sum().backward()

    mask = (out.detach() != 0).to(torch_module.float64)
    expected = 2.0 * mask @ w.to(torch_module.float64).t()
    actual = x.grad.to(torch_module.float64)
    matches = bool(torch_module.equal(actual, expected))

    # Also confirm the guard restored the caller's requested budget
    # after the call (contract: temporary override, not permanent).
    restored_correctly = functorch_config.activation_memory_budget == requested_budget

    return BudgetRngCase(
        budget=requested_budget,
        x_grad=actual.tolist(),
        expected_grad_from_forward_mask=expected.tolist(),
        matches=matches and restored_correctly,
    )


def diagnose(
    unguarded_budgets: Sequence[float] = (0.0, 0.005, 0.5, 1.0),
    guarded_budgets: Sequence[float] = (0.0, 0.005, 0.5),
) -> Dict[str, Any]:
    """Reproduce pytorch/pytorch#190758 from scratch against the
    currently installed torch build, at each of ``unguarded_budgets``
    (direct ``torch.compile``, no guard), and confirm ``safe_compile``
    prevents the divergence at each of ``guarded_budgets`` (all of
    which are individually known-buggy unguarded, per the upstream
    issue's own ablation). Never trusts a cached/prior result -- every
    call re-runs the actual repro against the live torch install.
    """
    torch_module, F, functorch_config = _import_torch()
    original_budget = getattr(functorch_config, "activation_memory_budget", 1.0)
    try:
        unguarded = [_run_case(torch_module, functorch_config, F, b) for b in unguarded_budgets]
        guarded = [_run_guarded_case(torch_module, functorch_config, F, b) for b in guarded_budgets]
    finally:
        functorch_config.activation_memory_budget = original_budget

    any_unguarded_bug = any(not c.matches for c in unguarded if c.budget < 1.0)
    all_guarded_correct = all(c.matches for c in guarded)

    return {
        "torch_version": torch_module.__version__,
        "issue_url": "https://github.com/pytorch/pytorch/issues/190758",
        "unguarded_cases": [dataclasses.asdict(c) for c in unguarded],
        "guarded_cases": [dataclasses.asdict(c) for c in guarded],
        "any_unguarded_rng_recompute_bug": any_unguarded_bug,
        "guard_fully_correct": all_guarded_correct,
    }
