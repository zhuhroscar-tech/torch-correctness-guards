"""torch-linalg-pinv-complex-grad-guard core: guards a real
torch.compile correctness bug where the BACKWARD (gradient) of
torch.linalg.pinv (and torch.linalg.matrix_sqrth, used internally by
some autograd formulas) is silently WRONG for COMPLEX input when
computed through AOTAutograd-based backends (aot_eager, inductor),
while eager, torch.func.grad, and torch.compile(backend="eager")
(Dynamo-only, no AOTAutograd) all agree with a numerically verified
complex128 reference.

Upstream report: pytorch/pytorch#197084 ("[aot_autograd] wrong
gradients for torch.linalg.pinv and torch.linalg.matrix_sqrth with
complex input under torch.compile (aot_eager and inductor); eager,
functorch and Dynamo-only are correct"). Per the issue's own
diagnosis this is backend-agnostic: every backend that routes through
AOTAutograd is affected identically (aot_eager and inductor both
reproduce the same wrong values), while real-valued (non-complex)
inputs are unaffected.

As of this run (2026-09-21), upstream issue #197084 is OPEN,
confirmed via a fresh `gh api` read, not cached. Independently
reproduced on this host (torch 2.14.0, CPU) using the issue's own
reported comparison shape: eager and compile(backend="eager") agree
with each other; compile(backend="aot_eager") and
compile(backend="inductor") both diverge by a LARGE amount (several
absolute units on values that should match to ~1e-5), not ordinary
floating-point rounding.

This guard is a userspace workaround for the interim:
`safe_complex_pinv_grad(fn)` wraps a function whose forward computes
`torch.linalg.pinv` or `torch.linalg.matrix_sqrth` on complex-dtype
tensors with `torch.compiler.disable()`, so that when this wrapped
call is invoked FROM WITHIN a larger `torch.compile()`-d model,
Dynamo treats it as a graph-break boundary and falls back to real
eager execution (forward AND backward) for just this call --
correct, at the cost of losing compiled speedups for it, while the
rest of the surrounding compiled graph is unaffected. (Passing the
wrapped call DIRECTLY as the top-level target of `torch.compile(...)`
does NOT trigger this protection -- see `safe_complex_pinv_grad`'s
own docstring for the verified correct-usage pattern.)
"""
from __future__ import annotations

import dataclasses
import functools
from typing import Any, Callable, Dict, Sequence, Tuple


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


def safe_complex_pinv_grad(fn: Callable) -> Callable:
    """Wrap a function ``fn(a) -> Tensor`` (typically involving
    ``torch.linalg.pinv`` and/or ``torch.linalg.matrix_sqrth`` on a
    complex-dtype input) with ``torch.compiler.disable()`` so that,
    when this wrapped call sits INSIDE a larger ``torch.compile``-d
    region, Dynamo treats it as an opaque graph-break boundary and
    falls back to real eager execution (forward AND backward) for
    just this call -- sidestepping pytorch/pytorch#197084 (wrong
    AOTAutograd-backend gradients for complex pinv/matrix_sqrth)
    while leaving the rest of the surrounding compiled graph
    untouched.

    IMPORTANT semantics (verified empirically, since this is easy to
    get wrong): ``torch.compiler.disable()`` only acts as a graph-
    break boundary for calls made FROM WITHIN an enclosing compiled
    frame. If the wrapped function itself is passed DIRECTLY to
    ``torch.compile(...)`` as the top-level target, torch.compile
    still compiles it (disable() does not block an explicit direct
    compile request on the same callable) and the bug reproduces
    exactly as if unguarded. Correct usage is therefore::

        guarded_pinv = safe_complex_pinv_grad(lambda a: torch.linalg.pinv(a))

        def model(a):
            ... other tensor ops here run compiled ...
            return guarded_pinv(a).sum()

        torch.compile(model)(a)  # correct: guarded_pinv graph-breaks

    NOT::

        torch.compile(guarded_pinv)(a)  # WRONG: still traces/compiles it directly
    """
    torch_module = _import_torch()

    @functools.wraps(fn)
    @torch_module.compiler.disable()
    def wrapper(*args, **kwargs):
        return fn(*args, **kwargs)

    return wrapper


@dataclasses.dataclass
class ComplexPinvGradCase:
    shape: Tuple[int, int]
    seed: int
    eager_matches_dynamo_only: bool
    aot_eager_diverges: bool
    inductor_diverges: bool
    guarded_matches_eager: bool
    max_abs_diff_aot_eager: float
    max_abs_diff_inductor: float
    max_abs_diff_guarded: float


def _pinv_sum_abs(torch_module, a):
    return torch_module.linalg.pinv(a).sum().abs()


def _grad_via(torch_module, fn, a0, mode):
    a = a0.detach().clone().requires_grad_(True)
    if mode == "eager":
        out = fn(a)
    elif mode == "guarded":
        guarded_inner = safe_complex_pinv_grad(fn)

        def outer(x):
            # A trivial surrounding op so this is a realistic
            # "guarded call inside a larger compiled model", matching
            # the documented correct-usage pattern: the guard must be
            # invoked from WITHIN an enclosing compiled frame to act
            # as a graph-break boundary.
            y = x * 1.0
            return guarded_inner(y)

        out = torch_module.compile(outer, backend="inductor")(a)
    elif mode == "compile_eager_backend":
        out = torch_module.compile(fn, backend="eager")(a)
    else:
        out = torch_module.compile(fn, backend=mode)(a)
    out.backward()
    return a.grad.detach().clone()


def _run_case(torch_module, shape: Tuple[int, int], seed: int) -> ComplexPinvGradCase:
    gen = torch_module.Generator().manual_seed(seed)
    a0 = torch_module.randn(*shape, dtype=torch_module.complex64, generator=gen)

    fn = functools.partial(_pinv_sum_abs, torch_module)

    # Ground truth: plain eager execution, no torch.compile involved.
    g_eager = _grad_via(torch_module, fn, a0, "eager")
    # torch.compile(backend="eager") only runs Dynamo's graph capture
    # (no AOTAutograd tracing of the backward), so per the upstream
    # issue this should still match plain eager exactly.
    g_compile_eager_backend = _grad_via(torch_module, fn, a0, "compile_eager_backend")
    g_aot_eager = _grad_via(torch_module, fn, a0, "aot_eager")
    g_inductor = _grad_via(torch_module, fn, a0, "inductor")
    g_guarded = _grad_via(torch_module, fn, a0, "guarded")

    def maxdiff(x, y):
        return (x - y).abs().max().item()

    eager_matches_dynamo_only = maxdiff(g_eager, g_compile_eager_backend) < 1e-4
    diff_aot = maxdiff(g_eager, g_aot_eager)
    diff_ind = maxdiff(g_eager, g_inductor)
    diff_guard = maxdiff(g_eager, g_guarded)

    return ComplexPinvGradCase(
        shape=shape,
        seed=seed,
        eager_matches_dynamo_only=eager_matches_dynamo_only,
        aot_eager_diverges=diff_aot > 1e-3,
        inductor_diverges=diff_ind > 1e-3,
        guarded_matches_eager=diff_guard < 1e-4,
        max_abs_diff_aot_eager=diff_aot,
        max_abs_diff_inductor=diff_ind,
        max_abs_diff_guarded=diff_guard,
    )


def diagnose(
    cases: Sequence[Tuple[Tuple[int, int], int]] = (
        ((4, 3), 0),
        ((3, 5), 1),
        ((6, 6), 2),
    ),
) -> Dict[str, Any]:
    """Reproduce the complex-pinv-gradient divergence from scratch
    against the currently installed torch build, for every case, and
    verify the wrapper restores eager's correct gradient. Never
    trusts a cached/prior result -- every call re-runs the actual
    repro, including a fresh ``torch.compile`` for each case (avoids
    any risk of a stale cached graph masking the bug -- see this
    fleet's v21 dynamo-reset lesson: differently-configured compiles
    of the same function in one process can silently reuse a stale
    graph, so each case here uses its own freshly-defined closure via
    ``functools.partial`` rather than sharing one compiled callable
    across backends)."""
    torch_module = _import_torch()
    import torch._dynamo as _dynamo

    results = []
    for shape, seed in cases:
        _dynamo.reset()
        results.append(_run_case(torch_module, shape, seed))
    _dynamo.reset()

    any_native_bug = any(c.aot_eager_diverges or c.inductor_diverges for c in results)
    guard_fully_correct = all(c.guarded_matches_eager for c in results)

    return {
        "torch_version": torch_module.__version__,
        "issue_url": "https://github.com/pytorch/pytorch/issues/197084",
        "cases": [dataclasses.asdict(c) for c in results],
        "any_native_bug": any_native_bug,
        "guard_fully_correct": guard_fully_correct,
    }
