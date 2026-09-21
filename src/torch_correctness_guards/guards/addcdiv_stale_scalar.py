"""torch-addcdiv-stale-scalar-guard core: detect and guard a real
``torch.compile(backend="inductor")`` correctness bug where a Python
scalar derived from ``.item()`` inside a loop -- and used later via
``addcdiv_``/``addcmul_`` (or any downstream arithmetic depending on
that scalar) -- gets silently cached with a STALE value across loop
iterations once a `.item()` graph break occurs inside a helper function
called from the loop body.

Upstream reference: pytorch/pytorch#185382 ("torch.compile: Inductor
silently produces wrong results for addcdiv_/addcmul_ when .item()
graph break occurs in a loop"), closed 2026-09-04 as fixed by PR #195040
("Fix stale float specialization in tensorify_python_scalars",
fixing #194976 -- the underlying stale-float-specialization root cause
that also produced #185382's symptom). The fix landed in PyTorch main
on 2026-08-27/28 but is NOT present in released torch 2.14.0 (the
version this repo's own CI installs via `torch>=2.0` and the version
this guard was reproduced against on 2026-09-19) -- so any project
pinned to torch<=2.14.x that has NOT cherry-picked #195040 remains
exposed. This guard lets such a project get eager-correct behavior
today without waiting for or forcing an upstream version bump, and the
diagnostic below independently re-verifies the bug's presence/absence
against whatever torch build the guard is actually installed alongside
(never trusting the issue tracker's report alone).

The bug, reproduced from scratch on this host (torch 2.14.0, macOS
arm64 CPU; see README for the exact commands): a hand-written Adam-style
optimizer step that reads a step counter via ``step_t.item()`` inside a
helper function (an extremely common, well-documented pattern for a
custom training loop -- e.g. the PyTorch docs' own optimizer tutorials
put bias-correction scalar math right after ``.item()``), then uses that
scalar in a bias-correction computation feeding ``addcdiv_``'s ``value=``
argument, diverges from eager starting at the 3rd loop iteration under
``torch.compile(backend="inductor")``. The error grows unboundedly with
iteration count because each iteration's wrong step_size compounds into
the next iteration's parameter values. ``aot_eager`` backend produces
the correct (eager-matching) result, confirming this is specifically an
Inductor codegen/caching issue, not a generic graph-capture problem.

This module's guard, ``safe_stale_scalar_step``, forces the entire
scalar-dependent computation (the ``.item()`` call itself plus every
downstream arithmetic expression that depends on its value, up to and
including the final ``value=`` scalar handed to ``addcdiv_``/
``addcmul_``) to run in real eager mode via ``torch.compiler.disable``,
so Inductor never has the opportunity to bake a stale float
specialization into a cached FX graph. This is a narrower, drop-in
alternative to setting ``torch._dynamo.config.capture_scalar_outputs =
True`` process-wide (the upstream issue's own documented workaround),
which changes graph-capture behavior for the entire process rather than
just the one call site that needs it.
"""
from __future__ import annotations

import dataclasses
from typing import Any, Callable, Dict, List, Sequence


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


def make_safe_stale_scalar_step(torch_module):
    """Build a guard function, bound to a specific torch module, that
    forces the Adam-style bias-correction scalar computation (the
    ``.item()`` read plus everything derived from it) to run in eager
    mode via ``torch.compiler.disable`` -- so Inductor cannot bake a
    stale specialized float into a cached graph across loop
    iterations. Returns a callable
    ``(step_tensor, beta1, beta2, lr) -> (step, bias_correction1,
    bias_correction2, step_size)`` matching the shape of the bias-
    correction math in a typical hand-written Adam step."""

    @torch_module.compiler.disable
    def _eager_bias_correction(step_tensor, beta1, beta2, lr):
        step = step_tensor.item()
        bc1 = 1 - beta1 ** step
        bc2 = 1 - beta2 ** step
        step_size = lr / bc1
        return step, bc1, bc2, step_size

    return _eager_bias_correction


@dataclasses.dataclass
class AdamStepCase:
    iterations: int
    eager_final_param: List[float]
    native_final_param: List[float]
    guarded_final_param: List[float]
    native_diverges: bool
    guard_matches_eager: bool
    max_abs_diff_native: float
    max_abs_diff_guarded: float


def _adam_step_native(torch_module, param, grad, m, v, step_t, beta1=0.9, beta2=0.999, lr=0.001, eps=1e-8):
    def _read_step(t):
        return t.item()  # graph break: matches the upstream issue's helper-function pattern

    step_t.add_(1)
    m.mul_(beta1).add_(grad, alpha=1 - beta1)
    v.mul_(beta2).addcmul_(grad, grad, value=1 - beta2)
    step = _read_step(step_t)
    bc1 = 1 - beta1 ** step
    bc2 = 1 - beta2 ** step
    step_size = lr / bc1
    denom = (v.sqrt() / (bc2 ** 0.5)).add_(eps)
    param.addcdiv_(m, denom, value=-step_size)
    return param


def _adam_step_guarded(torch_module, safe_bias_correction, param, grad, m, v, step_t, beta1=0.9, beta2=0.999, lr=0.001, eps=1e-8):
    step_t.add_(1)
    m.mul_(beta1).add_(grad, alpha=1 - beta1)
    v.mul_(beta2).addcmul_(grad, grad, value=1 - beta2)
    step, bc1, bc2, step_size = safe_bias_correction(step_t, beta1, beta2, lr)
    denom = (v.sqrt() / (bc2 ** 0.5)).add_(eps)
    param.addcdiv_(m, denom, value=-step_size)
    return param


def _run_adam_case(torch_module, iterations: int, seed: int) -> AdamStepCase:
    torch_module.manual_seed(seed)
    p0 = torch_module.randn(4, dtype=torch_module.float64)
    g = torch_module.randn(4, dtype=torch_module.float64)

    def fresh_state():
        return (
            p0.clone(),
            torch_module.zeros(4, dtype=torch_module.float64),
            torch_module.zeros(4, dtype=torch_module.float64),
            torch_module.zeros((), dtype=torch_module.float64),
        )

    # eager baseline
    p, m, v, step_t = fresh_state()
    for _ in range(iterations):
        _adam_step_native(torch_module, p, g, m, v, step_t)
    eager_final = p.detach().clone()

    # inductor, unguarded (native)
    p, m, v, step_t = fresh_state()
    torch_module._dynamo.reset()
    compiled_native = torch_module.compile(
        lambda param, grad, mm, vv, st: _adam_step_native(torch_module, param, grad, mm, vv, st),
        backend="inductor",
    )
    for _ in range(iterations):
        compiled_native(p, g, m, v, step_t)
    native_final = p.detach().clone()

    # inductor, guarded
    p, m, v, step_t = fresh_state()
    safe_bias_correction = make_safe_stale_scalar_step(torch_module)
    torch_module._dynamo.reset()
    compiled_guarded = torch_module.compile(
        lambda param, grad, mm, vv, st: _adam_step_guarded(
            torch_module, safe_bias_correction, param, grad, mm, vv, st
        ),
        backend="inductor",
    )
    for _ in range(iterations):
        compiled_guarded(p, g, m, v, step_t)
    guarded_final = p.detach().clone()

    diff_native = (eager_final - native_final).abs().max().item()
    diff_guarded = (eager_final - guarded_final).abs().max().item()

    return AdamStepCase(
        iterations=iterations,
        eager_final_param=eager_final.tolist(),
        native_final_param=native_final.tolist(),
        guarded_final_param=guarded_final.tolist(),
        native_diverges=diff_native > 1e-9,
        guard_matches_eager=diff_guarded <= 1e-9,
        max_abs_diff_native=diff_native,
        max_abs_diff_guarded=diff_guarded,
    )


def diagnose(iteration_counts: Sequence[int] = (3, 6, 10), seed: int = 20260919) -> Dict[str, Any]:
    """Reproduce the eager-vs-Inductor stale-scalar divergence from
    scratch against the currently installed torch build, at several
    loop-iteration counts (the bug's own trigger condition requires
    >= 3 iterations and its error compounds with more), and verify
    ``make_safe_stale_scalar_step``'s guard matches eager in every
    case. Never trusts a cached/prior result -- every call re-runs the
    actual repro."""
    torch_module = _import_torch()

    cases: List[AdamStepCase] = [
        _run_adam_case(torch_module, n, seed) for n in iteration_counts
    ]

    any_native_diverges = any(c.native_diverges for c in cases)
    guard_fully_correct = all(c.guard_matches_eager for c in cases)

    return {
        "torch_version": torch_module.__version__,
        "issue_url": "https://github.com/pytorch/pytorch/issues/185382",
        "upstream_fix_pr": "https://github.com/pytorch/pytorch/pull/195040",
        "cases": [dataclasses.asdict(c) for c in cases],
        "any_native_diverges": any_native_diverges,
        "guard_fully_correct": guard_fully_correct,
    }


# Public convenience wrapper: resolves torch lazily so importing this
# module without torch installed doesn't crash (matching the sibling
# guard repos' degradation pattern).
def safe_stale_scalar_step(step_tensor, beta1, beta2, lr):
    """Module-level convenience wrapper around
    ``make_safe_stale_scalar_step``: resolves torch on first call. See
    that function's docstring for the full rationale and return shape."""
    torch_module = _import_torch()
    return make_safe_stale_scalar_step(torch_module)(step_tensor, beta1, beta2, lr)
