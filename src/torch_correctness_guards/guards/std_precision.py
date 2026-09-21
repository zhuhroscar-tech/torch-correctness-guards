"""torch-compile-std-precision-guard core: detect and fix a real
``torch.compile(backend="inductor")`` CPU precision bug in
``torch.std``, ``torch.var``, ``torch.var_mean`` and ``torch.std_mean``.

Reproduced from scratch on this host (torch 2.14.0, macOS arm64 CPU;
see README for the exact commands):

  Eager mode's CPU ``std``/``var`` family accumulates internally in
  double precision regardless of the input dtype, matching a float64
  reference to full accuracy across the whole float32 range. Under
  ``torch.compile(backend="inductor")`` the same reduction is compiled
  to accumulate in float32 instead, so the *compiled* result silently
  disagrees with the *eager* result on the same input:

    x = torch.randn(4, 8) * 1e30
    torch.std(x)                        # eager:    1.0223e+30 (matches fp64)
    torch.compile(torch.std)(x)         # inductor: inf

    x = torch.randn(5) * 1e-30
    torch.autograd.grad(torch.std(x), x)            # eager:    O(1) gradient (matches fp64)
    torch.autograd.grad(torch.compile(torch.std)(x), x)  # inductor: all-zero gradient

The all-zero-gradient case is the more dangerous one: it does not
raise, warn, or produce an obviously-wrong finite number -- it just
silently stops a parameter from learning, with no signal that
compilation (as opposed to the model or the data) is the cause. The
inf/NaN case is comparatively loud but still requires the user to
realize a code change (adding ``torch.compile``) is the source, not
data corruption elsewhere in the pipeline.

Upstream reference: pytorch/pytorch#197089 (open as of this writing),
reporting the exact divergence reproduced here for ``std``,
``var_mean``, and ``std_mean`` on CPU eager vs Inductor. ``var`` alone
already overflows in eager for these magnitudes (1e60 does not fit
float32... but eager's own accumulation is double, so torch.var's
final float32 *cast* overflows in both eager and compiled equally --
that specific case is NOT part of this guard's scope, since eager and
compiled already agree there. ``std``, ``var_mean``, and ``std_mean``
are affected because their eager path keeps enough working precision
that only the compiled path diverges.

This module's guard functions upcast the input to float64 before
calling the underlying torch reduction (so the compiled graph never
touches float32 accumulation for these ops), then cast the result back
to the input's original dtype -- matching the standard mitigation
pattern already used by this fleet's other torch guards
(torch-fp16-layernorm-tail-guard's float32-upcast strategy for a
different bug, applied here one precision level higher since the
divergence itself is a float32-vs-double accumulation gap). The guard
functions are safe to call under ``torch.compile`` themselves: the
upcast happens before the reduction op, so the compiled graph traces
the same double-precision computation eager already performs, and the
observed result should equal eager's own output.
"""
from __future__ import annotations

import dataclasses
from typing import Any, Dict, List, Optional, Sequence


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


def safe_std(input, dim=None, *, correction=1, keepdim=False):
    """Drop-in guard for ``torch.std``: upcasts to float64 before
    reducing, then casts back to the input's original dtype. Safe to
    call directly or under ``torch.compile`` -- the double-precision
    accumulation happens inside the traced graph either way, avoiding
    Inductor's float32 accumulation path entirely."""
    torch_module = _import_torch()
    orig_dtype = input.dtype
    x64 = input.double()
    if dim is None:
        result = torch_module.std(x64, correction=correction)
    else:
        result = torch_module.std(x64, dim=dim, correction=correction, keepdim=keepdim)
    return result.to(orig_dtype)


def safe_var(input, dim=None, *, correction=1, keepdim=False):
    """Drop-in guard for ``torch.var``. Included for API symmetry with
    ``safe_std``; note ``var`` itself already overflows equally in
    eager and compiled for extreme magnitudes (out of this guard's
    documented divergence scope) because squaring reduces float32's
    effective range further than std's final sqrt does -- the guard
    still upcasts for consistency and helps the ordinary (non-extreme)
    case match eager to full double precision."""
    torch_module = _import_torch()
    orig_dtype = input.dtype
    x64 = input.double()
    if dim is None:
        result = torch_module.var(x64, correction=correction)
    else:
        result = torch_module.var(x64, dim=dim, correction=correction, keepdim=keepdim)
    return result.to(orig_dtype)


def safe_var_mean(input, dim=None, *, correction=1, keepdim=False):
    """Drop-in guard for ``torch.var_mean``."""
    torch_module = _import_torch()
    orig_dtype = input.dtype
    x64 = input.double()
    if dim is None:
        v, m = torch_module.var_mean(x64, correction=correction)
    else:
        v, m = torch_module.var_mean(x64, dim=dim, correction=correction, keepdim=keepdim)
    return v.to(orig_dtype), m.to(orig_dtype)


def safe_std_mean(input, dim=None, *, correction=1, keepdim=False):
    """Drop-in guard for ``torch.std_mean``."""
    torch_module = _import_torch()
    orig_dtype = input.dtype
    x64 = input.double()
    if dim is None:
        s, m = torch_module.std_mean(x64, correction=correction)
    else:
        s, m = torch_module.std_mean(x64, dim=dim, correction=correction, keepdim=keepdim)
    return s.to(orig_dtype), m.to(orig_dtype)


@dataclasses.dataclass
class MagnitudeCase:
    op: str
    magnitude_exponent: int
    eager_value: Optional[float]
    compiled_value: Optional[float]
    eager_is_finite: bool
    compiled_is_finite: bool
    diverges: bool  # eager finite, compiled non-finite (or vice versa)
    guard_compiled_value: Optional[float]
    guard_matches_eager: bool


def _to_float_or_none(t) -> Optional[float]:
    try:
        return float(t.item())
    except Exception:
        return None


def _run_magnitude_case(torch_module, op_name: str, exponent: float, seed: int) -> MagnitudeCase:
    import torch as _t  # local alias for clarity inside closures

    gen = torch_module.Generator().manual_seed(seed)
    x = torch_module.randn(4, 8, generator=gen, dtype=torch_module.float32) * (10.0 ** exponent)

    eager_fn = getattr(torch_module, op_name)
    guard_fn = {
        "std": safe_std,
        "var": safe_var,
    }[op_name]

    eager_val = _to_float_or_none(eager_fn(x))

    torch_module._dynamo.reset()
    compiled_fn = torch_module.compile(eager_fn)
    compiled_val = _to_float_or_none(compiled_fn(x))

    torch_module._dynamo.reset()
    compiled_guard_fn = torch_module.compile(guard_fn)
    guard_val = _to_float_or_none(compiled_guard_fn(x))

    eager_finite = eager_val is not None and eager_val == eager_val and abs(eager_val) != float("inf")
    compiled_finite = compiled_val is not None and compiled_val == compiled_val and abs(compiled_val) != float("inf")

    # Divergence includes both a finite/non-finite disagreement (inf or
    # NaN appearing on only one side) AND a case where both sides are
    # technically "finite" but the compiled value has silently
    # collapsed to a materially different magnitude (e.g. flushed to
    # exactly 0.0 for a genuinely nonzero eager value) -- the latter is
    # just as real a correctness bug and must not be masked by a
    # finite-vs-finite check alone.
    magnitude_diverges = False
    if eager_finite and compiled_finite and eager_val is not None and compiled_val is not None:
        denom = max(abs(eager_val), 1e-300)
        magnitude_diverges = abs(eager_val - compiled_val) / denom > 1e-3

    diverges = (eager_finite != compiled_finite) or magnitude_diverges

    guard_matches_eager = (
        guard_val is not None
        and eager_val is not None
        and (
            (guard_val == eager_val)
            or (abs(guard_val - eager_val) <= 1e-6 * max(1.0, abs(eager_val)))
        )
    )

    return MagnitudeCase(
        op=op_name,
        magnitude_exponent=int(exponent),
        eager_value=eager_val,
        compiled_value=compiled_val,
        eager_is_finite=eager_finite,
        compiled_is_finite=compiled_finite,
        diverges=diverges,
        guard_compiled_value=guard_val,
        guard_matches_eager=guard_matches_eager,
    )


@dataclasses.dataclass
class ZeroGradientCase:
    magnitude_exponent: int
    eager_output: float
    eager_grad_is_zero: bool
    compiled_output: float
    compiled_grad_is_zero: bool
    silent_zero_gradient_bug: bool  # eager grad nonzero, compiled grad all-zero
    guard_compiled_output: float
    guard_grad_is_zero: bool
    guard_matches_eager: bool


def _run_zero_gradient_case(torch_module, exponent: float, seed: int) -> ZeroGradientCase:
    gen = torch_module.Generator().manual_seed(seed)
    x = torch_module.randn(5, generator=gen, dtype=torch_module.float32) * (10.0 ** exponent)

    def grads(fn, xin):
        xx = xin.clone().requires_grad_(True)
        y = fn(xx)
        (g,) = torch_module.autograd.grad(y, xx)
        return float(y.item()), g

    eager_y, eager_g = grads(torch_module.std, x)
    eager_grad_zero = bool((eager_g == 0).all().item())

    torch_module._dynamo.reset()
    compiled_std = torch_module.compile(torch_module.std)
    compiled_y, compiled_g = grads(compiled_std, x)
    compiled_grad_zero = bool((compiled_g == 0).all().item())

    torch_module._dynamo.reset()
    compiled_guard = torch_module.compile(safe_std)
    guard_y, guard_g = grads(compiled_guard, x)
    guard_grad_zero = bool((guard_g == 0).all().item())

    guard_matches_eager = abs(guard_y - eager_y) <= 1e-6 * max(1.0, abs(eager_y)) and (
        guard_grad_zero == eager_grad_zero
    )

    return ZeroGradientCase(
        magnitude_exponent=int(exponent),
        eager_output=eager_y,
        eager_grad_is_zero=eager_grad_zero,
        compiled_output=compiled_y,
        compiled_grad_is_zero=compiled_grad_zero,
        silent_zero_gradient_bug=(not eager_grad_zero) and compiled_grad_zero,
        guard_compiled_output=guard_y,
        guard_grad_is_zero=guard_grad_zero,
        guard_matches_eager=guard_matches_eager,
    )


def diagnose(
    magnitude_exponents: Sequence[int] = (-30, -15, 0, 15, 30),
) -> Dict[str, Any]:
    """Reproduce the eager-vs-Inductor std/var precision divergence
    from scratch against the currently installed torch build, at
    several input magnitudes, and verify the guard functions match
    eager under compilation. Never trusts a cached/prior result --
    every call re-runs the actual repro."""
    torch_module = _import_torch()

    std_cases: List[MagnitudeCase] = [
        _run_magnitude_case(torch_module, "std", exp, seed=exp + 1000)
        for exp in magnitude_exponents
    ]
    var_cases: List[MagnitudeCase] = [
        _run_magnitude_case(torch_module, "var", exp, seed=exp + 2000)
        for exp in magnitude_exponents
    ]
    zero_grad_cases: List[ZeroGradientCase] = [
        _run_zero_gradient_case(torch_module, exp, seed=exp + 3000)
        for exp in magnitude_exponents
    ]

    any_std_divergence = any(c.diverges for c in std_cases)
    any_silent_zero_gradient = any(c.silent_zero_gradient_bug for c in zero_grad_cases)
    guard_fully_correct = (
        all(c.guard_matches_eager for c in std_cases)
        and all(c.guard_matches_eager for c in zero_grad_cases)
    )

    return {
        "torch_version": torch_module.__version__,
        "issue_urls": ["https://github.com/pytorch/pytorch/issues/197089"],
        "std_cases": [dataclasses.asdict(c) for c in std_cases],
        "var_cases": [dataclasses.asdict(c) for c in var_cases],
        "zero_gradient_cases": [dataclasses.asdict(c) for c in zero_grad_cases],
        "any_std_divergence": any_std_divergence,
        "any_silent_zero_gradient": any_silent_zero_gradient,
        "guard_fully_correct": guard_fully_correct,
    }
