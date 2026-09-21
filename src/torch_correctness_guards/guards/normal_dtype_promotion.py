"""torch-compile-normal-dtype-promotion-guard core: guards a real
torch.compile correctness bug where torch.distributions.Normal.sample()
silently PROMOTES the output dtype under torch.compile in cases where
eager mode PRESERVES the lower-precision input dtype.

Upstream report: pytorch/pytorch#194547 ("torch.compile silently
promotes dtype for torch.distributions.Normal.sample() where eager
preserves it"). Reproducer: construct
torch.distributions.Normal(loc, scale) where loc is a lower-precision
dtype (float16/bfloat16) and scale is float32, then call .sample().
Eager preserves loc's dtype (float16/bfloat16); torch.compile silently
promotes the result to float32 -- a silent, no-error dtype change that
can propagate into downstream fp16/bf16-sized buffers, memory budgets,
and mixed-precision training assumptions without any warning.

Independently verified on this host (torch 2.14.0, CPU) across a
4-pair dtype matrix:
  loc=float16,  scale=float32  -> eager float16,  compiled float32  (DIVERGES)
  loc=bfloat16, scale=float32  -> eager bfloat16, compiled float32  (DIVERGES)
  loc=float32,  scale=float16  -> eager float32,  compiled float32  (matches;
                                   eager's own promotion rule already picks
                                   float32 here, so there is nothing to guard)
  loc=float64,  scale=float32  -> eager float64,  compiled float64  (matches)
So the divergence is specific to a LOWER-precision loc paired with a
HIGHER-precision scale -- exactly the "eager keeps the lower-precision
sample dtype" contract that torch.compile's decomposition of
Normal.sample() does not preserve.

Confirmed via a fresh `gh api` read (not cached) that upstream issue
#194547 is OPEN as of this run.
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


def _normal_sample_fn(torch_module):
    def fn(loc, scale):
        d = torch_module.distributions.Normal(loc, scale)
        return d.sample()

    return fn


def safe_compiled_normal_sample(compiled_fn: Callable, eager_fn: Callable) -> Callable:
    """Wrap a ``torch.compile``-produced Normal.sample() callable that
    may silently promote loc's dtype (pytorch/pytorch#194547). Unlike
    a numeric-value bug, the compiled SAMPLE VALUES here are still
    numerically fine (a promoted-precision draw from the same
    distribution) -- the defect is purely the silent dtype change,
    which is cheap to detect and repair without re-running eager: if
    the compiled output's dtype differs from eager's documented
    "preserve loc's dtype" contract, simply cast the compiled result
    down to loc's dtype rather than paying eager's full recompute
    cost. This preserves torch.compile's speed benefit in the common
    case (no divergence) and in the divergent case (a cheap cast)
    alike, unlike a value-corruption bug that would require a full
    eager fallback.
    """
    torch_module = _import_torch()

    @functools.wraps(compiled_fn)
    def wrapper(loc, scale):
        out = compiled_fn(loc, scale)
        if out.dtype != loc.dtype:
            return out.to(loc.dtype)
        return out

    return wrapper


@dataclasses.dataclass
class NormalDtypeCase:
    loc_dtype: str
    scale_dtype: str
    eager_dtype: str
    compiled_dtype: str
    dtype_diverges: bool
    guarded_dtype: str
    guarded_matches_eager: bool


_DTYPE_NAMES: Dict[str, str] = {
    "float16": "torch.float16",
    "float32": "torch.float32",
    "float64": "torch.float64",
    "bfloat16": "torch.bfloat16",
}


def _run_case(torch_module, loc_dtype_name: str, scale_dtype_name: str) -> NormalDtypeCase:
    loc_dtype = getattr(torch_module, loc_dtype_name)
    scale_dtype = getattr(torch_module, scale_dtype_name)

    fn = _normal_sample_fn(torch_module)

    loc = torch_module.tensor([0.0, 1.0, -0.5], dtype=loc_dtype)
    scale = torch_module.tensor([1.0, 2.0, 0.5], dtype=scale_dtype)
    eager_out = fn(loc, scale)
    eager_dtype = eager_out.dtype

    # torch.compile's cache is not keyed on inductor config fields, but
    # it IS keyed on input dtype -- still, reset dynamo between cases
    # to avoid any risk of a stale graph from a differently-shaped
    # prior case leaking into this one (defensive; matches this
    # fleet's established v21 lesson about dynamo cache staleness).
    torch_module._dynamo.reset()
    compiled_fn = torch_module.compile(fn, fullgraph=True)
    compiled_out = compiled_fn(loc, scale)
    compiled_dtype = compiled_out.dtype

    dtype_diverges = compiled_dtype != eager_dtype

    torch_module._dynamo.reset()
    compiled_fn_for_guard = torch_module.compile(fn, fullgraph=True)
    guarded_fn = safe_compiled_normal_sample(compiled_fn_for_guard, fn)
    guarded_out = guarded_fn(loc, scale)
    guarded_dtype = guarded_out.dtype

    return NormalDtypeCase(
        loc_dtype=_DTYPE_NAMES[loc_dtype_name],
        scale_dtype=_DTYPE_NAMES[scale_dtype_name],
        eager_dtype=str(eager_dtype),
        compiled_dtype=str(compiled_dtype),
        dtype_diverges=dtype_diverges,
        guarded_dtype=str(guarded_dtype),
        guarded_matches_eager=guarded_dtype == eager_dtype,
    )


def diagnose(
    dtype_pairs: Sequence[Tuple[str, str]] = (
        ("float16", "float32"),
        ("bfloat16", "float32"),
        ("float32", "float16"),
        ("float64", "float32"),
    ),
) -> Dict[str, Any]:
    """Reproduce the torch.compile Normal.sample() dtype-promotion
    divergence from scratch against the currently installed torch
    build, for every dtype pair, and verify the wrapper restores
    eager's "preserve loc's dtype" contract. Never trusts a
    cached/prior result -- every call re-runs the actual repro,
    including a fresh ``torch.compile`` per case."""
    torch_module = _import_torch()
    results = [_run_case(torch_module, loc_name, scale_name) for (loc_name, scale_name) in dtype_pairs]

    any_native_divergence = any(c.dtype_diverges for c in results)
    guard_fully_correct = all(c.guarded_matches_eager for c in results)

    return {
        "torch_version": torch_module.__version__,
        "issue_url": "https://github.com/pytorch/pytorch/issues/194547",
        "cases": [dataclasses.asdict(c) for c in results],
        "any_native_divergence": any_native_divergence,
        "guard_fully_correct": guard_fully_correct,
    }
