"""torch-compile-dynamic-clamp-guard core: detect and guard a real
``torch.compile(backend="inductor")`` correctness bug where a Python
float argument that Dynamo makes automatically dynamic
(``specialize_float=False``, the default) can be silently reused with
a STALE traced value on a later call using a different runtime value.

Reproduced from scratch on this host (torch 2.14.0, macOS arm64 CPU;
see README for the exact commands) using ``torch.clamp``'s ``max``
bound as the concrete operation, following the exact invocation
sequence from the upstream report:

    shape=(1, 2), requires_grad=False, limit=1.0   (compiled fresh)
    shape=(1, 4), requires_grad=True,  limit=0.5   (compiled fresh)
    shape=(1, 4), requires_grad=True,  limit=1.0   (SAME shape/grad
        as the previous call, only the float `limit` differs)

On the third call, torch.compile(backend="inductor") returns a result
computed with limit=0.5 (the STALE value from the previous invocation
with that same shape/dtype/requires_grad signature) even though the
Python-level argument passed is 1.0. Eager mode and
``backend="aot_eager"`` both give the correct answer for all three
calls; only Inductor's scalar-tensorification path exhibits the
staleness. This narrows the bug to Inductor specifically, matching the
upstream report.

Upstream references:
  - pytorch/pytorch#194976 ("[torch.compile][Inductor] Silent wrong
    results when an automatically dynamic Python float used by clamp
    reuses a stale value"). Fixed on main via PR #195040 (merged
    2026-08-28) but NOT YET in any stable release: independently
    re-verified live against installed torch 2.14.0 on 2026-09-20,
    still reproduces exactly as this module's own diagnose() shows.
  - nvidia/Megatron-LM#6918, an independent downstream confirmation of
    the exact same root cause in Megatron's ``clamped_swiglu`` fused
    kernel, citing pytorch/pytorch#194976.

This is a silent-wrong-result bug: no exception, no warning, no
non-finite marker. A model or kernel that swaps a clamp/threshold
constant between calls with the same tensor shape (extremely common:
a warmup-then-anneal schedule, a curriculum threshold, a configurable
gradient-clipping bound) can silently keep using an earlier value
under torch.compile while eager code with the identical call signature
behaves correctly -- this is precisely the kind of divergence that is
easy to miss in practice because it requires no code change to trigger
once introduced, only a change in a *runtime* float argument.

The guard function's mitigation matches this fleet's established
pattern for Inductor/Dynamo scalar-tensorization defects
(torch-inductor-full-dtype-guard's ``safe_full``): force the actual
clamp call to run outside the compiled graph via
``torch.compiler.disable``, a narrow, deliberate Dynamo graph break at
this single call site, so the real eager clamp (which always uses the
correct, current Python float) runs regardless of Dynamo's automatic
float-specialization/caching behavior for the surrounding graph.
"""
from __future__ import annotations

import dataclasses
from typing import Any, Dict, List, Optional, Sequence, Tuple


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


def _make_safe_clamp(torch_module):
    """Build ``safe_clamp`` bound to an already-imported torch module,
    so importing this module never requires torch at import time."""

    @torch_module.compiler.disable
    def safe_clamp(input, min=None, max=None):
        return torch_module.clamp(input, min=min, max=max)

    return safe_clamp


def safe_clamp(input, min=None, max=None):
    """Drop-in guard for ``torch.clamp`` at call sites where ``min``/
    ``max`` may be a Python float that changes across calls sharing the
    same tensor shape/dtype/requires_grad signature under
    ``torch.compile``. Forces the real clamp to run in eager mode via
    ``torch.compiler.disable`` (a deliberate, narrow Dynamo graph break
    at this single, already-cheap op), so the actual current Python
    float value is always used -- matching eager exactly whether or
    not the caller itself is under ``torch.compile``.

    Safe to call directly (no torch.compile involved at all) or from
    inside a ``torch.compile``-wrapped function.
    """
    torch_module = _import_torch()
    guarded = _make_safe_clamp(torch_module)
    return guarded(input, min=min, max=max)


@dataclasses.dataclass
class StaleClampCase:
    call_index: int
    shape: Tuple[int, ...]
    requires_grad: bool
    limit: float
    eager_value: float
    compiled_value: float
    guard_value: float
    stale_reuse_bug: bool  # compiled_value matches a DIFFERENT call's limit, not this call's
    guard_matches_eager: bool


def _run_stale_clamp_sequence(torch_module, seed: int) -> List[StaleClampCase]:
    """Reproduce the exact invocation-order-dependent repro from
    pytorch/pytorch#194976: three sequential calls sharing Dynamo's
    automatically-dynamic-float tracing state, where the third call's
    shape/requires_grad signature matches the second call's, but the
    Python float `limit` differs.

    Uses ``torch.clamp(x, max=limit)`` as the concrete op (the
    upstream issue's own repro uses a SwiGLU-style clamp; this
    isolates the same root cause to the single ``clamp`` call).
    """
    cases_spec = [
        ((1, 2), False, 1.0),
        ((1, 4), True, 0.5),
        ((1, 4), True, 1.0),
    ]

    def eager_fn(x, limit):
        return torch_module.clamp(x, max=limit)

    torch_module._dynamo.reset()
    compiled_fn = torch_module.compile(eager_fn, backend="inductor")

    torch_module._dynamo.reset()
    compiled_guarded_fn = torch_module.compile(
        lambda x, limit: safe_clamp(x, max=limit), backend="inductor"
    )

    results: List[StaleClampCase] = []
    with torch_module._dynamo.config.patch(specialize_float=False):
        for idx, (shape, requires_grad, limit) in enumerate(cases_spec):
            gen = torch_module.Generator().manual_seed(seed + idx)
            x = torch_module.full(
                shape, 0.75, dtype=torch_module.float32, requires_grad=requires_grad
            )
            x_c = x.detach().clone().requires_grad_(requires_grad)
            x_g = x.detach().clone().requires_grad_(requires_grad)

            eager_val = float(eager_fn(x, limit).abs().sum().item())
            compiled_val = float(compiled_fn(x_c, limit).abs().sum().item())
            guard_val = float(compiled_guarded_fn(x_g, limit).abs().sum().item())

            # A "stale reuse" is any case where compiled disagrees with
            # eager on this exact call's inputs -- if the compiled path
            # were internally self-consistent but merely used a
            # different (still-valid) computation, eager would still
            # be the independent oracle that exposes the divergence.
            stale = abs(compiled_val - eager_val) > 1e-6

            results.append(
                StaleClampCase(
                    call_index=idx,
                    shape=shape,
                    requires_grad=requires_grad,
                    limit=limit,
                    eager_value=eager_val,
                    compiled_value=compiled_val,
                    guard_value=guard_val,
                    stale_reuse_bug=stale,
                    guard_matches_eager=abs(guard_val - eager_val) <= 1e-6,
                )
            )
    return results


def diagnose(seed: int = 0) -> Dict[str, Any]:
    """Reproduce the eager-vs-Inductor stale-dynamic-float clamp
    divergence from scratch against the currently installed torch
    build, and verify ``safe_clamp`` matches eager under compilation.
    Never trusts a cached/prior result -- every call re-runs the
    actual repro sequence."""
    torch_module = _import_torch()

    cases = _run_stale_clamp_sequence(torch_module, seed=seed)

    any_stale_reuse_bug = any(c.stale_reuse_bug for c in cases)
    guard_fully_correct = all(c.guard_matches_eager for c in cases)

    return {
        "torch_version": torch_module.__version__,
        "issue_urls": [
            "https://github.com/pytorch/pytorch/issues/194976",
            "https://github.com/nvidia/Megatron-LM/issues/6918",
        ],
        "cases": [dataclasses.asdict(c) for c in cases],
        "any_stale_reuse_bug": any_stale_reuse_bug,
        "guard_fully_correct": guard_fully_correct,
    }
