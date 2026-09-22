"""torch-inductor-int64-index-truncation-guard core: detect and guard a
real ``torch.compile(backend="inductor")`` correctness bug where an
int64 arithmetic expression built from ``torch.arange(..., dtype=int64)``
silently overflows as if part of the computation happened in 32-bit
integer arithmetic, producing a wrong result with no error, warning, or
non-finite marker.

Upstream reference: pytorch/pytorch#183901 ("torch.compile produces
int32-overflowed results for int64 arange multiplication on CUDA"),
open as of this guard's creation (2026-09-20) -- independently
re-checked via `gh api` (state=open, closed_at=null) and `gh search
prs` (no merged PR references it; PR #184492 "Fix int64 index
expressions in Triton codegen" and PR #183740 "Separate value index
expressions from indexing" were both proposed fixes but both are
CLOSED, not merged, as of this guard's creation -- the author of
#184492 explicitly paused it in favor of #183740, which was itself
closed without merging), never trusted from a cached issue summary
alone.

The bug, reproduced from scratch on this host (torch 2.14.0, macOS
arm64 CPU -- confirmed NOT CUDA-specific, unlike the upstream issue's
own CUDA-only repro; see README for the exact commands):

    def f(x):
        return torch.arange(0, 9, dtype=torch.int64) * torch.tensor(
            [1500000000], dtype=torch.int64,
        )

    f(x)                                          # eager:  correct int64 values
    torch.compile(f, backend="inductor")(x)        # inductor: silently wrong

Eager correctly computes every element of ``arange(0, 9) *
1500000000`` in genuine int64 arithmetic (values up to
12,000,000,000, which overflows a 32-bit signed integer). Inductor's
Triton codegen path (per the upstream issue's own bisection:
``lowerings``/index-expression dtype propagation) treats the
value-producing index expression's *kernel index dtype* -- often
int32 for a small kernel -- as the compute dtype for the multiply,
before storing the result into a declared-int64 output tensor. The
mismatch is silent: both the eager and compiled tensors report dtype
``torch.int64`` (confirmed in this guard's own repro output), so a
naive dtype check on the output gives no signal that anything is
wrong -- only the actual numeric values differ, and only for elements
whose true product exceeds the 32-bit signed range.

This module's guard function, ``safe_int64_arange_mul`` (and the more
general ``guard_int64_op`` factory), forces the wrapped computation to
run entirely in eager mode via ``torch.compiler.disable`` -- a Dynamo
graph break around just the affected int64 arithmetic -- so the real
eager kernel (which performs genuine int64 arithmetic throughout) runs
regardless of whether the caller itself is under ``torch.compile``.
This trades a small amount of graph fusion at the guarded call site for
correctness, matching this fleet's established call-site-graph-break
guard pattern (see torch-inductor-full-dtype-guard's ``safe_full`` and
torch-compile-shuffle-sample-frozen-guard's ``safe_shuffle``/
``safe_sample`` for the same shape of fix).
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


def make_safe_int64_arange_mul(torch_module):
    """Build a guard function, bound to a specific torch module, that
    forces ``torch.arange(start, end, dtype=torch.int64) * multiplier``
    to run in eager mode via ``torch_module.compiler.disable``, so the
    real int64 multiply (not Inductor's truncated-to-kernel-index-dtype
    version) always executes -- regardless of whether the caller is
    itself under ``torch.compile``. Returns a callable
    ``(start, end, multiplier) -> Tensor`` matching eager's own
    (correct) semantics in every case."""

    @torch_module.compiler.disable
    def _safe_arange_mul(start, end, multiplier):
        return torch_module.arange(start, end, dtype=torch_module.int64) * multiplier

    return _safe_arange_mul


def guard_int64_op(torch_module, fn):
    """General-purpose guard: wraps an arbitrary int64-producing
    callable ``fn`` (taking any arguments) with
    ``torch_module.compiler.disable``, forcing it to execute eagerly
    whenever called from inside a ``torch.compile``'d region. Use this
    when the affected computation is not exactly the
    arange-times-scalar shape ``safe_int64_arange_mul`` covers (e.g. a
    different int64 expression shape that hits the same Inductor
    index-dtype-propagation defect)."""
    return torch_module.compiler.disable(fn)


@dataclasses.dataclass
class Int64OverflowCase:
    description: str
    multiplier: int
    arange_end: int
    eager_result: List[int]
    native_compiled_result: List[int]
    guarded_compiled_result: List[int]
    native_diverges: bool  # eager != native compiled (the bug)
    guard_matches_eager: bool


def _run_case(
    torch_module,
    safe_fn,
    description: str,
    arange_end: int,
    multiplier: int,
) -> Int64OverflowCase:
    def native_fn(x):
        return torch_module.arange(0, arange_end, device=x.device, dtype=torch_module.int64) * torch_module.tensor(
            [multiplier], dtype=torch_module.int64, device=x.device,
        )

    def guarded_fn(x):
        return safe_fn(0, arange_end, torch_module.tensor([multiplier], dtype=torch_module.int64, device=x.device))

    x = torch_module.zeros(1)

    eager_result = native_fn(x).tolist()

    # Reset Dynamo's compile cache before each case: without this, a
    # closure-captured `multiplier`/`arange_end` from an earlier case
    # in the same diagnose() call (or an earlier test in the same
    # pytest process) can be silently reused as a stale guard/cache
    # hit instead of triggering a fresh compile for the NEW closure
    # values -- confirmed by hand (this exact flake reproduced when
    # cases ran back-to-back without a reset, and disappeared once
    # this reset was added). Matches the established pattern in this
    # fleet's other torch-*-guard repos (e.g.
    # torch-inductor-full-dtype-guard's _run_bool_fill_case).
    torch_module._dynamo.reset()
    compiled_native = torch_module.compile(native_fn, backend="inductor")
    native_result = compiled_native(x).tolist()

    torch_module._dynamo.reset()
    compiled_guarded = torch_module.compile(guarded_fn, backend="inductor")
    guarded_result = compiled_guarded(x).tolist()

    return Int64OverflowCase(
        description=description,
        multiplier=multiplier,
        arange_end=arange_end,
        eager_result=eager_result,
        native_compiled_result=native_result,
        guarded_compiled_result=guarded_result,
        native_diverges=(eager_result != native_result),
        guard_matches_eager=(eager_result == guarded_result),
    )


def diagnose() -> Dict[str, Any]:
    """Reproduce the int64-arange-multiply Inductor truncation bug from
    scratch against the currently installed torch build (CPU, no CUDA
    required -- confirmed this bug is not CUDA-specific, widening the
    upstream issue's own CUDA-only repro), and verify
    ``make_safe_int64_arange_mul``'s guard matches eager exactly in
    every case. Never trusts a cached/prior result -- every call
    re-runs the actual repro."""
    torch_module = _import_torch()
    safe_fn = make_safe_int64_arange_mul(torch_module)

    # Start from a clean Dynamo compile cache: diagnose() may be
    # called multiple times in the same process (e.g. once per test),
    # and a stale cache entry from a prior call/case can otherwise be
    # silently reused instead of recompiling for this call's closures.
    torch_module._dynamo.reset()

    # (description, arange_end, multiplier). The upstream issue's own
    # exact repro is arange(0, 9) * 1_500_000_000; a second, smaller
    # case confirms the guard generalizes and isn't tuned to one magic
    # constant, and a case with a multiplier too small to ever overflow
    # confirms the guard doesn't introduce a false positive divergence
    # on an already-correct (small-value) case.
    cases_spec = [
        ("upstream issue's exact repro: arange(0,9) * 1_500_000_000", 9, 1_500_000_000),
        ("smaller range, same overflow-inducing multiplier: arange(0,5) * 1_500_000_000", 5, 1_500_000_000),
        ("no-overflow control: arange(0,9) * 2 (values stay small, must not spuriously diverge)", 9, 2),
    ]

    cases: List[Int64OverflowCase] = [
        _run_case(torch_module, safe_fn, desc, end, mult)
        for desc, end, mult in cases_spec
    ]

    any_native_diverges = any(c.native_diverges for c in cases)
    guard_fully_correct = all(c.guard_matches_eager for c in cases)

    return {
        "torch_version": torch_module.__version__,
        "issue_url": "https://github.com/pytorch/pytorch/issues/183901",
        "cases": [dataclasses.asdict(c) for c in cases],
        "any_native_diverges": any_native_diverges,
        "guard_fully_correct": guard_fully_correct,
    }


# Public convenience wrapper: resolves torch lazily so importing this
# module without torch installed doesn't crash (matching the sibling
# guard repos' degradation pattern).
def safe_int64_arange_mul(start, end, multiplier):
    """Module-level convenience wrapper around
    ``make_safe_int64_arange_mul``: resolves torch on first call. See
    that function's docstring for the full rationale and semantics."""
    torch_module = _import_torch()
    return make_safe_int64_arange_mul(torch_module)(start, end, multiplier)
