"""torch-compile-transpose-argmin-guard core: detect and guard a real
``torch.compile(backend="inductor")`` correctness bug where calling
``.argmin()``/``.argmax()`` on a tensor produced by transposing and
then applying a further elementwise op (e.g. ``+ scalar`` or
``.contiguous()``) returns a DIFFERENT (wrong) flat index than eager,
with no error, warning, or graph break.

Reproduced from scratch on this host (torch 2.14.0, macOS arm64 CPU;
see README for the exact commands and this run's numbers)::

    def foo(x):
        xt = x.t()
        out = xt + 0.5
        return out.argmin()

    torch.manual_seed(0)
    x = torch.randn(2, 2)
    eager    = foo(x)                                   # tensor(1), correct
    compiled = torch.compile(foo, backend="inductor")(x) # tensor(2), WRONG

Upstream reference: pytorch/pytorch#197739 ("[torch.compile] torch.compile
returns incorrect argmin index for non-contiguous (transposed) tensor"),
labeled "module: correctness (silent)" + "module: inductor" +
"oncall: pt2", open and unfixed (no linked PR) as of this writing. This
project independently reproduced the issue's own repro plus additional
cases (argmax, a larger 4x6 shape, and a ``.contiguous()`` trigger
variant) against the currently installed torch build rather than
trusting the issue report alone.

Root-cause sensitivity, confirmed by direct testing: the bug requires
BOTH a transpose (non-contiguous strides) AND a subsequent elementwise
or layout-changing op between the transpose and the reduction --
``x.t().argmin()`` alone (no intervening op) is NOT affected; adding
``+ 0.5`` or calling ``.contiguous()`` after the transpose IS affected.
This matches the issue author's own observation.

This module's guard function, ``safe_reduce_index``, forces the whole
transpose -> op -> argmin/argmax sequence to run outside the compiled
graph via ``torch.compiler.disable`` (a Dynamo graph break at this
single, already-cheap call site), so Inductor's problematic index
codegen for the transposed-then-mutated-layout case never runs under
compilation for that call. This guard is deliberately conservative
(always routes through eager for the reduction step) rather than
narrowly scoped to only the specific op combinations tested here,
because the issue does not fully characterize which intervening ops
trigger it (confirmed here: ``+ scalar`` and ``.contiguous()`` both
trigger it; a bare transpose with no intervening op does not).
"""
from __future__ import annotations

import dataclasses
from typing import Any, Dict, List, Sequence


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


def _make_safe_reduce_index(torch_module):
    """Build the guard function bound to a specific torch module (so
    the same core logic works against whatever torch build is
    actually installed, without importing torch at module load
    time)."""

    @torch_module.compiler.disable
    def _eager_reduce(x, op, mode):
        if op == "add":
            y = x + 0.5
        elif op == "contiguous":
            y = x.contiguous()
        elif op == "none":
            y = x
        else:
            raise ValueError(f"unknown op: {op!r}")
        if mode == "argmin":
            return y.argmin()
        elif mode == "argmax":
            return y.argmax()
        else:
            raise ValueError(f"unknown mode: {mode!r}")

    def safe_reduce_index(x, op="add", mode="argmin"):
        """Drop-in guard for ``x.t().<op>().argmin()`` /
        ``.argmax()``-style sequences: computes the transpose's
        already-materialized input `x` (expected to already be
        transposed by the caller), applies `op`, and reduces via
        `mode`, matching eager output exactly under
        ``torch.compile(backend="inductor")`` (pytorch/pytorch#197739).
        """
        return _eager_reduce(x, op, mode)

    return safe_reduce_index


@dataclasses.dataclass
class TransposeArgReduceCase:
    op: str
    mode: str
    shape: List[int]
    seed: int
    eager_value: int
    native_compiled_value: int
    native_diverges: bool
    guard_matches_eager: bool


def _run_case(torch_module, safe_reduce_index, shape: Sequence[int], op: str, mode: str, seed: int) -> TransposeArgReduceCase:
    def native(x):
        xt = x.t()
        if op == "add":
            y = xt + 0.5
        elif op == "contiguous":
            y = xt.contiguous()
        elif op == "none":
            y = xt
        else:
            raise ValueError(f"unknown op: {op!r}")
        return y.argmin() if mode == "argmin" else y.argmax()

    torch_module.manual_seed(seed)
    x = torch_module.randn(*shape)

    eager_out = native(x)

    torch_module._dynamo.reset()
    compiled_native = torch_module.compile(native, backend="inductor")
    native_out = compiled_native(x)

    torch_module._dynamo.reset()

    def guarded(x):
        xt = x.t()
        return safe_reduce_index(xt, op=op, mode=mode)

    compiled_guarded = torch_module.compile(guarded, backend="inductor")
    guarded_out = compiled_guarded(x)

    return TransposeArgReduceCase(
        op=op,
        mode=mode,
        shape=list(shape),
        seed=seed,
        eager_value=int(eager_out.item()),
        native_compiled_value=int(native_out.item()),
        native_diverges=int(eager_out.item()) != int(native_out.item()),
        guard_matches_eager=int(eager_out.item()) == int(guarded_out.item()),
    )


def diagnose(
    cases: Sequence[Dict[str, Any]] = (
        {"shape": (2, 2), "op": "add", "mode": "argmin", "seed": 0},
        {"shape": (2, 2), "op": "contiguous", "mode": "argmin", "seed": 0},
        {"shape": (2, 2), "op": "none", "mode": "argmin", "seed": 0},
        {"shape": (2, 2), "op": "add", "mode": "argmax", "seed": 0},
        {"shape": (4, 6), "op": "add", "mode": "argmin", "seed": 1},
        {"shape": (6, 3), "op": "contiguous", "mode": "argmax", "seed": 2},
    ),
) -> Dict[str, Any]:
    """Reproduce the eager-vs-Inductor transpose+op+argmin/argmax
    divergence from scratch against the currently installed torch
    build, for every case, and verify ``safe_reduce_index`` matches
    eager in every case. Never trusts a cached/prior result -- every
    call re-runs the actual repro."""
    torch_module = _import_torch()
    safe_reduce_index = _make_safe_reduce_index(torch_module)

    results: List[TransposeArgReduceCase] = [
        _run_case(
            torch_module,
            safe_reduce_index,
            case["shape"],
            case["op"],
            case["mode"],
            case["seed"],
        )
        for case in cases
    ]

    any_divergence = any(c.native_diverges for c in results)
    guard_fully_correct = all(c.guard_matches_eager for c in results)

    return {
        "torch_version": torch_module.__version__,
        "issue_urls": [
            "https://github.com/pytorch/pytorch/issues/197739",
        ],
        "cases": [dataclasses.asdict(c) for c in results],
        "any_transpose_argreduce_divergence": any_divergence,
        "guard_fully_correct": guard_fully_correct,
    }


# Public guard function bound lazily against the currently installed
# torch build (import-time torch import would break "torch not
# installed" degradation -- see TorchUnavailableError above).
def safe_reduce_index(x, op="add", mode="argmin"):
    """Module-level convenience wrapper: resolves torch on first call
    and delegates to the bound guard. See ``_make_safe_reduce_index``
    for the full rationale."""
    torch_module = _import_torch()
    return _make_safe_reduce_index(torch_module)(x, op=op, mode=mode)
