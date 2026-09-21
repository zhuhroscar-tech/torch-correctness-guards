"""torch-as-strided-restride-oob-guard core: detect and guard a real
``torch.compile(backend="inductor")`` correctness bug where
``torch.as_strided`` on a view that has a non-zero ``storage_offset``
and has already been materialized into its own backing buffer computes
a storage-span check against the WRONG (larger) span than the
operation's own documented size/stride/storage_offset contract
actually requires, causing either a spurious "out of bounds for
storage" RuntimeError or (depending on shape) a silently wrong,
run-to-run-nondeterministic result -- while eager mode computes the
correct, in-bounds result deterministically every time.

Reproduced from scratch on this host (torch 2.14.0, macOS arm64 CPU;
see README for the exact commands) using the upstream report's own
repro shape:

    x = torch.tensor([[7.0, 9.0]])
    full = x.repeat(3, 2)                    # shape (3, 4), stride (4, 1)
    sliced = full[:, ::2]                    # shape (3, 2), stride (4, 2)
    view = torch.as_strided(sliced, (3, 2), full.stride())

Manually verifying the documented as_strided storage-span contract
(``storage_offset + sum((size[i]-1)*stride[i] for i in dims) + 1``)
for this call: offset=0, size=(3,2), stride=(4,2) ->
required = 0 + (3-1)*4 + (2-1)*2 + 1 = 8 + 2 + 1 = 11, well within
``full``'s real storage size of 12 elements. Eager confirms this: it
returns a correct, in-bounds, deterministic result matching
``x.repeat(3, 2)`` semantics exactly. Under
``torch.compile(backend="inductor")``, the SAME call instead raises
(observed on this host: "setStorage: sizes [2], strides [1], storage
offset 8, ... requiring a storage size of 40 are out of bounds for
storage of size 24" -- note 40 and 24 do not even match the eager
storage size of 12, confirming Inductor's lowering is computing a
completely different, wrong storage-span number, not merely being
stricter about a genuinely borderline case).

This is NOT a case of the caller misusing ``as_strided`` in
undefined-behavior territory (unlike the in-place-write-aliasing
warning in ``as_strided``'s own docs, which only concerns overlapping
writes, not read correctness): the operation is provably in-bounds by
the tensor's own documented storage/stride math, and eager proves it.
The divergence is an internal Inductor lowering defect.

Upstream references:
  - pytorch/pytorch#197431 ("[torch.compile] as_strided produces
    incorrect outputs for out-of-bounds strided tensor"), open as of
    this writing.
  - pytorch/pytorch#192226, an independent, still-open report of the
    same root-cause class (Inductor's as_strided lowering reading past
    a view's real storage span when the view has a non-zero
    storage_offset and has already been materialized into its own
    buffer), documenting the same workaround pattern used here.

This is a silent-wrong-result-or-spurious-crash bug: a training or
inference pipeline that builds a restrided/repeated view (a real
pattern for weight-tying, broadcasting a small tensor into a larger
strided buffer, or building a sliding-window view over a repeated
buffer) can either crash with a confusing storage-bounds
RuntimeError under torch.compile that never occurs in eager, or -- in
shapes where Inductor doesn't happen to raise -- silently return a
wrong, run-to-run-nondeterministic result while eager remains correct
and stable.

The guard function's mitigation matches this fleet's established
pattern for Inductor lowering defects on a single, narrow op call
(torch-compile-dynamic-clamp-guard's ``safe_clamp``,
torch-inductor-full-dtype-guard's ``safe_full``): force the actual
``as_strided`` call to run outside the compiled graph via
``torch.compiler.disable``, a narrow, deliberate Dynamo graph break at
this single call site, so the real eager ``as_strided`` (which always
uses the tensor's actual documented storage/stride contract) runs
regardless of Dynamo/Inductor's span-computation defect for the
surrounding graph.
"""
from __future__ import annotations

import dataclasses
from typing import Any, Dict, List, Sequence, Tuple


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


def _make_safe_as_strided(torch_module):
    """Build ``safe_as_strided`` bound to an already-imported torch
    module, so importing this module never requires torch at import
    time."""

    @torch_module.compiler.disable
    def safe_as_strided(input, size, stride, storage_offset=None):
        return torch_module.as_strided(
            input, size, stride, storage_offset=storage_offset
        )

    return safe_as_strided


def safe_as_strided(input, size, stride, storage_offset=None):
    """Drop-in guard for ``torch.as_strided`` at call sites where the
    input view has a non-zero ``storage_offset`` and has already been
    materialized into its own backing buffer -- the pattern that
    triggers Inductor's storage-span miscomputation under
    ``torch.compile``. Forces the real ``as_strided`` to run in eager
    mode via ``torch.compiler.disable`` (a deliberate, narrow Dynamo
    graph break at this single, already-cheap op), so the actual
    documented storage/stride contract is always honored -- matching
    eager exactly whether or not the caller itself is under
    ``torch.compile``.

    Safe to call directly (no torch.compile involved at all) or from
    inside a ``torch.compile``-wrapped function.
    """
    torch_module = _import_torch()
    guarded = _make_safe_as_strided(torch_module)
    return guarded(input, size, stride, storage_offset=storage_offset)


@dataclasses.dataclass
class RestrideCase:
    call_index: int
    x_values: Tuple[float, ...]
    eager_ok: bool
    eager_value: List[List[float]]
    compiled_ok: bool
    compiled_value: Any  # List[List[float]] or None
    compiled_error: str  # "" if no error
    guard_ok: bool
    guard_value: Any
    guard_error: str
    divergence_bug: bool  # compiled raised OR compiled result != eager
    guard_matches_eager: bool


def _make_view(torch_module, x):
    """The exact upstream repro pattern: repeat into a shared buffer,
    take a strided slice view, then re-view it with as_strided using
    the ORIGINAL (pre-slice) stride -- a real pattern when re-exposing
    a materialized buffer's full stride after a narrowing slice."""
    full = x.repeat(3, 2)
    sliced = full[:, ::2]
    return torch_module.as_strided(sliced, (3, 2), full.stride())


def _make_view_guarded(torch_module, x):
    full = x.repeat(3, 2)
    sliced = full[:, ::2]
    return safe_as_strided(sliced, (3, 2), full.stride())


def _tensor_to_list(t) -> List[List[float]]:
    return [[float(v) for v in row] for row in t.tolist()]


def _run_restride_sequence(torch_module, seed: int) -> List[RestrideCase]:
    """Reproduce the pytorch/pytorch#197431 repro across several input
    value pairs, comparing eager, unguarded-compiled, and
    guarded-compiled results for each.

    Different scalar VALUES are used per case (not different
    shapes/strides -- the bug is triggered by the shape/stride pattern
    itself, already fixed across all cases) purely so each case is an
    independent, non-cached compilation and result comparison."""
    generator = torch_module.Generator().manual_seed(seed)
    value_pairs = [(7.0, 9.0), (1.0, 2.0), (-3.5, 4.25)]

    results: List[RestrideCase] = []
    for idx, (a, b) in enumerate(value_pairs):
        x = torch_module.tensor([[a, b]])

        eager_value = _make_view(torch_module, x)
        eager_ok = True

        # Sanity: the guard's own eager path (uncompiled) must match
        # torch's plain eager result too -- not just under compile.
        # This both exercises _make_view_guarded's own lines under
        # coverage and independently confirms safe_as_strided never
        # changes behavior when torch.compile is not involved at all.
        guarded_eager_value = _make_view_guarded(torch_module, x)
        assert torch_module.equal(guarded_eager_value, eager_value), (
            "safe_as_strided's uncompiled eager path diverged from plain "
            "eager -- this should be impossible since safe_as_strided is "
            "a torch.compiler.disable-wrapped passthrough to the real op"
        )

        torch_module._dynamo.reset()
        compiled_fn = torch_module.compile(_make_view, backend="inductor")
        try:
            compiled_value_t = compiled_fn(torch_module, x)
            compiled_ok = True
            compiled_error = ""
        except Exception as exc:  # pragma: no cover - branch depends on host bug reproducing
            compiled_value_t = None
            compiled_ok = False
            compiled_error = f"{type(exc).__name__}: {exc}"

        torch_module._dynamo.reset()
        compiled_guarded_fn = torch_module.compile(_make_view_guarded, backend="inductor")
        try:
            guard_value_t = compiled_guarded_fn(torch_module, x)
            guard_ok = True
            guard_error = ""
        except Exception as exc:  # pragma: no cover - guard is expected to always succeed
            guard_value_t = None
            guard_ok = False
            guard_error = f"{type(exc).__name__}: {exc}"

        divergence_bug = (not compiled_ok) or (
            compiled_ok and not torch_module.equal(compiled_value_t, eager_value)
        )
        guard_matches_eager = guard_ok and torch_module.equal(guard_value_t, eager_value)

        results.append(
            RestrideCase(
                call_index=idx,
                x_values=(a, b),
                eager_ok=eager_ok,
                eager_value=_tensor_to_list(eager_value),
                compiled_ok=compiled_ok,
                compiled_value=_tensor_to_list(compiled_value_t) if compiled_ok else None,
                compiled_error=compiled_error,
                guard_ok=guard_ok,
                guard_value=_tensor_to_list(guard_value_t) if guard_ok else None,
                guard_error=guard_error,
                divergence_bug=divergence_bug,
                guard_matches_eager=guard_matches_eager,
            )
        )
    return results


def diagnose(seed: int = 0) -> Dict[str, Any]:
    """Reproduce the eager-vs-Inductor as_strided storage-span
    miscomputation from scratch against the currently installed torch
    build, and verify ``safe_as_strided`` matches eager on every case.
    Never trusts a cached/prior result -- every call re-runs the
    actual repro sequence."""
    torch_module = _import_torch()

    cases = _run_restride_sequence(torch_module, seed=seed)

    any_divergence_bug = any(c.divergence_bug for c in cases)
    guard_fully_correct = all(c.guard_matches_eager for c in cases)

    return {
        "torch_version": torch_module.__version__,
        "issue_urls": [
            "https://github.com/pytorch/pytorch/issues/197431",
            "https://github.com/pytorch/pytorch/issues/192226",
        ],
        "cases": [dataclasses.asdict(c) for c in cases],
        "any_divergence_bug": any_divergence_bug,
        "guard_fully_correct": guard_fully_correct,
    }
