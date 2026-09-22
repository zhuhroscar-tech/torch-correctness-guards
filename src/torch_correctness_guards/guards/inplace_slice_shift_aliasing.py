"""torch-inductor-inplace-slice-shift-aliasing-guard core: guards a real
torch.compile (Inductor) correctness bug where an in-place row-shift
assignment silently corrupts data because the assignment is lowered to
an overlapping-memory copy.

Upstream report: pytorch/pytorch#197829 ("[inductor] `x[1:] =
x[:-1].clone()` compiles to a copy over overlapping memory: silent
wrong results on CUDA and CPU").

Root cause per the issue author's own trace (upstream comment,
2026-09-20): ``remove_noop_ops`` drops the ``.clone()`` as a no-op
while the graph is still functional (the clone looks redundant in
isolation), then the reinplacing pass turns the resulting
``_generalized_scatter(x, src, ...)`` into ``view(x).copy_(src)`` --
but by then ``src`` is a VIEW OF ``x``'s OWN STORAGE (the clone that
would have made it independent was already removed), and
``should_reinplace_scatter``'s ``can_inplace`` check only looks at
LATER uses of that storage, not whether ``src`` itself already aliases
``x``. The in-place copy then reads and writes the same overlapping
buffer: on CPU this is wrong at ANY tensor size, the same way every
run (row 0 clobbers every row); on CUDA it is correct at small shapes
but wrong non-deterministically at larger ones.

This is a DISTINCT root cause and code path from this fleet's sibling
repo torch-inductor-scatter-copyback-alias-guard (pytorch#195451,
which guards a RETURN-VALUE aliasing contract change with correct
underlying values) and torch-inductor-duplicate-index-writeorder-guard
(pytorch#197582, computed-vs-literal duplicate INDEX tensors in
index-assignment). Here the assignment TARGET is corrupted in place
with genuinely wrong values -- there is no separate "output" to patch
after the call, because ``x`` itself is silently wrong.

As of this run (2026-09-20), upstream issue #197829 is OPEN with no
merged fix. This guard is a userspace workaround for the interim:
``safe_slice_shift(x, shift=1, dim=0)`` performs the equivalent
row-shift by forcing the assignment to run OUTSIDE any enclosing
``torch.compile`` region via ``torch.compiler.disable``, so Inductor
never gets a chance to reinplace it incorrectly -- restoring eager's
correct semantics even when called from inside a compiled function.
"""
from __future__ import annotations

import dataclasses
from typing import Any, Dict, List, Sequence, Tuple


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


def _unsafe_slice_shift(torch_module, x, shift: int = 1, dim: int = 0):
    """The exact buggy pattern from pytorch/pytorch#197829: an in-place
    row-shift written as a slice assignment from an overlapping,
    explicitly-cloned source. Correct in eager mode; silently wrong
    under torch.compile(backend="inductor") per the upstream root
    cause. Kept as a private helper so `diagnose()` can reproduce the
    bug on demand -- never call this directly inside compiled code."""
    if dim != 0:
        raise NotImplementedError("this guard only covers dim=0, matching the upstream issue's repro")
    n = x.shape[0]
    if shift <= 0 or shift >= n:
        raise ValueError(f"shift must be in [1, {n - 1}] for a tensor with {n} rows")
    dst = x[shift:]
    src = x[: n - shift].clone()
    dst.copy_(src)
    return x


def safe_slice_shift(x, shift: int = 1, dim: int = 0):
    """Drop-in replacement for the ``x[shift:] = x[:-shift].clone()``
    pattern that is SAFE to call from inside a ``torch.compile``d
    function: forces the row-shift to execute eagerly via
    ``torch.compiler.disable``, so Inductor's reinplacing pass never
    sees (and cannot mis-optimize) the overlapping-memory assignment.
    Mutates ``x`` in place and returns it, matching the semantics of
    the buggy pattern it replaces. When torch is new enough that the
    upstream bug is already fixed, this is still correct (eager
    execution is always correct) -- just with one extra graph-break
    boundary that costs nothing once #197829 is fixed and this guard
    is retired.
    """
    torch_module = _import_torch()

    disable = getattr(torch_module, "compiler", None)
    disable_fn = getattr(disable, "disable", None) if disable is not None else None

    def _do(x, shift, dim):
        return _unsafe_slice_shift(torch_module, x, shift=shift, dim=dim)

    if disable_fn is not None:
        _do = disable_fn(_do)

    return _do(x, shift, dim)


@dataclasses.dataclass
class SliceShiftCase:
    shape: List[int]
    shift: int
    eager_correct: bool
    compiled_mismatches: int
    compiled_total: int
    compiled_matches_eager: bool
    guarded_matches_eager: bool


def _reference_shift(torch_module, x, shift: int, dim: int):
    """Independent oracle: build the expected shifted tensor without
    any in-place overlapping-memory assignment at all (allocate a
    fresh tensor and index-copy row by row), so the oracle shares no
    code path with either the buggy pattern or the guard."""
    n = x.shape[0]
    out = x.clone()
    for i in range(n - 1, shift - 1, -1):
        out[i] = x[i - shift]
    return out


def _run_case(torch_module, shape: Sequence[int], shift: int) -> SliceShiftCase:
    numel = 1
    for d in shape:
        numel *= d

    def fresh(fn):
        x = torch_module.arange(1, 1 + numel, dtype=torch_module.float64).reshape(shape).clone()
        expected = _reference_shift(torch_module, x, shift, 0)
        fn(x, shift, 0)
        return x, expected

    torch_module._dynamo.reset()
    eager_x, eager_expected = fresh(lambda x, s, d: _unsafe_slice_shift(torch_module, x, s, d))
    eager_correct = bool(torch_module.equal(eager_x, eager_expected))

    torch_module._dynamo.reset()
    compiled_fn = torch_module.compile(
        lambda x, s, d: _unsafe_slice_shift(torch_module, x, s, d), fullgraph=True
    )
    compiled_x, compiled_expected = fresh(compiled_fn)
    compiled_mismatches = int((compiled_x != compiled_expected).sum().item())
    compiled_matches_eager = compiled_mismatches == 0

    torch_module._dynamo.reset()

    def guarded_call(x, s, d):
        def inner(x, s, d):
            return safe_slice_shift(x, shift=s, dim=d)

        compiled_wrapper = torch_module.compile(inner, fullgraph=False)
        return compiled_wrapper(x, s, d)

    guarded_x, guarded_expected = fresh(guarded_call)
    guarded_matches_eager = bool(torch_module.equal(guarded_x, guarded_expected))

    return SliceShiftCase(
        shape=list(shape),
        shift=shift,
        eager_correct=eager_correct,
        compiled_mismatches=compiled_mismatches,
        compiled_total=numel,
        compiled_matches_eager=compiled_matches_eager,
        guarded_matches_eager=guarded_matches_eager,
    )


def diagnose(
    cases: Sequence[Tuple[Sequence[int], int]] = (
        ((3, 16, 16), 1),
        ((8, 4), 1),
        ((1024, 2048), 1),
    ),
) -> Dict[str, Any]:
    """Reproduce the in-place slice-shift aliasing divergence from
    scratch against the currently installed torch build, for every
    case, and verify safe_slice_shift() restores correctness. Never
    trusts a cached/prior result -- every call re-runs the actual
    repro, resetting dynamo before each differently-configured compile
    (see this fleet's operating-prompt v21 note on torch.compile's
    cache not being keyed on all config)."""
    torch_module = _import_torch()
    results = [_run_case(torch_module, shape, shift) for (shape, shift) in cases]

    any_native_bug = any((not c.compiled_matches_eager) and c.eager_correct for c in results)
    guard_fully_correct = all(c.guarded_matches_eager for c in results)

    return {
        "torch_version": torch_module.__version__,
        "issue_url": "https://github.com/pytorch/pytorch/issues/197829",
        "cases": [dataclasses.asdict(c) for c in results],
        "any_native_bug": any_native_bug,
        "guard_fully_correct": guard_fully_correct,
    }
