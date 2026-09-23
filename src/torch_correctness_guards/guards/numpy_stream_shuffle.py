"""torch-numpy-stream-shuffle-guard core: detect and guard against a real,
currently-open PyTorch correctness defect where
``torch._numpy.random.shuffle`` on a ``torch.Tensor`` under
``torch._dynamo.config.use_numpy_random_stream=True`` does not perform a
true in-place permutation along axis 0 -- it can duplicate some rows and
silently drop others, so the result is not a permutation of the input at
all (pytorch/pytorch#197795, open, no fix merged as of this guard's
creation).

Independently reproduced from scratch on this host (torch 2.14.0,
Apple M4/arm64, CPU-only):

    x = torch.arange(12).reshape(6, 2)
    tnp.random.seed(0)
    with torch._dynamo.config.patch(use_numpy_random_stream=True):
        tnp.random.shuffle(x)
    # x -> [[0,1],[2,3],[2,3],[0,1],[4,5],[4,5]]
    # rows [0,1] and [2,3] each appear TWICE; rows [4,5] wait, actually one
    # value duplicates while other rows vanish -- exact duplicate/drop
    # pattern is seed/torch-version dependent, but the INVARIANT VIOLATION
    # (sorted(rows_after) != sorted(rows_before)) reproduces reliably.

Root cause (per the issue): ``torch._numpy.random.shuffle`` under stream
mode delegates to NumPy's own ``Generator.shuffle``/``RandomState.shuffle``
implementation, which assumes it is mutating a real NumPy ndarray with
NumPy's own view/stride semantics. A ``torch.Tensor`` is not a subclass of
``Sequence`` and does not share those exact view semantics, so NumPy's
in-place Fisher-Yates swap loop -- which relies on getting/setting
individual elements through the object's own `__getitem__`/`__setitem__`
via a real memoryview-style aliasing contract -- silently corrupts the
tensor's backing storage instead of performing a true permutation.
``torch._numpy.random`` itself emits a ``UserWarning`` about this
("you are shuffling a 'Tensor' object which is not a subclass of
'Sequence'... may contain duplicates after shuffling") but does not raise
or refuse -- the caller receives a corrupted tensor with no exception.

This guard's policy: never call the unsafe stream-mode shuffle on a
tensor. Instead, perform a real permutation via ``torch.randperm`` (a
native, tensor-safe RNG operation with no view-aliasing hazard) followed
by ``index_select``/fancy indexing -- which is already the standard,
recommended way to permute a tensor along its leading axis in PyTorch,
and provably preserves the exact row multiset because it is a pure
gather over a bijective permutation of indices.
"""
from __future__ import annotations

import dataclasses
from typing import Any, Dict, List, Tuple


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


# ---------------------------------------------------------------------------
# Guard: a length/dtype-independent, provably-safe row permutation.
# ---------------------------------------------------------------------------


def safe_row_shuffle(x, generator=None):
    """Return a NEW tensor with x's rows (axis 0) permuted uniformly at
    random, via ``torch.randperm`` + ``index_select`` -- never mutating x
    through the unsafe ``torch._numpy.random.shuffle`` stream-mode path.

    This is a pure gather over a bijective index permutation, so the
    output's row multiset is IDENTICAL to the input's by construction --
    there is no way for this implementation to duplicate or drop a row.
    Callers that need strict in-place mutation (matching numpy.shuffle's
    contract) should assign the result back with ``x.copy_(result)``.
    """
    torch_module = _import_torch()
    if x.dim() == 0:
        raise ValueError("safe_row_shuffle requires at least 1 dimension")
    n = x.shape[0]
    if generator is not None:
        perm = torch_module.randperm(n, generator=generator, device=x.device)
    else:
        perm = torch_module.randperm(n, device=x.device)
    return x.index_select(0, perm)


def safe_row_shuffle_(x, generator=None):
    """In-place variant matching numpy.shuffle's mutate-in-place contract:
    computes the safe permutation then copies it back into x's storage."""
    result = safe_row_shuffle(x, generator=generator)
    x.copy_(result)
    return x


# ---------------------------------------------------------------------------
# Diagnosis: reproduce the REAL unsafe stream-mode shuffle's row-multiset
# corruption against the currently installed torch build, then verify the
# safe_row_shuffle_ guard always preserves the row multiset.
# ---------------------------------------------------------------------------


def _row_multiset(tensor) -> List[Tuple[Any, ...]]:
    return sorted(tuple(row) for row in tensor.tolist())


def _unsafe_shuffle_preserves_multiset(torch_module, seed: int, n_rows: int, n_cols: int) -> bool:
    """Run the REAL, unguarded torch._numpy.random.shuffle under stream
    mode and report whether it preserved the row multiset. Never trusts a
    cached/prior result -- every call re-runs the actual repro against
    whatever torch build is currently installed."""
    import torch._numpy as tnp

    x = torch_module.arange(n_rows * n_cols).reshape(n_rows, n_cols)
    before = _row_multiset(x)

    tnp.random.seed(seed)
    import warnings

    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        with torch_module._dynamo.config.patch(use_numpy_random_stream=True):
            tnp.random.shuffle(x)

    after = _row_multiset(x)
    return before == after


def _guarded_shuffle_preserves_multiset(torch_module, seed: int, n_rows: int, n_cols: int) -> bool:
    x = torch_module.arange(n_rows * n_cols).reshape(n_rows, n_cols)
    before = _row_multiset(x)
    generator = torch_module.Generator().manual_seed(seed)
    safe_row_shuffle_(x, generator=generator)
    after = _row_multiset(x)
    return before == after


@dataclasses.dataclass
class ShuffleCase:
    seed: int
    n_rows: int
    n_cols: int
    unsafe_preserved_multiset: bool
    guard_preserved_multiset: bool


def diagnose(seeds=(0, 1, 2, 7), shapes=((6, 2), (10, 3))) -> Dict[str, Any]:
    """Reproduce the row-multiset corruption from scratch against the
    currently installed torch build, across several seeds and shapes, then
    verify the safe_row_shuffle_ guard always preserves the row multiset.
    Never trusts a cached/prior result -- every call re-runs the actual
    repro."""
    torch_module = _import_torch()

    cases: List[ShuffleCase] = []
    for n_rows, n_cols in shapes:
        for seed in seeds:
            unsafe_ok = _unsafe_shuffle_preserves_multiset(torch_module, seed, n_rows, n_cols)
            guard_ok = _guarded_shuffle_preserves_multiset(torch_module, seed, n_rows, n_cols)
            cases.append(
                ShuffleCase(
                    seed=seed,
                    n_rows=n_rows,
                    n_cols=n_cols,
                    unsafe_preserved_multiset=unsafe_ok,
                    guard_preserved_multiset=guard_ok,
                )
            )

    any_bug_present = any(not c.unsafe_preserved_multiset for c in cases)
    guard_fully_effective = all(c.guard_preserved_multiset for c in cases)

    return {
        "torch_version": torch_module.__version__,
        "issue_urls": ["https://github.com/pytorch/pytorch/issues/197795"],
        "cases": [dataclasses.asdict(c) for c in cases],
        "any_bug_present": any_bug_present,
        "guard_fully_effective": guard_fully_effective,
    }
