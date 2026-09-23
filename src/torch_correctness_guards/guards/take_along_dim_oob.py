"""torch-take-along-dim-oob-guard core: detect and guard a real
``torch.take_along_dim(input, indices, dim=<int>)`` correctness bug
where an out-of-bounds index is silently wrapped modulo the dimension
size instead of raising, producing a plausible-looking wrong result
with no error or warning.

Upstream reference: pytorch/pytorch#196106 ("take_along_dim(...,
dim=...) silently wraps out-of-bounds indices instead of raising"),
opened 2026-09-05, status as of this guard's creation (2026-09-19):
open, triaged (labels: triaged, module: advanced indexing, module:
correctness (silent), release triage), one contributor comment
expressing intent to work on it but no merged fix and no open PR
referencing take_along_dim in pytorch/pytorch as of this repo's
creation date -- independently re-checked via `gh api` and
`gh search prs`, never trusted from a cached issue summary alone.

The bug, reproduced from scratch on this host (torch 2.14.0, macOS
arm64 CPU; see README for the exact commands): every sibling indexing
operation (``gather``, ``take``, plain ``tensor[idx]``,
``index_select``) and ``take_along_dim``'s own ``dim=None`` (flattened)
path correctly raise ``IndexError``/``RuntimeError`` on an out-of-range
index. Only the ``dim=<int>`` path silently computes
``input[idx % size]`` instead. This affects large positive indices,
indices past ``-size`` on the negative side, and generalizes to any
number of dimensions. Valid negative indices (``-size .. -1``) are
unaffected and must continue to work exactly as today -- the guard
below is careful not to over-restrict that already-correct path.

This module's guard, ``safe_take_along_dim``, range-checks
``indices`` against the real size of ``input`` along ``dim`` *before*
delegating to the native (buggy) op, raising ``IndexError`` with a
message in the same style as ``gather``/``index_select`` when (and
only when) an index is genuinely out of the valid
``[-size, size - 1]`` range. When ``dim is None`` it delegates straight
to the native op, which already raises correctly on that path (no
double-guarding of an already-correct code path).
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


def make_safe_take_along_dim(torch_module):
    """Build a guard function, bound to a specific torch module, that
    range-checks ``indices`` before calling the native (buggy)
    ``torch.take_along_dim`` when ``dim`` is not ``None``. Returns a
    callable ``(input, indices, dim=None) -> Tensor`` matching
    ``torch.take_along_dim``'s own signature and (correct) semantics
    for every case, including the already-correct ``dim=None`` path,
    which is delegated to the native op unchanged."""

    def _safe_take_along_dim(input, indices, dim=None):
        if dim is None:
            # Already correct upstream: delegate unchanged.
            return torch_module.take_along_dim(input, indices, dim=None)

        ndim = input.dim()
        real_dim = dim if dim >= 0 else dim + ndim
        size = input.shape[real_dim]

        if indices.numel() > 0:
            min_idx = int(indices.min().item())
            max_idx = int(indices.max().item())
            if max_idx >= size or min_idx < -size:
                raise IndexError(
                    f"take_along_dim(): index out of bounds for dimension "
                    f"{real_dim} with size {size} (valid range "
                    f"[{-size}, {size - 1}]); got index range "
                    f"[{min_idx}, {max_idx}]"
                )
        return torch_module.take_along_dim(input, indices, dim=dim)

    return _safe_take_along_dim


@dataclasses.dataclass
class TakeAlongDimCase:
    description: str
    dim: Optional[int]
    expected_valid: bool
    native_raised: bool
    native_result: Optional[List[float]]
    guard_raised: bool
    guard_result: Optional[List[float]]
    oracle_result: Optional[List[float]]
    native_silently_wrong: bool
    guard_correct: bool


def _to_list(tensor) -> List[float]:
    return tensor.detach().to(dtype=tensor.dtype).flatten().tolist()


def _run_case(
    torch_module,
    safe_fn,
    description: str,
    values,
    idx_values,
    dim: Optional[int],
    expected_valid: bool,
) -> TakeAlongDimCase:
    x = torch_module.tensor(values, dtype=torch_module.float64)
    if isinstance(idx_values[0], list):
        idx = torch_module.tensor(idx_values, dtype=torch_module.int64)
    else:
        idx = torch_module.tensor(idx_values, dtype=torch_module.int64)

    native_raised = False
    native_result = None
    try:
        native_out = torch_module.take_along_dim(x, idx, dim=dim)
        native_result = _to_list(native_out)
    except (IndexError, RuntimeError):
        native_raised = True

    guard_raised = False
    guard_result = None
    try:
        guard_out = safe_fn(x, idx, dim=dim)
        guard_result = _to_list(guard_out)
    except (IndexError, RuntimeError):
        guard_raised = True

    oracle_result = None
    if expected_valid:
        import numpy as np

        x_np = x.numpy()
        idx_np = idx.numpy()
        # Every expected_valid case in this module's own case list uses an
        # explicit dim (0 or 1); dim=None cases are only ever OOB (used to
        # confirm that already-correct path keeps raising). If a future
        # case list adds a valid dim=None input, extend the oracle here.
        real_dim = dim if dim >= 0 else dim + x_np.ndim
        oracle = np.take_along_axis(x_np, idx_np, axis=real_dim)
        oracle_result = oracle.flatten().tolist()

    native_silently_wrong = (not expected_valid) and (not native_raised)

    if expected_valid:
        guard_correct = (not guard_raised) and (guard_result == oracle_result)
    else:
        guard_correct = guard_raised

    return TakeAlongDimCase(
        description=description,
        dim=dim,
        expected_valid=expected_valid,
        native_raised=native_raised,
        native_result=native_result,
        guard_raised=guard_raised,
        guard_result=guard_result,
        oracle_result=oracle_result,
        native_silently_wrong=native_silently_wrong,
        guard_correct=guard_correct,
    )


def diagnose() -> Dict[str, Any]:
    """Reproduce the take_along_dim(dim=...) out-of-bounds silent-wrap
    bug from scratch against the currently installed torch build, and
    verify ``make_safe_take_along_dim``'s guard against a NumPy oracle
    (for valid cases) and against the raise-on-OOB contract shared by
    every sibling indexing op (for invalid cases). Never trusts a
    cached/prior result -- every call re-runs the actual repro."""
    torch_module = _import_torch()
    safe_fn = make_safe_take_along_dim(torch_module)

    cases_spec = [
        ("1D OOB positive index just past size (idx=9, size=3)", [10.0, 20.0, 30.0], [9], 0, False),
        ("1D OOB large positive index (idx=100, size=3)", [10.0, 20.0, 30.0], [100], 0, False),
        ("1D OOB huge positive index (idx=2**40, size=3)", [10.0, 20.0, 30.0], [2 ** 40], 0, False),
        ("1D OOB negative index past -size (idx=-4, size=3)", [10.0, 20.0, 30.0], [-4], 0, False),
        ("1D valid negative index (idx=-1, size=3)", [10.0, 20.0, 30.0], [-1], 0, True),
        ("1D valid positive index (idx=1, size=3)", [10.0, 20.0, 30.0], [1], 0, True),
        ("2D OOB indices along dim=1 (idx=[[5],[7]], size=3)", [[0.0, 1.0, 2.0], [3.0, 4.0, 5.0]], [[5], [7]], 1, False),
        ("2D valid indices along dim=1 (idx=[[0],[2]], size=3)", [[0.0, 1.0, 2.0], [3.0, 4.0, 5.0]], [[0], [2]], 1, True),
        ("1D OOB index, dim=None flattened path (already correct upstream)", [10.0, 20.0, 30.0], [9], None, False),
    ]

    cases: List[TakeAlongDimCase] = [
        _run_case(torch_module, safe_fn, desc, vals, idxs, dim, valid)
        for desc, vals, idxs, dim, valid in cases_spec
    ]

    any_native_silently_wrong = any(c.native_silently_wrong for c in cases)
    guard_fully_correct = all(c.guard_correct for c in cases)

    return {
        "torch_version": torch_module.__version__,
        "issue_url": "https://github.com/pytorch/pytorch/issues/196106",
        "cases": [dataclasses.asdict(c) for c in cases],
        "any_native_silently_wrong": any_native_silently_wrong,
        "guard_fully_correct": guard_fully_correct,
    }


# Public convenience wrapper: resolves torch lazily so importing this
# module without torch installed doesn't crash (matching the sibling
# guard repos' degradation pattern).
def safe_take_along_dim(input, indices, dim=None):
    """Module-level convenience wrapper around
    ``make_safe_take_along_dim``: resolves torch on first call. See
    that function's docstring for the full rationale and semantics."""
    torch_module = _import_torch()
    return make_safe_take_along_dim(torch_module)(input, indices, dim=dim)
