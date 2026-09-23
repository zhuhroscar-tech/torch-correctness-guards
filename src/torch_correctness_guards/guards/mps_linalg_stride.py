"""torch-mps-linalg-stride-guard core: detect and guard a real output-
layout (stride) correctness bug on the MPS (Apple Silicon GPU) backend,
where ``torch.linalg.solve_triangular``, ``torch.cholesky_solve`` and
``torch.linalg.solve`` return a ROW-major result on MPS while CPU and
CUDA return COLUMN-major (F-contiguous) for the identical mathematical
inputs. The numeric *values* agree (to ~1e-6/1e-8); only the memory
layout (``.stride()``) differs.

Upstream reference: pytorch/pytorch#197236 ("[MPS] linalg solves
return row-major where CPU and CUDA return column-major"), status as
of this guard's creation (2026-09-21): OPEN, re-checked live via
``gh api repos/pytorch/pytorch/issues/197236`` (never trusted from a
cached issue summary alone).

Root cause per the issue's own diagnosis (three layers, not guessed):
1. MPS's native kernels (``linalg_solve_triangular_mps_impl`` etc. in
   ``aten/src/ATen/native/mps/operations/LinearAlgebra.mm``) write
   through an ``MPSMatrixDescriptor`` whose ``rowBytes`` assumes
   row-major, ignoring the output tensor's requested strides.
2. The C++ meta function (``TORCH_META_FUNC(_linalg_solve_ex)`` in
   ``BatchLinearAlgebra.cpp``) explicitly special-cases MPS to predict
   row-major strides, while the Python meta used by ``torch.compile``
   (``torch/_meta_registrations.py``) unconditionally predicts
   F-contiguous (column-major) -- the two metas DISAGREE with each
   other, and Inductor trusts the Python one, causing a hard
   ``AssertionError`` under ``torch.compile`` on MPS (not merely an
   eager-mode layout surprise).
3. Every *other* linalg op's meta (``cholesky_ex``, ``lu_factor_ex``,
   ``lu_solve``, ``inv_ex``) and eager probes of ``cholesky``/
   ``lu_solve`` agree on F-contiguous regardless of device -- so MPS's
   row-major solve output is the odd one out, not an intentional
   device-specific contract.

Real-world impact: any Apple-Silicon workflow that calls one of these
three ops on MPS and then relies on the CPU/CUDA-standard column-major
layout (e.g. a subsequent ``.view()``/``.reshape()``, a raw-buffer
handoff to another library, or simply feeding the result into
``torch.compile``) gets a silently different memory layout on MPS --
or, under compile, a hard crash -- with no error in plain eager mode
to signal the discrepancy.

This module's guard, ``safe_solve_triangular`` / ``safe_cholesky_solve``
/ ``safe_solve``, works around the defect by calling the native
(value-correct) op unchanged and then normalizing the OUTPUT layout to
column-major (F-contiguous) via a transpose-contiguous-transpose
round trip whenever the op ran on MPS and produced a differently
laid-out result -- matching the CPU/CUDA convention exactly, without
altering any value.

Environment note: GitHub Actions macOS runners report
``torch.backends.mps.is_available() == True`` but the MPS device is
virtualized (see actions/runner-images#9918) -- the same documented
CI limitation already handled by this fleet's other torch-mps-*-guard
repos. This module goes further than a plain allocation probe: a live
CI run (2026-09-21) showed the virtualized device pass a simple
tensor-multiply probe yet still fail ``torch.linalg.cholesky``/matmul
with a Metal *compiler* error ("unsupported deferred-static-alloca-
size ... cooperative_tensor"), because two of this module's three
guarded ops depend on exactly that matmul/cooperative-tensor kernel
class. ``mps_is_functional`` therefore exercises a cholesky-based
probe (not just elementwise ops) and reports honestly as
"mps_functional: false" when it fails, rather than crashing mid-suite
or silently skipping.
"""
from __future__ import annotations

import dataclasses
from typing import Any, Callable, Dict, List, Optional


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


def mps_is_functional(torch_module) -> bool:
    """Return True only if MPS is both reported available AND can
    actually allocate and compute on a real tensor -- distinguishing a
    genuine Apple Silicon GPU from a virtualized CI runner that
    reports availability but cannot allocate (see module docstring).
    Never trusts ``torch.backends.mps.is_available()`` alone.

    A plain elementwise probe is NOT sufficient for this module: live
    GitHub Actions macos-latest CI runs (2026-09-21) showed a
    virtualized MPS device that passes a simple tensor-multiply probe
    but then fails ``torch.linalg.cholesky``/matmul with a Metal
    *compiler* error (``Failed to created pipeline state object ...
    unsupported deferred-static-alloca-size ... cooperative_tensor``)
    -- a distinct failure mode from the plain allocation failure this
    probe originally targeted. Since two of this module's three
    guarded ops go through exactly that matmul/cooperative-tensor
    kernel class, the probe exercises it directly so CI honestly
    reports "not functional" instead of crashing mid-suite.
    """
    if not (
        hasattr(torch_module.backends, "mps")
        and torch_module.backends.mps.is_available()
    ):
        return False
    try:
        probe = torch_module.tensor([1.0, 2.0], device="mps")
        (probe * 2).sum().item()
        mat_probe = torch_module.randn(3, 3, device="mps")
        mat_probe = mat_probe @ mat_probe.T + torch_module.eye(3, device="mps")
        torch_module.linalg.cholesky(mat_probe)
        return True
    except RuntimeError:
        return False


def _is_column_major_2d(t) -> bool:
    """A 2D tensor is column-major (F-contiguous) iff stride == (1, n_rows)."""
    strides = t.stride()
    return len(strides) == 2 and strides[0] == 1 and strides[1] == t.shape[0]


def _normalize_to_column_major(t):
    """Return a tensor with the SAME values as ``t`` but laid out
    column-major (F-contiguous), matching the CPU/CUDA convention for
    these ops. Uses a transpose-contiguous-transpose round trip, which
    only rearranges memory layout and never touches values -- verified
    by every test in this module against a CPU oracle."""
    return t.transpose(-1, -2).contiguous().transpose(-1, -2)


def _run_guarded(native_fn: Callable, *args, **kwargs):
    """Call ``native_fn`` and, if the result lives on MPS and is not
    already column-major, normalize its layout. Every other case
    (non-MPS device, or already column-major) delegates unchanged --
    no double-guarding of an already-correct path."""
    result = native_fn(*args, **kwargs)
    is_mps = getattr(result, "device", None) is not None and result.device.type == "mps"
    if is_mps and not _is_column_major_2d(result):
        return _normalize_to_column_major(result)
    return result


def safe_solve_triangular(A, B, *, upper, **kwargs):
    """Guarded replacement for ``torch.linalg.solve_triangular``: same
    values, but always column-major layout regardless of device."""
    torch_module = _import_torch()
    return _run_guarded(torch_module.linalg.solve_triangular, A, B, upper=upper, **kwargs)


def safe_cholesky_solve(B, L, **kwargs):
    """Guarded replacement for ``torch.cholesky_solve``: same values,
    but always column-major layout regardless of device."""
    torch_module = _import_torch()
    return _run_guarded(torch_module.cholesky_solve, B, L, **kwargs)


def safe_solve(A, B, **kwargs):
    """Guarded replacement for ``torch.linalg.solve``: same values,
    but always column-major layout regardless of device."""
    torch_module = _import_torch()
    return _run_guarded(torch_module.linalg.solve, A, B, **kwargs)


@dataclasses.dataclass
class OpCase:
    op_name: str
    ran: bool
    skip_reason: Optional[str]
    cpu_stride: Optional[List[int]]
    native_mps_stride: Optional[List[int]]
    guard_mps_stride: Optional[List[int]]
    native_layout_matches_cpu: Optional[bool]
    guard_layout_matches_cpu: Optional[bool]
    native_values_match_cpu: Optional[bool]
    guard_values_match_cpu: Optional[bool]


def _make_inputs(torch_module, device):
    """Deterministic, well-conditioned inputs shared by all three ops,
    mirroring the upstream issue's own reproduction exactly (seed 0,
    5x5, SPD-via-A@A.T+5I) so this guard's evidence is directly
    comparable to the issue report."""
    torch_module.manual_seed(0)
    A = torch_module.randn(5, 5, dtype=torch_module.float32)
    A = A @ A.T + 5 * torch_module.eye(5)
    B = torch_module.randn(5, 5, dtype=torch_module.float32)
    return A.to(device), B.to(device)


def _run_op_case(torch_module, mps_functional: bool, op_name: str) -> OpCase:
    if not mps_functional:
        return OpCase(
            op_name=op_name,
            ran=False,
            skip_reason=(
                "MPS reported available but failed a real allocation "
                "probe (virtualized/non-functional runner -- see "
                "module docstring); this is an environment "
                "limitation, not a code defect."
                if hasattr(torch_module.backends, "mps") and torch_module.backends.mps.is_available()
                else "MPS not available on this host"
            ),
            cpu_stride=None,
            native_mps_stride=None,
            guard_mps_stride=None,
            native_layout_matches_cpu=None,
            guard_layout_matches_cpu=None,
            native_values_match_cpu=None,
            guard_values_match_cpu=None,
        )

    A_cpu, B_cpu = _make_inputs(torch_module, "cpu")
    A_mps, B_mps = _make_inputs(torch_module, "mps")

    if op_name == "solve_triangular":
        A_cpu_op = torch_module.triu(A_cpu)
        A_mps_op = torch_module.triu(A_mps)
        native_fn = lambda a, b: torch_module.linalg.solve_triangular(a, b, upper=True)
        guard_fn = lambda a, b: safe_solve_triangular(a, b, upper=True)
        cpu_result = native_fn(A_cpu_op, B_cpu)
        native_mps_result = native_fn(A_mps_op, B_mps)
        guard_mps_result = guard_fn(A_mps_op, B_mps)
    elif op_name == "cholesky_solve":
        L_cpu = torch_module.linalg.cholesky(A_cpu)
        L_mps = torch_module.linalg.cholesky(A_mps)
        native_fn = lambda l, b: torch_module.cholesky_solve(b, l)
        guard_fn = lambda l, b: safe_cholesky_solve(b, l)
        cpu_result = native_fn(L_cpu, B_cpu)
        native_mps_result = native_fn(L_mps, B_mps)
        guard_mps_result = guard_fn(L_mps, B_mps)
    elif op_name == "solve":
        native_fn = torch_module.linalg.solve
        guard_fn = safe_solve
        cpu_result = native_fn(A_cpu, B_cpu)
        native_mps_result = native_fn(A_mps, B_mps)
        guard_mps_result = guard_fn(A_mps, B_mps)
    else:  # pragma: no cover - defensive, all call sites use the three names above
        raise ValueError(f"unknown op_name: {op_name}")

    def _values_match(a, b) -> bool:
        return bool(torch_module.allclose(a.cpu(), b.cpu(), atol=1e-5, rtol=1e-5))

    return OpCase(
        op_name=op_name,
        ran=True,
        skip_reason=None,
        cpu_stride=list(cpu_result.stride()),
        native_mps_stride=list(native_mps_result.stride()),
        guard_mps_stride=list(guard_mps_result.stride()),
        native_layout_matches_cpu=native_mps_result.stride() == cpu_result.stride(),
        guard_layout_matches_cpu=guard_mps_result.stride() == cpu_result.stride(),
        native_values_match_cpu=_values_match(native_mps_result, cpu_result),
        guard_values_match_cpu=_values_match(guard_mps_result, cpu_result),
    )


def diagnose() -> Dict[str, Any]:
    """Reproduce the MPS linalg-solve row-major-vs-column-major layout
    bug from scratch against the currently installed torch build, for
    all three affected ops, and verify the ``safe_*`` guards produce
    a CPU-layout-matching result on every exercised device. Never
    trusts a cached or previously-reported result -- every call
    re-runs the actual repro.
    """
    torch_module = _import_torch()
    mps_functional = mps_is_functional(torch_module)

    cases: List[OpCase] = [
        _run_op_case(torch_module, mps_functional, "solve_triangular"),
        _run_op_case(torch_module, mps_functional, "cholesky_solve"),
        _run_op_case(torch_module, mps_functional, "solve"),
    ]

    any_native_layout_mismatch = any(
        c.ran and c.native_layout_matches_cpu is False for c in cases
    )
    guard_fully_correct = all(
        c.guard_layout_matches_cpu and c.guard_values_match_cpu
        for c in cases
        if c.ran
    )

    return {
        "torch_version": torch_module.__version__,
        "issue_url": "https://github.com/pytorch/pytorch/issues/197236",
        "mps_functional": mps_functional,
        "cases": [dataclasses.asdict(c) for c in cases],
        "any_native_layout_mismatch": any_native_layout_mismatch,
        "guard_fully_correct": guard_fully_correct,
    }
