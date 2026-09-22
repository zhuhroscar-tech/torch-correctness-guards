"""torch-mps-copy-dtype-guard core: detect and guard a real
``.to()``/``.copy_()`` correctness bug on the MPS (Apple Silicon GPU)
backend, where transferring an MPS tensor into a CPU ``float64`` or
``complex128`` destination silently writes zeros (or leaves stale
memory untouched) instead of raising or correctly converting.

Upstream reference: pytorch/pytorch#197715 ("[MPS] Copy from MPS
into a CPU float64 or complex128 tensor silently writes nothing
(2.14 regression)"), status as of this guard's creation
(2026-09-20): OPEN, labeled "module: correctness (silent)" and
"module: mps". A related PR (#197722, "[MPS] Raise when a castout
kernel is asked to store a dtype Metal cannot represent") exists
upstream but only makes the failure loud (a raised error), not
correct -- it does not by itself deliver a converted result, and as
of this guard's creation is UNMERGED (state=OPEN, independently
re-checked via ``gh pr view 197722 --repo pytorch/pytorch``, never
trusted from a cached issue summary alone).

The bug, reproduced from scratch on this host (Apple M4, macOS
15.7.7, torch 2.14.0, real MPS device -- not simulated): Metal's
``store_at_offs`` switch (see aten's mps copy/cast kernels) has no
case for ``Double``/``ComplexDouble`` destinations, so a direct
``mps_tensor.to("cpu", torch.float64)`` or
``dst.copy_(mps_tensor)`` where ``dst`` is float64/complex128
silently no-ops the store: a freshly allocated destination reads
back as all-zeros, and a pre-filled destination is left completely
UNCHANGED (not even zeroed) with no error, warning, or exception.
Confirmed via control: the identical transfer to a float32 CPU
destination (not in the affected dtype set) is correct.

Real-world impact: any macOS/Apple-Silicon workflow that moves an
MPS tensor to the CPU for float64-precision post-processing (numpy
interop, scientific reductions, complex-valued signal processing) or
uses ``copy_`` into a pre-allocated float64/complex128 buffer gets
silently wrong (zeroed or stale) data with no signal that anything
went wrong -- a silent data-corruption bug specific to Apple Silicon
GPU-to-CPU transfers on affected torch versions.

This module's guard, ``safe_to``/``safe_copy_``, works around the
defect by never asking Metal's store kernel to write directly into a
float64/complex128 destination: it always performs the device
transfer at the SOURCE tensor's own dtype (which Metal always
handles correctly) and only upcasts to float64/complex128 as a
second, CPU-side step -- exactly the manual two-step sequence this
module's own repro proves is correct, matching a CPU-computed
oracle exactly.

Environment note: GitHub Actions macOS runners report
``torch.backends.mps.is_available() == True`` but the MPS device is
virtualized and cannot actually allocate GPU memory (see
actions/runner-images#9918, pytorch/torchchat#1416) -- a widely
documented CI limitation, not specific to this repo. This module
detects that case explicitly (a real MPS tensor allocation probe,
not just the availability flag) and reports it honestly as
"mps_functional: false" rather than silently skipping or falsely
claiming MPS was exercised.
"""
from __future__ import annotations

import dataclasses
from typing import Any, Dict, List, Optional


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
    Never trusts ``torch.backends.mps.is_available()`` alone."""
    if not (
        hasattr(torch_module.backends, "mps")
        and torch_module.backends.mps.is_available()
    ):
        return False
    try:
        probe = torch_module.tensor([1.0, 2.0], device="mps")
        (probe * 2).sum().item()
        return True
    except RuntimeError:
        return False


# The two destination dtypes Metal's store_at_offs switch cannot
# represent, per the upstream issue and this module's own
# from-scratch reproduction.
_AFFECTED_DTYPES = None  # populated lazily once torch is imported


def _affected_dtypes(torch_module):
    global _AFFECTED_DTYPES
    if _AFFECTED_DTYPES is None:
        _AFFECTED_DTYPES = {torch_module.float64, torch_module.complex128}
    return _AFFECTED_DTYPES


def safe_to(tensor, device, dtype=None):
    """Guarded replacement for ``tensor.to(device, dtype)``.

    When the source tensor lives on MPS and the requested destination
    ``dtype`` is one of the two dtypes Metal's store kernel cannot
    represent (float64, complex128), this performs the transfer at
    the source's own dtype first (always correct on MPS) and only
    upcasts on the CPU side afterward -- avoiding the silent
    zero-fill entirely. Every other combination delegates straight to
    the native ``.to()`` unchanged (no double-guarding of an
    already-correct path).
    """
    torch_module = _import_torch()
    is_mps_source = getattr(tensor, "device", None) is not None and tensor.device.type == "mps"
    target_is_affected = dtype is not None and dtype in _affected_dtypes(torch_module)
    target_is_cpu = device == "cpu" or (hasattr(device, "type") and device.type == "cpu")

    if is_mps_source and target_is_cpu and target_is_affected:
        transferred = tensor.to("cpu")
        return transferred.to(dtype)
    return tensor.to(device, dtype) if dtype is not None else tensor.to(device)


def safe_copy_(dst, src):
    """Guarded replacement for ``dst.copy_(src)``.

    When ``src`` lives on MPS and ``dst``'s dtype is one of the two
    affected dtypes, performs the transfer at the source's own dtype
    into a temporary, then copies the correctly-converted values into
    ``dst`` -- avoiding the silent no-op entirely. Returns ``dst``
    (matching ``Tensor.copy_``'s in-place contract).
    """
    torch_module = _import_torch()
    is_mps_source = getattr(src, "device", None) is not None and src.device.type == "mps"
    dst_is_affected = dst.dtype in _affected_dtypes(torch_module)
    dst_is_cpu = dst.device.type == "cpu"

    if is_mps_source and dst_is_cpu and dst_is_affected:
        transferred = src.to("cpu").to(dst.dtype)
        dst.copy_(transferred)
        return dst
    dst.copy_(src)
    return dst


@dataclasses.dataclass
class DtypeCase:
    dtype_name: str
    ran: bool
    skip_reason: Optional[str]
    native_to_row: Optional[List[float]]
    guard_to_row: Optional[List[float]]
    native_copy_row: Optional[List[float]]
    guard_copy_row: Optional[List[float]]
    oracle_row: Optional[List[float]]
    native_to_matches_oracle: Optional[bool]
    guard_to_matches_oracle: Optional[bool]
    native_copy_matches_oracle: Optional[bool]
    guard_copy_matches_oracle: Optional[bool]


def _as_float_list(t) -> List[float]:
    """Flatten a tensor's values to a plain list of floats for JSON
    reporting. Complex tensors are flattened to alternating
    (real, imag) pairs so the report stays JSON-serializable."""
    if t.is_complex():
        return [x for c in t.detach().cpu().tolist() for x in (c.real, c.imag)]
    return t.detach().cpu().tolist()


def _run_dtype_case(torch_module, mps_functional: bool, dtype, dtype_name: str) -> DtypeCase:
    if not mps_functional:
        return DtypeCase(
            dtype_name=dtype_name,
            ran=False,
            skip_reason=(
                "MPS reported available but failed a real allocation "
                "probe (virtualized/non-functional runner -- see "
                "module docstring); this is an environment "
                "limitation, not a code defect."
                if hasattr(torch_module.backends, "mps") and torch_module.backends.mps.is_available()
                else "MPS not available on this host"
            ),
            native_to_row=None,
            guard_to_row=None,
            native_copy_row=None,
            guard_copy_row=None,
            oracle_row=None,
            native_to_matches_oracle=None,
            guard_to_matches_oracle=None,
            native_copy_matches_oracle=None,
            guard_copy_matches_oracle=None,
        )

    base_dtype = torch_module.float32 if dtype != torch_module.complex128 else torch_module.complex64
    src_cpu_values = torch_module.arange(1, 9, dtype=torch_module.float32) / 3
    if dtype == torch_module.complex128:
        src_cpu = torch_module.complex(src_cpu_values, src_cpu_values)
    else:
        src_cpu = src_cpu_values
    src_mps = src_cpu.to("mps")

    # Oracle: compute entirely on CPU (never touches MPS), matching
    # the documented, CPU-verified-correct semantics.
    oracle = src_cpu.to(dtype)

    # Native (buggy) path: direct .to(cpu, dtype) from MPS.
    native_to_result = src_mps.to("cpu", dtype)

    # Guarded .to() path.
    guard_to_result = safe_to(src_mps, "cpu", dtype)

    # Native (buggy) path: .copy_() into a pre-filled destination.
    fill_value = 9.0 if dtype != torch_module.complex128 else complex(9.0, 9.0)
    native_dst = torch_module.full((8,), fill_value, dtype=dtype)
    native_dst.copy_(src_mps)

    # Guarded .copy_() path.
    guard_dst = torch_module.full((8,), fill_value, dtype=dtype)
    safe_copy_(guard_dst, src_mps)

    def _matches(a, b) -> bool:
        return bool(torch_module.allclose(a, b, atol=1e-6, rtol=1e-6))

    return DtypeCase(
        dtype_name=dtype_name,
        ran=True,
        skip_reason=None,
        native_to_row=_as_float_list(native_to_result),
        guard_to_row=_as_float_list(guard_to_result),
        native_copy_row=_as_float_list(native_dst),
        guard_copy_row=_as_float_list(guard_dst),
        oracle_row=_as_float_list(oracle),
        native_to_matches_oracle=_matches(native_to_result, oracle),
        guard_to_matches_oracle=_matches(guard_to_result, oracle),
        native_copy_matches_oracle=_matches(native_dst, oracle),
        guard_copy_matches_oracle=_matches(guard_dst, oracle),
    )


def diagnose() -> Dict[str, Any]:
    """Reproduce the MPS-to-CPU float64/complex128 silent copy bug
    from scratch against the currently installed torch build, for
    both affected dtypes, and verify ``safe_to``/``safe_copy_``
    against a CPU-only oracle. Never trusts a cached or
    previously-reported result -- every call re-runs the actual
    repro.
    """
    torch_module = _import_torch()
    mps_functional = mps_is_functional(torch_module)

    cases: List[DtypeCase] = [
        _run_dtype_case(torch_module, mps_functional, torch_module.float64, "float64"),
        _run_dtype_case(torch_module, mps_functional, torch_module.complex128, "complex128"),
    ]

    any_native_silently_wrong = any(
        c.ran
        and (c.native_to_matches_oracle is False or c.native_copy_matches_oracle is False)
        for c in cases
    )
    guard_fully_correct = all(
        c.guard_to_matches_oracle and c.guard_copy_matches_oracle
        for c in cases
        if c.ran
    )

    return {
        "torch_version": torch_module.__version__,
        "issue_url": "https://github.com/pytorch/pytorch/issues/197715",
        "mps_functional": mps_functional,
        "cases": [dataclasses.asdict(c) for c in cases],
        "any_native_silently_wrong": any_native_silently_wrong,
        "guard_fully_correct": guard_fully_correct,
    }
