"""torch-multioutput-alias-guard core: detect and fix a real PyTorch
correctness bug CLASS affecting multi-output operations whose ``out=``
parameter accepts a tuple of tensors: when the SAME tensor object is
passed for two different slots of that tuple, the op silently computes
and writes a WRONG result into the shared tensor instead of raising --
because the op's meta function validates each output's dtype/shape
independently but never calls ``at::assert_no_overlap`` the way
``kthvalue``/``cross`` already do for the identical aliasing hazard.

Confirmed on two ops so far, independently reproduced on this host:

1. ``torch.aminmax(x, out=(t, t))`` -- upstream issue
   pytorch/pytorch#195338 ("torch.aminmax silently returns wrong values
   when out= tensors alias"), labeled "module: correctness (silent)" by
   the PyTorch team. The two writes race for the same memory location;
   whichever value (min or max) is written last silently overwrites the
   other, so the aliased call can even return an arithmetically
   impossible min > max pair with zero warning.

2. ``torch.linalg.slogdet(x, out=(sign, logabsdet))`` when ``sign`` and
   ``logabsdet`` are the SAME tensor -- newly identified during this
   candidate's evidence-gathering (not itself a separate filed upstream
   issue at time of writing; discovered by testing the LAPACK-backed
   op-class question the #195338 issue author explicitly could not
   answer on their own non-LAPACK build). Reproduced on this host
   (torch 2.14.0, Accelerate/LAPACK build): a reference call gives
   sign=1.0, logabsdet=2.0794...; the aliased call silently returns
   2.0794... in the shared tensor, losing the sign value entirely.

Both are the SAME underlying defect class (missing
``assert_no_overlap`` in a multi-output op's meta function), not two
unrelated bugs -- this module guards both with one shared aliasing
check rather than shipping two near-duplicate single-op tools.

This module's guard functions (``safe_aminmax``, ``safe_slogdet``)
validate that every pair of output tensor arguments does not alias
(via ``Tensor.data_ptr()`` identity, the same check PyTorch's own
``assert_no_overlap`` uses as its fast path) BEFORE calling the
underlying op, and raise a clear ``RuntimeError`` matching the style of
PyTorch's own overlap error message when they do. When no output
tensors are passed (the common case: ``out=None``, letting the op
allocate its own outputs), the guard is a zero-cost passthrough --
aliasing can only happen when the CALLER explicitly passes tensors,
so there is nothing to check.
"""
from __future__ import annotations

import dataclasses
from typing import Any, Dict, List, Optional, Sequence, Tuple


class TorchUnavailableError(RuntimeError):
    """Raised when torch cannot be imported. Kept as a distinct type so
    callers can distinguish "torch isn't installed" from an actual
    diagnostic failure."""


class AliasedMultiOutputError(RuntimeError):
    """Raised by the guard functions when two output tensor arguments
    of a multi-output op alias each other. Message mirrors PyTorch's
    own ``assert_no_overlap`` wording so the failure reads consistently
    with a native PyTorch error rather than a foreign-looking one."""


def _import_torch():
    try:
        import torch  # noqa: F401
        import torch.linalg  # noqa: F401
    except Exception as exc:  # pragma: no cover - exercised only without torch
        raise TorchUnavailableError(
            "torch is required for diagnosis and guarding; install the "
            "'torch' extra."
        ) from exc
    return torch


def _tensors_alias(a, b) -> bool:
    """True when two tensors share the same underlying storage location
    (same data_ptr) -- the same fast-path check PyTorch's own C++
    ``assert_no_overlap`` performs before falling back to a fuller
    overlap analysis. Sufficient for the same-tensor-object case this
    guard targets (the exact shape of both #195338 and the slogdet
    finding: literally the SAME tensor object passed twice)."""
    if a is None or b is None:
        return False
    return a.data_ptr() == b.data_ptr()


def _assert_no_output_overlap(out: Optional[Tuple[Any, Any]], op_name: str) -> None:
    if out is None:
        return
    a, b = out
    if _tensors_alias(a, b):
        raise AliasedMultiOutputError(
            f"unsupported operation: {op_name}'s two output tensors "
            "refer to a single memory location. Please pass two "
            "distinct tensors (or leave out=None to let the op "
            "allocate its own outputs)."
        )


def _make_safe_aminmax(torch_module):
    def safe_aminmax(input, *, dim=None, keepdim=False, out=None):
        """Drop-in guard for ``torch.aminmax``: raises
        ``AliasedMultiOutputError`` BEFORE computing anything if ``out``
        passes the same tensor for both min and max, instead of
        silently computing and writing a wrong (possibly min>max)
        result (pytorch/pytorch#195338). Matches
        ``torch.aminmax``'s normal behavior in every other case."""
        _assert_no_output_overlap(out, "aminmax")
        if out is None:
            return torch_module.aminmax(input, dim=dim, keepdim=keepdim)
        return torch_module.aminmax(input, dim=dim, keepdim=keepdim, out=out)

    return safe_aminmax


def _make_safe_slogdet(torch_module):
    def safe_slogdet(input, *, out=None):
        """Drop-in guard for ``torch.linalg.slogdet``: raises
        ``AliasedMultiOutputError`` BEFORE computing anything if ``out``
        passes the same tensor for both sign and logabsdet, instead of
        silently losing the sign value (this candidate's own
        LAPACK-backed finding, same defect class as #195338). Matches
        ``torch.linalg.slogdet``'s normal behavior in every other
        case."""
        _assert_no_output_overlap(out, "linalg.slogdet")
        if out is None:
            return torch_module.linalg.slogdet(input)
        return torch_module.linalg.slogdet(input, out=out)

    return safe_slogdet


@dataclasses.dataclass
class AminmaxAliasCase:
    values: List[float]
    native_aliased_result: Optional[float]
    native_aliased_raised: bool
    reference_min: float
    reference_max: float
    native_result_is_wrong: bool  # native aliased result != reference min or max coherently
    guard_raised: bool


@dataclasses.dataclass
class SlogdetAliasCase:
    matrix: List[List[float]]
    native_aliased_result: Optional[float]
    native_aliased_raised: bool
    reference_sign: float
    reference_logabsdet: float
    native_result_lost_sign: bool
    guard_raised: bool


def _run_aminmax_case(torch_module, safe_aminmax, values: Sequence[float]) -> AminmaxAliasCase:
    x = torch_module.tensor(values, dtype=torch_module.float64)

    mn = torch_module.empty((), dtype=torch_module.float64)
    mx = torch_module.empty((), dtype=torch_module.float64)
    torch_module.aminmax(x, out=(mn, mx))
    ref_min, ref_max = mn.item(), mx.item()

    native_aliased_result: Optional[float] = None
    native_aliased_raised = False
    try:
        same = torch_module.empty((), dtype=torch_module.float64)
        torch_module.aminmax(x, out=(same, same))
        native_aliased_result = same.item()
    except RuntimeError:
        native_aliased_raised = True

    guard_raised = False
    try:
        gsame = torch_module.empty((), dtype=torch_module.float64)
        safe_aminmax(x, out=(gsame, gsame))
    except AliasedMultiOutputError:
        guard_raised = True

    native_result_is_wrong = (
        not native_aliased_raised
        and native_aliased_result is not None
        and not (
            abs(native_aliased_result - ref_min) < 1e-12
            and abs(native_aliased_result - ref_max) < 1e-12
        )
    )

    return AminmaxAliasCase(
        values=list(values),
        native_aliased_result=native_aliased_result,
        native_aliased_raised=native_aliased_raised,
        reference_min=ref_min,
        reference_max=ref_max,
        native_result_is_wrong=native_result_is_wrong,
        guard_raised=guard_raised,
    )


def _run_slogdet_case(torch_module, safe_slogdet, matrix: Sequence[Sequence[float]]) -> SlogdetAliasCase:
    B = torch_module.tensor(matrix, dtype=torch_module.float64)
    ref_sign, ref_logabsdet = torch_module.linalg.slogdet(B)
    ref_sign, ref_logabsdet = ref_sign.item(), ref_logabsdet.item()

    native_aliased_result: Optional[float] = None
    native_aliased_raised = False
    try:
        same = torch_module.empty((), dtype=torch_module.float64)
        torch_module.linalg.slogdet(B, out=(same, same))
        native_aliased_result = same.item()
    except RuntimeError:
        native_aliased_raised = True

    guard_raised = False
    try:
        gsame = torch_module.empty((), dtype=torch_module.float64)
        safe_slogdet(B, out=(gsame, gsame))
    except AliasedMultiOutputError:
        guard_raised = True

    # The bug: the shared tensor ends up holding ONLY the logabsdet value
    # (whichever write landed last), so reading it back as "sign" gives a
    # value that is not a valid sign (not +-1, and not what a correct
    # sign readout should be) whenever logabsdet != the true sign. This
    # directly demonstrates the sign value was silently lost/overwritten,
    # without relying on a fragile combined-value formula.
    native_result_lost_sign = (
        not native_aliased_raised
        and native_aliased_result is not None
        and abs(native_aliased_result - ref_logabsdet) < 1e-9
        and abs(native_aliased_result - ref_sign) > 1e-6
    )

    return SlogdetAliasCase(
        matrix=[list(row) for row in matrix],
        native_aliased_result=native_aliased_result,
        native_aliased_raised=native_aliased_raised,
        reference_sign=ref_sign,
        reference_logabsdet=ref_logabsdet,
        native_result_lost_sign=native_result_lost_sign,
        guard_raised=guard_raised,
    )


def diagnose(
    aminmax_cases: Sequence[Sequence[float]] = (
        (3.0, 1.0, 4.0, 1.0, 5.0, 9.0, 2.0, 6.0),
        (-7.0, 7.0, 0.0),
        (2.0, 2.0, 2.0),
    ),
    slogdet_cases: Sequence[Sequence[Sequence[float]]] = (
        ((2.0, 0.0, 0.0), (0.0, 2.0, 0.0), (0.0, 0.0, 2.0)),
        ((1.0, 2.0), (3.0, 4.0)),
    ),
) -> Dict[str, Any]:
    """Reproduce the aminmax/slogdet same-tensor output-aliasing
    divergence from scratch against the currently installed torch
    build, for every case, and verify the guard functions raise before
    any wrong result is computed. Never trusts a cached/prior result --
    every call re-runs the actual repro."""
    torch_module = _import_torch()
    safe_aminmax = _make_safe_aminmax(torch_module)
    safe_slogdet = _make_safe_slogdet(torch_module)

    aminmax_results = [_run_aminmax_case(torch_module, safe_aminmax, c) for c in aminmax_cases]
    slogdet_results = [_run_slogdet_case(torch_module, safe_slogdet, c) for c in slogdet_cases]

    any_aminmax_bug = any(c.native_result_is_wrong for c in aminmax_results)
    any_slogdet_bug = any(c.native_result_lost_sign for c in slogdet_results)
    guard_fully_correct = all(c.guard_raised for c in aminmax_results) and all(
        c.guard_raised for c in slogdet_results
    )

    return {
        "torch_version": torch_module.__version__,
        "issue_urls": [
            "https://github.com/pytorch/pytorch/issues/195338",
        ],
        "aminmax_cases": [dataclasses.asdict(c) for c in aminmax_results],
        "slogdet_cases": [dataclasses.asdict(c) for c in slogdet_results],
        "any_aminmax_alias_bug": any_aminmax_bug,
        "any_slogdet_alias_bug": any_slogdet_bug,
        "guard_fully_correct": guard_fully_correct,
    }


# Public guard functions bound lazily against the currently installed
# torch build (import-time torch import would break "torch not
# installed" degradation -- see TorchUnavailableError above).
def safe_aminmax(input, *, dim=None, keepdim=False, out=None):
    """Module-level convenience wrapper: resolves torch on first call
    and delegates to the bound guard. See ``_make_safe_aminmax`` for
    the full rationale."""
    torch_module = _import_torch()
    return _make_safe_aminmax(torch_module)(input, dim=dim, keepdim=keepdim, out=out)


def safe_slogdet(input, *, out=None):
    """Module-level convenience wrapper: resolves torch on first call
    and delegates to the bound guard. See ``_make_safe_slogdet`` for
    the full rationale."""
    torch_module = _import_torch()
    return _make_safe_slogdet(torch_module)(input, out=out)
