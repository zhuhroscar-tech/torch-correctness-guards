"""torch-expand-fill-guard core: detect and fix a real
``torch.compile(backend="inductor")`` correctness bug where calling
``.fill_(scalar)`` on a broadcast view created by ``Tensor.expand()``
writes wrong values into the underlying base tensor.

Reproduced from scratch on this host (torch 2.14.0, macOS arm64 CPU;
see README for the exact commands)::

    def f(base):
        v = base.expand(3, -1)
        v.fill_(2.0)
        return base

    base = torch.tensor([-1.0, -0.5, 0.0, 0.5])
    f(base.clone())                                    # eager:    [2., 2., 2., 2.]
    torch.compile(f, backend="inductor")(base.clone())  # inductor: [8., 7., 6., 5.]

``expand()`` returns a view with stride 0 on the broadcast dimension:
every "row" of the expanded view aliases the SAME underlying storage
positions. Eager's in-place ``fill_`` correctly collapses this to the
scalar value at every logical position (the repeated writes to the
same physical element are idempotent, so the result is simply the
fill value everywhere). Inductor's lowering for an in-place fill on a
stride-0 (broadcast) view does not account for the aliasing and
instead computes/writes some other value per logical index -- with no
error, warning, or shape/dtype mismatch to signal anything went
wrong. The result is not even predictable garbage: it varies with the
broadcast size and the base tensor's original contents, which is
exactly the shape of failure most likely to be missed in code review
(the output "looks like a tensor" and fails silently downstream).

Upstream reference: pytorch/pytorch#197448 ("[torch.compile] `expand`
+ `fill_` produces incorrect results", open as of this writing).
Distinct from the previously-fixed/closed #175791 (expand +
broadcasted index_put via ``__setitem__``) and #183986 (expand +
index_add/index_copy/index_fill/index_put, CUDA-only in the original
report) -- this guard targets the plain scalar ``.fill_()`` path,
which reproduces on CPU and remains open.

This module's guard function, ``safe_fill_``, forces the actual
in-place fill outside the compiled graph via ``torch.compiler.disable``
(a Dynamo graph break at this single, already-cheap call site), so the
real eager kernel -- which handles the stride-0 aliasing correctly --
runs regardless of whether the caller is itself under
``torch.compile``. This trades a small amount of graph fusion at the
``fill_()`` call site for correctness; the guard function is meant as
a drop-in replacement for ``Tensor.fill_`` at call sites where the
receiver may be an expanded (broadcast) view under compilation.
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


def _make_safe_fill_(torch_module):
    """Build the guard function bound to a specific torch module (so
    the same core logic works against whatever torch build is actually
    installed, without importing torch at module load time)."""

    @torch_module.compiler.disable
    def _eager_fill_(tensor, value):
        return tensor.fill_(value)

    def safe_fill_(tensor, value):
        """Drop-in guard for ``Tensor.fill_``: forces the actual
        in-place fill to run in eager mode via a Dynamo graph break, so
        a broadcast (stride-0) view produced by ``.expand()`` gets the
        same correct collapse-to-scalar semantics eager already
        provides -- matching eager output exactly instead of the
        silently wrong per-element values Inductor's lowering produces
        for this aliasing pattern under ``torch.compile``."""
        return _eager_fill_(tensor, value)

    return safe_fill_


@dataclasses.dataclass
class ExpandFillCase:
    expand_rows: int
    base_values: List[float]
    fill_value: float
    eager_result: List[float]
    compiled_native_result: List[float]
    compiled_guarded_result: List[float]
    native_diverges: bool  # eager != compiled_native (the bug)
    guard_matches_eager: bool


def _run_expand_fill_case(
    torch_module,
    safe_fill_,
    base_values: Sequence[float],
    expand_rows: int,
    fill_value: float,
) -> ExpandFillCase:
    def f_native(base):
        v = base.expand(expand_rows, -1)
        v.fill_(fill_value)
        return base

    def f_guarded(base):
        v = base.expand(expand_rows, -1)
        safe_fill_(v, fill_value)
        return base

    base_eager = torch_module.tensor(list(base_values))
    eager_out = f_native(base_eager.clone())

    torch_module._dynamo.reset()
    compiled_native = torch_module.compile(f_native, backend="inductor")
    base_native = torch_module.tensor(list(base_values))
    native_out = compiled_native(base_native.clone())

    torch_module._dynamo.reset()
    compiled_guarded = torch_module.compile(f_guarded, backend="inductor")
    base_guarded = torch_module.tensor(list(base_values))
    guarded_out = compiled_guarded(base_guarded.clone())

    return ExpandFillCase(
        expand_rows=expand_rows,
        base_values=list(base_values),
        fill_value=fill_value,
        eager_result=[float(x) for x in eager_out.tolist()],
        compiled_native_result=[float(x) for x in native_out.tolist()],
        compiled_guarded_result=[float(x) for x in guarded_out.tolist()],
        native_diverges=not torch_module.equal(eager_out, native_out),
        guard_matches_eager=bool(torch_module.equal(eager_out, guarded_out)),
    )


def diagnose(
    cases: Sequence[Dict[str, Any]] = (
        {"base_values": [-1.0, -0.5, 0.0, 0.5], "expand_rows": 3, "fill_value": 2.0},
        {"base_values": [1.0, 2.0, 3.0], "expand_rows": 2, "fill_value": 0.0},
        {"base_values": [10.0], "expand_rows": 5, "fill_value": -7.5},
        {"base_values": [0.0, 0.0, 0.0, 0.0, 0.0], "expand_rows": 4, "fill_value": 1.0},
    ),
) -> Dict[str, Any]:
    """Reproduce the eager-vs-Inductor ``expand()`` + in-place
    ``fill_()`` divergence from scratch against the currently installed
    torch build, and verify ``safe_fill_`` matches eager in every
    case. Never trusts a cached/prior result -- every call re-runs the
    actual repro.
    """
    torch_module = _import_torch()
    safe_fill_ = _make_safe_fill_(torch_module)

    results: List[ExpandFillCase] = [
        _run_expand_fill_case(
            torch_module,
            safe_fill_,
            case["base_values"],
            case["expand_rows"],
            case["fill_value"],
        )
        for case in cases
    ]

    any_divergence = any(c.native_diverges for c in results)
    guard_fully_correct = all(c.guard_matches_eager for c in results)

    return {
        "torch_version": torch_module.__version__,
        "issue_urls": ["https://github.com/pytorch/pytorch/issues/197448"],
        "cases": [dataclasses.asdict(c) for c in results],
        "any_expand_fill_divergence": any_divergence,
        "guard_fully_correct": guard_fully_correct,
    }


# Public guard function bound lazily against the currently installed
# torch build (import-time torch import would break "torch not
# installed" degradation -- see TorchUnavailableError above).
def safe_fill_(tensor, value):
    """Module-level convenience wrapper: resolves torch on first call
    and delegates to the bound guard. See ``_make_safe_fill_`` for the
    full rationale."""
    torch_module = _import_torch()
    return _make_safe_fill_(torch_module)(tensor, value)
