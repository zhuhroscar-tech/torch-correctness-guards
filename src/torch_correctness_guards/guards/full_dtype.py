"""torch-inductor-full-dtype-guard core: detect and fix a real
``torch.compile(backend="inductor")`` correctness bug where
``torch.full(size, fill_value, dtype=X)`` silently drops the ``dtype``
cast when ``fill_value`` is a *symbolic* (traced) scalar -- a closure
int across a recompile, a ``.item()`` result, or any other value
Dynamo represents as a SymInt/SymBool rather than a Python constant.

Reproduced from scratch on this host (torch 2.14.0, macOS arm64 CPU;
see README for the exact commands):

  def f(x):
      return torch.full((2,), x.item(), dtype=torch.bool).sum()

  f(torch.tensor(3))                        # eager:    tensor(2)
  torch.compile(f, fullgraph=True)(torch.tensor(3))  # inductor: tensor(6)

Eager saturates the bool fill of 3 to True before the .sum(), matching
the documented (and reasonable) semantics: any nonzero fill becomes
True, so summing two Trues gives 2. Inductor's ``SymPyOps.to_dtype``
(torch/_inductor/index_propagation.py) retags the traced expression's
dtype without applying the actual cast for a non-constant value, so
the raw fill value (3) survives uncast into the ``.sum()``, giving 6
instead of 2 -- with no error, warning, or non-finite marker. This is
strictly worse than the more common "wrong number" class of bug: nothing
about the output looks obviously invalid, so a training or inference
pipeline can silently consume the wrong boolean mask indefinitely.

A second symptom of the same missing cast is not merely a wrong value
but a missing safety check: ``torch.full(size, 300, dtype=torch.int8)``
under eager mode raises ``RuntimeError: value cannot be converted to
type int8_t without overflow`` for a symbolic fill -- exactly the
overflow protection an int8 dtype exists to provide. Under
``torch.compile(backend="inductor")`` the same call silently succeeds
and returns a wrapped-around, meaningless int8 value instead of
raising, again with zero signal that anything went wrong.

Upstream reference: pytorch/pytorch#194062 (open as of this writing,
"[inductor] [silent incorrectness] torch.full ignores dtype when
fill_value is symbolic"). A related fold-based issue in the same area,
#193876, was fixed by commit aca31b5 -- but that fix is scoped to a
different code path (``pointless_cumsum_replacement``, an optimization
that folds ``full(...).cumsum(dim)``) and explicitly does NOT fix
#194062 itself; that commit's own message says the exemption it adds
"has a shelf life: once #194062 is fixed it becomes strictly wrong".
This guard targets #194062 directly, which remains open.

This module's guard function, ``safe_full``, forces the actual
``torch.full`` call outside the compiled graph via
``torch.compiler.disable`` (a Dynamo graph break at this single,
already-cheap allocation call site), so the real eager kernel -- which
performs the genuine per-element dtype cast and overflow check -- runs
regardless of whether the caller is itself under ``torch.compile``.
This trades a small amount of graph fusion at the ``full()`` call site
for correctness; the guard function is meant as a drop-in replacement
for ``torch.full`` at call sites where the fill value may be symbolic
under compilation (a traced Python int/bool, or any tensor-derived
scalar via ``.item()``).
"""
from __future__ import annotations

import dataclasses
from typing import Any, Dict, List, Optional, Sequence, Tuple


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


def _make_safe_full(torch_module):
    """Build the guard function bound to a specific torch module (so
    the same core logic works against whatever torch build is actually
    installed, without importing torch at module load time)."""

    @torch_module.compiler.disable
    def _eager_full(size, fill_value, dtype=None, **kwargs):
        return torch_module.full(size, fill_value, dtype=dtype, **kwargs)

    def safe_full(size, fill_value, dtype=None, **kwargs):
        """Drop-in guard for ``torch.full``: forces the actual fill
        (and its dtype cast / overflow check) to run in eager mode via
        a Dynamo graph break, so a symbolic ``fill_value`` under
        ``torch.compile(backend="inductor")`` gets the same cast
        semantics eager already provides -- matching eager output
        exactly instead of silently keeping an uncast raw value."""
        return _eager_full(size, fill_value, dtype=dtype, **kwargs)

    return safe_full


@dataclasses.dataclass
class BoolFillCase:
    fill_value: int
    eager_result: int
    compiled_native_result: int
    compiled_guarded_result: int
    native_diverges: bool  # eager != compiled_native (the bug)
    guard_matches_eager: bool


def _run_bool_fill_case(torch_module, safe_full, fill_value: int) -> BoolFillCase:
    # The fill value must actually be SYMBOLIC (a traced SymInt), not a
    # Python-constant closure captured on first compile -- Inductor's
    # constant-and-index-propagation path folds a genuine compile-time
    # constant correctly, so the bug only manifests for a value Dynamo
    # cannot prove constant. Using x.item() with
    # capture_scalar_outputs=True (matching the upstream issue's own
    # repro) reliably produces a symbolic fill on every call, without
    # depending on a fragile multi-recompile promotion sequence.
    prior_capture_flag = torch_module._dynamo.config.capture_scalar_outputs
    torch_module._dynamo.config.capture_scalar_outputs = True
    try:
        def f_native(x):
            return torch_module.full((2,), x.item(), dtype=torch_module.bool).sum()

        def f_guarded(x):
            return safe_full((2,), x.item(), dtype=torch_module.bool).sum()

        x = torch_module.tensor(fill_value)

        eager_val = int(f_native(x).item())

        torch_module._dynamo.reset()
        compiled_native = torch_module.compile(f_native, fullgraph=True)
        native_val = int(compiled_native(x).item())

        torch_module._dynamo.reset()
        compiled_guarded = torch_module.compile(f_guarded, fullgraph=False)
        guarded_val = int(compiled_guarded(x).item())
    finally:
        torch_module._dynamo.config.capture_scalar_outputs = prior_capture_flag

    return BoolFillCase(
        fill_value=fill_value,
        eager_result=eager_val,
        compiled_native_result=native_val,
        compiled_guarded_result=guarded_val,
        native_diverges=(native_val != eager_val),
        guard_matches_eager=(guarded_val == eager_val),
    )


@dataclasses.dataclass
class Int8OverflowCase:
    fill_value: int
    eager_raised: bool
    compiled_native_raised: bool
    compiled_native_silent_value: Optional[int]
    compiled_guarded_raised: bool
    native_silently_wrong: bool  # eager raises, compiled native does not
    guard_matches_eager: bool
    overflow_dtype: str = "int8"  # widened 2026-09-19: the same missing-cast
    # bug and the same safe_full() fix were confirmed to generalize to every
    # narrow integer dtype torch.full's overflow check covers (int16, uint8),
    # not just int8. Kept as a trailing field with a default so existing
    # positional/keyword construction of this dataclass from before this
    # widening still works.


def _run_int8_overflow_case(
    torch_module, safe_full, fill_value: int, dtype=None
) -> Int8OverflowCase:
    """Reproduce the missing-cast silent-overflow bug for one (dtype,
    fill_value) pair. Originally hardcoded to dtype=torch.int8; widened to
    accept any narrow integer dtype so the same bug/fix can be verified
    beyond int8 -- see ``diagnose()``'s ``overflow_dtype_fill_values``."""
    if dtype is None:
        dtype = torch_module.int8
    prior_capture_flag = torch_module._dynamo.config.capture_scalar_outputs
    torch_module._dynamo.config.capture_scalar_outputs = True
    try:
        def g_native(x):
            return torch_module.full((2,), x.item(), dtype=dtype)

        def g_guarded(x):
            return safe_full((2,), x.item(), dtype=dtype)

        hundred = torch_module.tensor(fill_value)

        eager_raised = False
        try:
            g_native(hundred)
        except RuntimeError:
            eager_raised = True

        torch_module._dynamo.reset()
        compiled_native = torch_module.compile(g_native, fullgraph=True)
        native_raised = False
        native_value: Optional[int] = None
        try:
            result = compiled_native(hundred)
            native_value = int(result[0].item())
        except RuntimeError:
            native_raised = True

        torch_module._dynamo.reset()
        compiled_guarded = torch_module.compile(g_guarded, fullgraph=False)
        guarded_raised = False
        try:
            compiled_guarded(hundred)
        except RuntimeError:
            guarded_raised = True
    finally:
        torch_module._dynamo.config.capture_scalar_outputs = prior_capture_flag

    return Int8OverflowCase(
        fill_value=fill_value,
        eager_raised=eager_raised,
        compiled_native_raised=native_raised,
        compiled_native_silent_value=native_value,
        compiled_guarded_raised=guarded_raised,
        native_silently_wrong=(eager_raised and not native_raised),
        guard_matches_eager=(guarded_raised == eager_raised),
        overflow_dtype=str(dtype).replace("torch.", ""),
    )


def diagnose(
    bool_fill_values: Sequence[int] = (0, 1, 2, 3, 7),
    int8_overflow_fill_values: Sequence[int] = (300, -200, 1000),
    extra_overflow_dtype_cases: Optional[Sequence[Tuple[str, int]]] = None,
) -> Dict[str, Any]:
    """Reproduce the eager-vs-Inductor ``torch.full`` symbolic-fill
    dtype-cast divergence from scratch against the currently installed
    torch build, and verify ``safe_full`` matches eager in every case.
    Never trusts a cached/prior result -- every call re-runs the
    actual repro.

    ``extra_overflow_dtype_cases`` widens the originally int8-only
    overflow sweep to other narrow integer dtypes (name, fill_value)
    pairs -- e.g. ``[("int16", 40000), ("uint8", 300)]``. Defaults to
    int16 and uint8 in addition to the original int8 sweep.

    IMPORTANT, confirmed 2026-09-19: unlike the int8 case (a universal
    Inductor defect, reproducible in any process state), whether
    int16/uint8 exhibit the SAME silent-overflow bug is dependent on
    whether ``numpy`` happens to be importable in the current process
    at compile time -- not a property of this package's own declared
    dependencies (which do not include numpy). Verified directly: a
    clean venv with only this package's ``.[dev,torch]`` extra
    installed (no numpy) does NOT reproduce the int16/uint8 bug (both
    correctly raise under torch.compile, matching eager), while the
    exact same torch build in a process that also happens to have
    numpy importable DOES reproduce it identically to int8. The root
    mechanism was not traced further than this reproducible correlation
    (not asserted to be causal); treat ``any_overflow_dtype_silent_overflow``
    and each extra case's ``native_silently_wrong`` as environment-
    reported facts about the current process, not fixed properties of
    the installed torch version. ``safe_full()``'s guard is NOT subject
    to this hazard -- ``torch.compiler.disable`` forces real eager
    execution regardless of numpy's presence, so ``guard_matches_eager``
    is unconditionally true for these cases on this torch version.
    """
    torch_module = _import_torch()
    safe_full = _make_safe_full(torch_module)

    if extra_overflow_dtype_cases is None:
        extra_overflow_dtype_cases = [("int16", 40000), ("uint8", 300)]

    bool_cases: List[BoolFillCase] = [
        _run_bool_fill_case(torch_module, safe_full, fv) for fv in bool_fill_values
    ]
    int8_cases: List[Int8OverflowCase] = [
        _run_int8_overflow_case(torch_module, safe_full, fv)
        for fv in int8_overflow_fill_values
    ]
    extra_overflow_cases: List[Int8OverflowCase] = [
        _run_int8_overflow_case(
            torch_module, safe_full, fv, dtype=getattr(torch_module, dtype_name)
        )
        for dtype_name, fv in extra_overflow_dtype_cases
    ]
    all_overflow_cases = int8_cases + extra_overflow_cases

    any_bool_divergence = any(c.native_diverges for c in bool_cases)
    any_int8_silent_overflow = any(c.native_silently_wrong for c in int8_cases)
    any_overflow_dtype_silent_overflow = any(
        c.native_silently_wrong for c in all_overflow_cases
    )
    guard_fully_correct = all(c.guard_matches_eager for c in bool_cases) and all(
        c.guard_matches_eager for c in all_overflow_cases
    )

    return {
        "torch_version": torch_module.__version__,
        "issue_urls": ["https://github.com/pytorch/pytorch/issues/194062"],
        "bool_fill_cases": [dataclasses.asdict(c) for c in bool_cases],
        "int8_overflow_cases": [dataclasses.asdict(c) for c in int8_cases],
        "extra_overflow_dtype_cases": [
            dataclasses.asdict(c) for c in extra_overflow_cases
        ],
        "any_bool_fill_divergence": any_bool_divergence,
        "any_int8_silent_overflow": any_int8_silent_overflow,
        "any_overflow_dtype_silent_overflow": any_overflow_dtype_silent_overflow,
        "guard_fully_correct": guard_fully_correct,
    }


# Public guard function bound lazily against the currently installed
# torch build (import-time torch import would break "torch not
# installed" degradation -- see TorchUnavailableError above).
def safe_full(size, fill_value, dtype=None, **kwargs):
    """Module-level convenience wrapper: resolves torch on first call
    and delegates to the bound guard. See ``_make_safe_full`` for the
    full rationale."""
    torch_module = _import_torch()
    return _make_safe_full(torch_module)(size, fill_value, dtype=dtype, **kwargs)
