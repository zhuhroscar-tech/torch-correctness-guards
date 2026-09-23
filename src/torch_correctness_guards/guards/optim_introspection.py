"""torch-optim-introspection-guard core: detect and safely work around
PyTorch's ``get_optimizer_state_dict()`` step-counter mutation bug
(pytorch/pytorch#164929, open since 2025-10-08; fix PR #166362 closed
unmerged 2026-03-10; last upstream activity 2026-07-30 -- see README for
the live-source verification trail).

The bug: ``torch.distributed.checkpoint.state_dict.get_optimizer_state_dict``
calls an internal ``_init_optim_state`` helper that runs ``optimizer.step()``
with ``lr=0`` on a freshly-constructed optimizer to force it to allocate its
per-parameter state buffers (needed for FSDP sharding). That ``step()`` call
is not truly a no-op: for optimizers whose step formula depends on the
*step count itself* (Adam-family bias correction terms use
``1 - beta1**step`` and ``1 - beta2**step``), incrementing the step counter
from 0 to 1 changes every subsequent real step's bias-correction scale --
even though the parameter *values* are mathematically unchanged by the
lr=0 step. The result: merely *inspecting* an optimizer's state (e.g. for
logging, debugging, or a mid-training checkpoint peek) silently changes
the trajectory of training, with no error, warning, or visible symptom
until two runs that should be bit-for-bit identical diverge.

This module provides:

  - ``PROBE_OPTIMIZERS``: a small registry of optimizer constructors that
    is used to both diagnose which optimizer families are affected on the
    currently installed torch version, and to regression-test the guard.
  - ``diagnose()``: reproduces the bug from scratch, per optimizer family,
    against whatever torch version is actually installed on this host --
    never trusts the upstream issue tracker's version report alone.
  - ``safe_get_optimizer_state_dict()``: a drop-in wrapper around
    ``get_optimizer_state_dict`` that snapshots the optimizer's *full*
    state (not just the fields the bug is known to touch) before the
    call and restores it afterward, so introspection is truly read-only
    regardless of which internal fields a given optimizer implementation
    happens to mutate.
"""
from __future__ import annotations

import copy
import dataclasses
from typing import Any, Callable, Dict, List, Optional


class TorchUnavailableError(RuntimeError):
    """Raised when torch (or the distributed.checkpoint.state_dict module)
    cannot be imported. Kept as a distinct type so callers can distinguish
    "torch isn't installed" from an actual diagnostic failure."""


def _import_torch():
    try:
        import torch  # noqa: F401
        from torch.distributed.checkpoint.state_dict import (  # noqa: F401
            StateDictOptions,
            get_optimizer_state_dict,
        )
    except Exception as exc:  # pragma: no cover - exercised only without torch
        raise TorchUnavailableError(
            "torch (with torch.distributed.checkpoint.state_dict) is required "
            "for diagnosis and guarding; install the 'torch' extra."
        ) from exc
    return torch, StateDictOptions, get_optimizer_state_dict


@dataclasses.dataclass(frozen=True)
class OptimizerSpec:
    name: str
    factory: str  # attribute name on torch.optim
    kwargs: Dict[str, Any]


# Ordered so Adam-family (known-affected: bias correction depends on step
# count) comes first, then non-Adam families that do not track a
# step-dependent bias-correction term and are expected to be unaffected by
# THIS specific bug, even though the guard is applied uniformly to all of
# them since a future optimizer implementation change could reintroduce
# step-dependence in any of them.
PROBE_OPTIMIZERS: List[OptimizerSpec] = [
    OptimizerSpec("AdamW", "AdamW", {"lr": 0.1}),
    OptimizerSpec("Adam", "Adam", {"lr": 0.1}),
    OptimizerSpec("NAdam", "NAdam", {"lr": 0.1}),
    OptimizerSpec("RAdam", "RAdam", {"lr": 0.1}),
    OptimizerSpec("RMSprop", "RMSprop", {"lr": 0.1}),
    OptimizerSpec("SGD-momentum", "SGD", {"lr": 0.1, "momentum": 0.9}),
    OptimizerSpec("SGD-plain", "SGD", {"lr": 0.1}),
    OptimizerSpec("Adagrad", "Adagrad", {"lr": 0.1}),
    OptimizerSpec("Adadelta", "Adadelta", {"lr": 0.1}),
]


def safe_get_optimizer_state_dict(model, optimizer, options=None):
    """Read an optimizer's state via ``get_optimizer_state_dict`` without
    letting the read mutate the optimizer's actual training trajectory.

    Strategy: snapshot the optimizer's full ``state_dict()`` (param_groups
    plus per-parameter state) before the call. If the optimizer had no
    state at all yet (a fresh optimizer, the exact case that triggers the
    bug's internal ``_init_optim_state`` initialization step), restore it
    to that same pristine empty-state condition afterward rather than a
    stale non-empty snapshot -- otherwise a subsequent real step would see
    inconsistent internal buffers. If it already had state (a later
    mid-training introspection call), restore that exact prior state
    byte-for-byte via ``load_state_dict``, undoing whatever the internal
    lr=0 probe step changed.

    This snapshots the *whole* state dict, not just the ``step`` counter
    the known bug happens to touch, so it also protects against any other
    field a given optimizer's ``step()`` implementation might mutate as a
    side effect of the same internal probing mechanism.

    Restoration runs in a ``finally`` block: if the wrapped
    ``get_optimizer_state_dict`` call raises AFTER it has already applied
    its internal step-counter mutation (e.g. a downstream distributed
    comms failure, an internal torch assertion, or any other exception
    part-way through the call), the optimizer's real training state is
    still restored/cleared before the exception propagates. Without this,
    the guard silently reintroduces the exact class of bug it exists to
    prevent -- a leaked, incremented step counter on a fresh optimizer,
    or an unrestored mid-training state -- but with an exception on top,
    making it more likely to be misdiagnosed as an unrelated failure.
    """
    _torch, _StateDictOptions, get_optimizer_state_dict_fn = _import_torch()

    had_state = bool(optimizer.state) and any(
        len(param_state) > 0 for param_state in optimizer.state.values()
    )
    snapshot = copy.deepcopy(optimizer.state_dict()) if had_state else None

    try:
        result = get_optimizer_state_dict_fn(model, optimizer, options=options)
    finally:
        if had_state:
            optimizer.load_state_dict(snapshot)
        else:
            optimizer.state.clear()

    return result


def _build_model_and_optimizer(torch_module, spec: OptimizerSpec, seed: int, dim: int = 5):
    torch_module.manual_seed(seed)
    weight = torch_module.randn(dim, dim)
    model = torch_module.nn.Linear(dim, dim, bias=False)
    model.weight.data.copy_(weight)
    optimizer_cls = getattr(torch_module.optim, spec.factory)
    optimizer = optimizer_cls(model.parameters(), **spec.kwargs)
    return model, optimizer, weight


def _run_probe(torch_module, state_dict_options_cls, get_state_dict_fn, spec: OptimizerSpec, seed: int, mode: str):
    """mode is one of 'baseline' (never introspect), 'raw' (introspect with
    the unguarded upstream function), or 'guarded' (introspect with
    safe_get_optimizer_state_dict). Returns the post-first-real-step
    weight tensor, which is the observable proxy for "did introspection
    change the training trajectory"."""
    model, optimizer, weight = _build_model_and_optimizer(torch_module, spec, seed)
    if mode == "raw":
        get_state_dict_fn(model, optimizer, options=state_dict_options_cls(full_state_dict=True))
    elif mode == "guarded":
        safe_get_optimizer_state_dict(model, optimizer, options=state_dict_options_cls(full_state_dict=True))
    elif mode != "baseline":
        raise ValueError(f"unknown mode: {mode}")
    model.weight.grad = weight.clone()
    optimizer.step()
    return model.weight.detach().clone()


@dataclasses.dataclass
class OptimizerDiagnosis:
    name: str
    bug_present: bool          # raw introspection diverges from baseline
    guard_effective: bool      # guarded introspection matches baseline
    max_abs_diff_raw: float
    max_abs_diff_guarded: float


def diagnose(seed: int = 20260915) -> Dict[str, Any]:
    """Reproduce the bug from scratch against the currently installed torch
    build, for every optimizer family in PROBE_OPTIMIZERS, and verify the
    guard neutralizes it in every case. Never trusts a cached/prior
    result -- every call re-runs the actual repro."""
    torch_module, state_dict_options_cls, get_state_dict_fn = _import_torch()

    results: List[OptimizerDiagnosis] = []
    for spec in PROBE_OPTIMIZERS:
        baseline = _run_probe(torch_module, state_dict_options_cls, get_state_dict_fn, spec, seed, "baseline")
        raw = _run_probe(torch_module, state_dict_options_cls, get_state_dict_fn, spec, seed, "raw")
        guarded = _run_probe(torch_module, state_dict_options_cls, get_state_dict_fn, spec, seed, "guarded")

        diff_raw = (baseline - raw).abs().max().item()
        diff_guarded = (baseline - guarded).abs().max().item()

        results.append(
            OptimizerDiagnosis(
                name=spec.name,
                bug_present=diff_raw > 1e-6,
                guard_effective=diff_guarded <= 1e-6,
                max_abs_diff_raw=diff_raw,
                max_abs_diff_guarded=diff_guarded,
            )
        )

    return {
        "torch_version": torch_module.__version__,
        "issue_url": "https://github.com/pytorch/pytorch/issues/164929",
        "results": [dataclasses.asdict(r) for r in results],
        "any_bug_present": any(r.bug_present for r in results),
        "guard_fully_effective": all(r.guard_effective for r in results),
    }
