"""torch-checkpoint-noise-guard core: guard against silently wrong
gradients for stochastic activations (F.rrelu / nn.RReLU) under
non-reentrant activation checkpointing or any saved-tensors hook that
materializes a tensor at pack time.

Reproduced from scratch on this host (torch 2.14.0, macOS arm64 CPU)
using the EXACT repro code from pytorch/pytorch#193671 (open, root
cause diagnosed by the reporter, fix not yet merged as of this run):

    F.rrelu(x, lower, upper, training=True) dispatches to
    aten::rrelu_with_noise(self, noise, ...), where `noise` is a
    *mutable output argument*: the kernel allocates it uninitialized
    (via at::empty_like) and fills it with the sampled negative slopes
    as a side effect. The autograd-generated wrapper constructs the
    `SavedVariable` for `noise` BEFORE the kernel runs, so any hook
    that captures the tensor at "pack" time (torch.utils.checkpoint's
    non-reentrant recomputation machinery, or an explicit
    `saved_tensors_hooks` pack callback) can observe the buffer before
    it is filled.

    Under non-reentrant checkpointing specifically: recomputation is
    aborted via `_StopRecomputationError` once every saved tensor from
    the original forward has been re-produced. If `noise` is among the
    last tensors saved in a segment, the abort fires before the kernel
    that fills it actually runs, so the backward pass reads
    uninitialized memory as if it were the real sampled noise. Because
    `rrelu_with_noise` is a mutable (not inplace/out) op, no
    `increment_version` call is emitted either, so SavedVariable's
    staleness check cannot catch this.

    The forward output is bit-exact and unaffected in all cases, which
    is exactly what makes this dangerous: nothing about the model's
    predictions changes, only the gradients used to update parameters,
    silently, with no error, warning, or NaN in most runs (uninitialized
    memory sometimes surfaces as extreme finite values, as reproduced
    below, or occasionally as 0/NaN, but never a raised exception).

Independently reproduced on this host using the issue's own snippet
(see tests/test_core.py::test_reproduce_upstream_bug_193671 for the
exact numbers observed in this run): the checkpoint and saved-hooks
gradient paths diverge from the uncheckpointed reference by very large
finite magnitudes (order 1e0 to 1e22 across repeated runs -- consistent
with "reading uninitialized memory", not a fixed deterministic value),
while forward outputs remain bit-identical and `set_checkpoint_early_stop
(False)` (which disables the early-abort machinery that causes the bug)
makes the divergence disappear (diff exactly 0.0), confirming the
reported mechanism rather than an unrelated numerical issue.

This module does not patch torch internals (out of scope for a small
guard package, and the actual fix lives in PyTorch's generated autograd
code). Instead it provides `safe_rrelu`, a pure-Python drop-in
replacement for `F.rrelu(..., training=True)` built entirely from
ordinary (non-mutable-output) ops -- `torch.empty_like().uniform_()`
plus `torch.where()` -- so there is no hidden output-argument kernel
for a pack hook to observe uninitialized. This was independently
verified on this host to produce IDENTICAL gradients under checkpoint,
saved_tensors_hooks, and plain eager execution (zero difference, no
NaN), whereas the buggy `F.rrelu` diverges under the first two.
"""
from __future__ import annotations

import contextlib
import dataclasses
from typing import Any, Dict, Optional


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


def safe_rrelu(x, lower: float = 1.0 / 8, upper: float = 1.0 / 3, training: bool = True, generator=None):
    """Drop-in replacement for ``torch.nn.functional.rrelu`` that avoids
    the silently-wrong-gradient bug under non-reentrant activation
    checkpointing or any pack-time saved-tensors hook
    (pytorch/pytorch#193671).

    In eval mode (``training=False``) this delegates to the real
    ``F.rrelu`` unchanged, matching its documented behavior (a fixed
    slope of ``(lower + upper) / 2``, no randomness, no mutable-output
    kernel involved).

    In training mode, this samples the per-element negative slope with
    ``torch.empty_like(x).uniform_(lower, upper, generator=generator)``
    and applies it with ``torch.where`` -- ordinary (non-mutable-output)
    ops whose `SavedVariable`s are captured only after they hold their
    final value, so there is no uninitialized-buffer race for a
    checkpoint/hook pack callback to observe. Verified on this host to
    produce numerically identical gradients across eager, non-reentrant
    checkpointing, and an explicit clone-based ``saved_tensors_hooks``
    context (see tests/test_core.py), unlike the real ``F.rrelu`` which
    diverges under the latter two.
    """
    torch_module = _import_torch()
    import torch.nn.functional as F

    if not training:
        return F.rrelu(x, lower, upper, training=False)
    negative_slope = torch_module.empty_like(x).uniform_(lower, upper, generator=generator)
    return torch_module.where(x >= 0, x, x * negative_slope)


@dataclasses.dataclass
class RreluCheckpointCase:
    mode: str  # "checkpoint" or "saved_hooks"
    buggy_diff: float
    buggy_forward_diff: float
    guard_diff: float
    guard_has_nan: bool
    no_early_stop_diff: Optional[float]  # only populated for the "checkpoint" mode


def _rrelu_buggy(t):
    import torch.nn.functional as F

    return F.rrelu(t, 0.1, 0.9, training=True)


def _rrelu_guarded(t):
    return safe_rrelu(t, 0.1, 0.9, training=True)


def _grad(torch_module, fn, outer=None, use_checkpoint: bool = False):
    from torch.utils.checkpoint import checkpoint

    if outer is None:
        outer = contextlib.nullcontext()
    torch_module.manual_seed(0)
    x = torch_module.randn(4096, requires_grad=True)
    with outer:
        y = checkpoint(fn, x, use_reentrant=False) if use_checkpoint else fn(x)
        y.sum().backward()
    return x.grad


def _run_checkpoint_case(torch_module) -> RreluCheckpointCase:
    from torch.utils.checkpoint import set_checkpoint_early_stop

    ref_buggy = _grad(torch_module, _rrelu_buggy)
    ckpt_buggy = _grad(torch_module, _rrelu_buggy, use_checkpoint=True)
    buggy_diff = float((ckpt_buggy - ref_buggy).abs().max().item())

    torch_module.manual_seed(0)
    a = torch_module.randn(4096, requires_grad=True)
    from torch.utils.checkpoint import checkpoint as _ckpt

    out_ckpt = _ckpt(_rrelu_buggy, a, use_reentrant=False)
    torch_module.manual_seed(0)
    b = torch_module.randn(4096, requires_grad=True)
    out_eager = _rrelu_buggy(b)
    forward_diff = float((out_ckpt - out_eager).abs().max().item())

    with set_checkpoint_early_stop(False):
        ckpt_no_early_stop = _grad(torch_module, _rrelu_buggy, use_checkpoint=True)
    no_early_stop_diff = float((ckpt_no_early_stop - ref_buggy).abs().max().item())

    ref_guard = _grad(torch_module, _rrelu_guarded)
    ckpt_guard = _grad(torch_module, _rrelu_guarded, use_checkpoint=True)
    guard_diff = float((ckpt_guard - ref_guard).abs().max().item())

    return RreluCheckpointCase(
        mode="checkpoint",
        buggy_diff=buggy_diff,
        buggy_forward_diff=forward_diff,
        guard_diff=guard_diff,
        guard_has_nan=bool(torch_module.isnan(ckpt_guard).any().item()),
        no_early_stop_diff=no_early_stop_diff,
    )


def _run_saved_hooks_case(torch_module) -> RreluCheckpointCase:
    from torch.autograd.graph import saved_tensors_hooks

    clone_hooks = saved_tensors_hooks(lambda t: t.clone(), lambda t: t)

    ref_buggy = _grad(torch_module, _rrelu_buggy)
    hooks_buggy = _grad(torch_module, _rrelu_buggy, outer=clone_hooks)
    buggy_diff = float((hooks_buggy - ref_buggy).abs().max().item())

    ref_guard = _grad(torch_module, _rrelu_guarded)
    hooks_guard = _grad(torch_module, _rrelu_guarded, outer=clone_hooks)
    guard_diff = float((hooks_guard - ref_guard).abs().max().item())

    return RreluCheckpointCase(
        mode="saved_hooks",
        buggy_diff=buggy_diff,
        buggy_forward_diff=0.0,
        guard_diff=guard_diff,
        guard_has_nan=bool(torch_module.isnan(hooks_guard).any().item()),
        no_early_stop_diff=None,
    )


def diagnose() -> Dict[str, Any]:
    """Reproduce pytorch/pytorch#193671 from scratch against the
    currently installed torch build, using the exact mechanism from the
    upstream issue, and verify ``safe_rrelu`` avoids it. Never trusts a
    cached/prior result -- every call re-runs the actual repro.

    Because the corruption reads uninitialized memory, the exact
    magnitude of ``buggy_diff`` is expected to vary run to run (it has
    been observed on this host anywhere from order 1.0 to order 1e22);
    what is deterministic and checked here is that it is nonzero (a
    real divergence occurred) while ``guard_diff`` is exactly 0.0 and
    contains no NaN in every mode, and that disabling the early-stop
    machinery (the documented root-cause trigger) makes the buggy
    divergence disappear too, confirming the mechanism rather than
    treating any nonzero number as sufficient evidence on its own.
    """
    torch_module = _import_torch()

    checkpoint_case = _run_checkpoint_case(torch_module)
    saved_hooks_case = _run_saved_hooks_case(torch_module)

    bug_reproduced = checkpoint_case.buggy_diff > 0.0 or saved_hooks_case.buggy_diff > 0.0
    mechanism_confirmed = (
        checkpoint_case.no_early_stop_diff is not None
        and checkpoint_case.no_early_stop_diff == 0.0
        and checkpoint_case.buggy_forward_diff == 0.0
    )
    guard_fully_correct = (
        checkpoint_case.guard_diff == 0.0
        and not checkpoint_case.guard_has_nan
        and saved_hooks_case.guard_diff == 0.0
        and not saved_hooks_case.guard_has_nan
    )

    return {
        "torch_version": torch_module.__version__,
        "issue_urls": ["https://github.com/pytorch/pytorch/issues/193671"],
        "checkpoint_case": dataclasses.asdict(checkpoint_case),
        "saved_hooks_case": dataclasses.asdict(saved_hooks_case),
        "bug_reproduced": bug_reproduced,
        "mechanism_confirmed": mechanism_confirmed,
        "guard_fully_correct": guard_fully_correct,
    }
