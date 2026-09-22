"""torch-embeddingbag-freq-scale-guard core: detect and guard a real
``torch.nn.functional.embedding_bag(..., scale_grad_by_freq=True)``
correctness bug on the MPS (Apple Silicon GPU) backend, where the
1/frequency gradient scaling documented and implemented on CPU is
silently never applied on MPS.

Upstream reference: pytorch/pytorch#190061 ("[MPS] embedding_bag
silently ignores scale_grad_by_freq=True"), status as of this guard's
creation (2026-09-20): open, labeled "module: correctness (silent)".
A fix PR (#190062, "[MPS] Honor include_last_offset and
scale_grad_by_freq in embedding_bag") exists upstream but is UNMERGED
(state=OPEN, mergedAt=null -- independently re-checked via
``gh pr view 190062 --repo pytorch/pytorch``, never trusted from a
cached issue summary alone).

The bug, reproduced from scratch on this host (Apple M4, macOS
15.7.7, torch 2.14.0, real MPS device -- not simulated): CPU
``embedding_bag`` backward divides each index's gradient contribution
by how many times that index occurs in the batch (mode="sum",
scale_grad_by_freq=True); MPS returns the UNSCALED gradient with no
error or warning, silently ignoring the flag entirely. Confirmed via
control test that scale_grad_by_freq=False gives identical CPU/MPS
results -- isolating the defect to the scale_grad_by_freq=True path
specifically, and via aten/src/ATen/native/mps/operations/
EmbeddingBag.mm inspection: the MPS backward kernel accepts the flag
as a parameter but never reads it (no frequency-count computation at
all, unlike the CPU path, which does).

Real-world impact: any training loop on Apple Silicon relying on
``scale_grad_by_freq=True`` to down-weight gradients from
frequently-repeated tokens/IDs (a standard technique from the
original word2vec/GloVe literature, still used in some
recommendation-system and NLP embedding training setups) silently
gets a DIFFERENT, uncorrected gradient on MPS than on CPU/CUDA, with
no warning -- a silent training-divergence bug specific to Apple
Silicon GPU training.

This module's guard, ``safe_embedding_bag``, applies the correct
1/frequency gradient scaling itself via a custom autograd Function
whenever ``scale_grad_by_freq=True`` is requested, REGARDLESS of
device -- so the result is guaranteed correct (matching the
CPU-documented semantics) on MPS, CPU, or CUDA alike, rather than
depending on which backend's native kernel happens to implement the
flag. When ``scale_grad_by_freq=False`` (the common case), the guard
delegates straight to the native op unchanged (no double-guarding of
an already-correct path).

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


def make_safe_embedding_bag(torch_module):
    """Build a guard function, bound to a specific torch module, that
    applies the correct 1/frequency gradient scaling for
    ``scale_grad_by_freq=True`` via a custom autograd Function,
    independent of which backend's native kernel does or doesn't
    implement the flag. Returns a callable matching
    ``torch.nn.functional.embedding_bag``'s ``(input, weight,
    offsets=None, mode="mean", scale_grad_by_freq=False) -> Tensor``
    signature for the subset of arguments this guard covers.
    """
    F = torch_module.nn.functional

    class _FreqScaledEmbeddingBag(torch_module.autograd.Function):
        @staticmethod
        def forward(ctx, weight, input, offsets, mode_idx):
            modes = {0: "sum", 1: "mean", 2: "max"}
            mode = modes[mode_idx]
            out = F.embedding_bag(
                input,
                weight,
                offsets,
                mode=mode,
                scale_grad_by_freq=False,
            )
            ctx.save_for_backward(weight, input, offsets)
            ctx.mode = mode
            return out

        @staticmethod
        def backward(ctx, grad_output):
            weight, input, offsets = ctx.saved_tensors
            mode = ctx.mode
            with torch_module.enable_grad():
                w = weight.detach().requires_grad_(True)
                out = F.embedding_bag(input, w, offsets, mode=mode, scale_grad_by_freq=False)
                (raw_grad_weight,) = torch_module.autograd.grad(
                    out, w, grad_output, retain_graph=False
                )
            flat_input = input.reshape(-1)
            num_embeddings = weight.shape[0]
            counts = torch_module.bincount(
                flat_input, minlength=num_embeddings
            ).to(dtype=raw_grad_weight.dtype)
            counts = counts.clamp(min=1)
            scale = (1.0 / counts).unsqueeze(1)
            scaled_grad_weight = raw_grad_weight * scale
            return scaled_grad_weight, None, None, None

    def _safe_embedding_bag(input, weight, offsets=None, mode="mean", scale_grad_by_freq=False):
        if not scale_grad_by_freq:
            # Already correct (and this flag is irrelevant) on every
            # backend when the flag isn't requested: delegate unchanged.
            return F.embedding_bag(input, weight, offsets, mode=mode)
        mode_idx = {"sum": 0, "mean": 1, "max": 2}[mode]
        return _FreqScaledEmbeddingBag.apply(weight, input, offsets, mode_idx)

    return _safe_embedding_bag


@dataclasses.dataclass
class DeviceCase:
    device: str
    ran: bool
    skip_reason: Optional[str]
    native_grad_row1: Optional[List[float]]
    guard_grad_row1: Optional[List[float]]
    cpu_oracle_grad_row1: Optional[List[float]]
    native_matches_oracle: Optional[bool]
    guard_matches_oracle: Optional[bool]


def _row1_grad(torch_module, weight_grad_fn, device: str) -> List[float]:
    """Run the shared repro (index 1 appears twice in a 4-row, 3-dim
    embedding table, mode='sum') on `device` and return the gradient
    row for the doubly-occurring index (index 1), which is where the
    1/frequency scaling bug is visible."""
    weight = torch_module.zeros(4, 3, device=device, requires_grad=True)
    input_idx = torch_module.tensor([1, 1, 0, 2], device=device)
    offsets = torch_module.tensor([0], device=device)
    grad_weight = weight_grad_fn(weight, input_idx, offsets)
    return grad_weight[1].detach().cpu().tolist()


def diagnose() -> Dict[str, Any]:
    """Reproduce the MPS embedding_bag(scale_grad_by_freq=True) silent
    gradient-scaling bug from scratch against the currently installed
    torch build, on every device this host can actually exercise, and
    verify ``make_safe_embedding_bag``'s guard against a CPU oracle
    (CPU's own native behavior, which correctly implements 1/frequency
    scaling and is the documented reference). Never trusts a cached or
    previously-reported result -- every call re-runs the actual repro.
    """
    torch_module = _import_torch()
    F = torch_module.nn.functional
    safe_fn = make_safe_embedding_bag(torch_module)

    def native_grad(device):
        def _fn(weight, input_idx, offsets):
            out = F.embedding_bag(
                input_idx, weight, offsets, mode="sum", scale_grad_by_freq=True
            )
            out.sum().backward()
            return weight.grad

        return _row1_grad(torch_module, _fn, device)

    def guard_grad(device):
        def _fn(weight, input_idx, offsets):
            out = safe_fn(input_idx, weight, offsets, mode="sum", scale_grad_by_freq=True)
            out.sum().backward()
            return weight.grad

        return _row1_grad(torch_module, _fn, device)

    cpu_oracle = native_grad("cpu")

    devices_to_try = ["cpu"]
    mps_functional = mps_is_functional(torch_module)
    if mps_functional:
        devices_to_try.append("mps")

    cases: List[DeviceCase] = []
    for device in ("cpu", "mps"):
        if device not in devices_to_try:
            cases.append(
                DeviceCase(
                    device=device,
                    ran=False,
                    skip_reason=(
                        "MPS reported available but failed a real allocation "
                        "probe (virtualized/non-functional runner -- see "
                        "module docstring); this is an environment "
                        "limitation, not a code defect."
                        if device == "mps" and hasattr(torch_module.backends, "mps")
                        and torch_module.backends.mps.is_available()
                        else f"{device} not available on this host"
                    ),
                    native_grad_row1=None,
                    guard_grad_row1=None,
                    cpu_oracle_grad_row1=cpu_oracle,
                    native_matches_oracle=None,
                    guard_matches_oracle=None,
                )
            )
            continue

        native = native_grad(device)
        guard = guard_grad(device)
        native_matches = all(
            abs(a - b) < 1e-6 for a, b in zip(native, cpu_oracle)
        )
        guard_matches = all(
            abs(a - b) < 1e-6 for a, b in zip(guard, cpu_oracle)
        )
        cases.append(
            DeviceCase(
                device=device,
                ran=True,
                skip_reason=None,
                native_grad_row1=native,
                guard_grad_row1=guard,
                cpu_oracle_grad_row1=cpu_oracle,
                native_matches_oracle=native_matches,
                guard_matches_oracle=guard_matches,
            )
        )

    any_native_silently_wrong = any(
        c.ran and c.native_matches_oracle is False for c in cases
    )
    guard_fully_correct = all(
        c.guard_matches_oracle for c in cases if c.ran
    )

    return {
        "torch_version": torch_module.__version__,
        "issue_url": "https://github.com/pytorch/pytorch/issues/190061",
        "mps_functional": mps_functional,
        "cases": [dataclasses.asdict(c) for c in cases],
        "any_native_silently_wrong": any_native_silently_wrong,
        "guard_fully_correct": guard_fully_correct,
    }


# Public convenience wrapper: resolves torch lazily so importing this
# module without torch installed doesn't crash (matching the sibling
# guard repos' degradation pattern).
def safe_embedding_bag(input, weight, offsets=None, mode="mean", scale_grad_by_freq=False):
    """Module-level convenience wrapper around
    ``make_safe_embedding_bag``: resolves torch on first call. See
    that function's docstring for the full rationale and semantics."""
    torch_module = _import_torch()
    return make_safe_embedding_bag(torch_module)(
        input, weight, offsets=offsets, mode=mode, scale_grad_by_freq=scale_grad_by_freq
    )
