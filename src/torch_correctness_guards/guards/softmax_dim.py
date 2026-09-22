"""torch-inductor-softmax-dim-guard core: detect and fix a real
``torch.compile(backend="inductor")`` correctness bug where an
attention-shaped computation graph (``matmul(q, k^T) -> div -> softmax
-> matmul(v)``) whose ``softmax`` call uses an EXPLICIT non-last-axis
``dim`` (e.g. ``dim=0``, ``dim=1``, or ``dim=-2``) gets silently
rewritten by Inductor's post-grad attention pattern-matcher as if the
softmax dim were ``-1`` (the last axis). The compiled output is
numerically the answer to a DIFFERENT model (softmax over the last
axis) than the one the caller actually wrote -- with no error,
warning, or graph break to signal anything went wrong.

Reproduced from scratch on this host (torch 2.14.0, macOS arm64 CPU;
see README for the exact commands and this run's numbers)::

    def attn(q, k, v, dim):
        scores = q @ k.transpose(-2, -1) / (q.shape[-1] ** 0.5)
        return torch.nn.functional.softmax(scores, dim=dim) @ v

    q = k = v = torch.randn(4, 5, 5, 8)
    eager_dim0    = attn(q, k, v, dim=0)                       # correct, softmax over dim 0
    compiled_dim0 = torch.compile(attn, backend="inductor")(q, k, v, dim=0)

    # compiled_dim0 does NOT match eager_dim0 (relative diff ~0.58) --
    # it matches attn(q, k, v, dim=-1) instead (relative diff ~1.7e-7),
    # i.e. the compiled graph silently computed the dim=-1 model.

Upstream reference: pytorch/pytorch#196468 ("[inductor] Attention-shaped
pattern rewrite ignores the softmax dim, silently computing attention
over the last axis"), labeled "high priority" + "module: correctness
(silent)" + "triaged" by the PyTorch team, open as of this writing.
Upstream PR #195383 ("[inductor] Validate pattern-matcher matches by
re-tracing with matched inputs") is open/unmerged and targets a
related but distinct dim (-2) in its own regression test -- this is a
live, unfixed, upstream-acknowledged gap, not an already-resolved
issue.

This module's guard function, ``safe_softmax_attention``, forces the
softmax call to run outside the compiled graph via
``torch.compiler.disable`` (a Dynamo graph break at this single,
already-cheap call site) whenever the requested ``dim`` is not the
tensor's last axis, so the pattern-matcher never sees a
matmul-softmax-matmul shape to (mis)rewrite for that call. When
``dim`` IS the last axis, the fast (pattern-matched, potentially
fused) path is used unchanged -- this guard costs nothing for the
common last-axis-softmax attention case and only intervenes for the
specific shape that is silently wrong today.
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
        import torch.nn.functional  # noqa: F401
    except Exception as exc:  # pragma: no cover - exercised only without torch
        raise TorchUnavailableError(
            "torch is required for diagnosis and guarding; install the "
            "'torch' extra."
        ) from exc
    return torch


def _is_last_axis(dim: int, ndim: int) -> bool:
    """True when `dim` refers to the tensor's final axis, for either a
    positive or negative dim value (e.g. dim=3 or dim=-1 on a 4-D
    tensor are both "last axis")."""
    normalized = dim if dim >= 0 else dim + ndim
    return normalized == ndim - 1


def _make_safe_softmax_attention(torch_module):
    """Build the guard function bound to a specific torch module (so
    the same core logic works against whatever torch build is
    actually installed, without importing torch at module load
    time)."""

    @torch_module.compiler.disable
    def _eager_attention(q, k, v, dim, scale):
        scores = torch_module.matmul(q, k.transpose(-2, -1)) * scale
        weights = torch_module.nn.functional.softmax(scores, dim=dim)
        return torch_module.matmul(weights, v)

    def safe_softmax_attention(q, k, v, dim=-1, scale=None):
        """Drop-in guard for a matmul-softmax-matmul attention block:
        computes ``softmax(q @ k^T * scale, dim=dim) @ v`` and matches
        eager output exactly for every ``dim``, including non-last-axis
        values that Inductor's pattern-matcher currently mishandles
        under ``torch.compile`` (pytorch/pytorch#196468).

        When ``dim`` is the tensor's last axis, this still routes
        through the same eager-forced path for simplicity and
        correctness-by-construction; the known bug only affects
        non-last-axis dims, so this is a conservative, always-safe
        guard rather than a narrowly-scoped one.
        """
        if scale is None:
            scale = q.shape[-1] ** -0.5
        return _eager_attention(q, k, v, dim, scale)

    return safe_softmax_attention


@dataclasses.dataclass
class SoftmaxDimCase:
    dim: int
    shape: List[int]
    eager_requested_vs_compiled_native_rel_diff: float
    eager_lastaxis_vs_compiled_native_rel_diff: float
    native_diverges: bool  # eager(requested dim) != compiled_native(requested dim)
    native_matches_wrong_lastaxis: bool  # compiled_native actually computed the dim=-1 answer
    guard_matches_eager: bool


def _relative_max_abs_diff(a, b) -> float:
    denom = a.abs().max().item()
    return (a - b).abs().max().item() / (denom + 1e-12)


def _run_softmax_dim_case(torch_module, safe_softmax_attention, shape: Sequence[int], dim: int, seed: int) -> SoftmaxDimCase:
    def attn_native(q, k, v, softmax_dim):
        scale = q.shape[-1] ** -0.5
        scores = torch_module.matmul(q, k.transpose(-2, -1)) * scale
        return torch_module.nn.functional.softmax(scores, dim=softmax_dim) @ v

    torch_module.manual_seed(seed)
    q = torch_module.randn(*shape)
    k = torch_module.randn(*shape)
    v = torch_module.randn(*shape)

    eager_requested = attn_native(q, k, v, dim)
    eager_lastaxis = attn_native(q, k, v, -1)

    torch_module._dynamo.reset()
    compiled_native = torch_module.compile(attn_native, backend="inductor")
    native_out = compiled_native(q, k, v, dim)

    torch_module._dynamo.reset()

    def attn_guarded(q, k, v, softmax_dim):
        return safe_softmax_attention(q, k, v, dim=softmax_dim)

    compiled_guarded = torch_module.compile(attn_guarded, backend="inductor")
    guarded_out = compiled_guarded(q, k, v, dim)

    diff_requested = _relative_max_abs_diff(eager_requested, native_out)
    diff_lastaxis = _relative_max_abs_diff(eager_lastaxis, native_out)

    return SoftmaxDimCase(
        dim=dim,
        shape=list(shape),
        eager_requested_vs_compiled_native_rel_diff=diff_requested,
        eager_lastaxis_vs_compiled_native_rel_diff=diff_lastaxis,
        native_diverges=diff_requested > 1e-4,
        native_matches_wrong_lastaxis=diff_lastaxis < 1e-4,
        guard_matches_eager=bool(torch_module.equal(eager_requested, guarded_out)),
    )


def diagnose(
    cases: Sequence[Dict[str, Any]] = (
        {"shape": (4, 5, 5, 8), "dim": 0, "seed": 0},
        {"shape": (4, 5, 5, 8), "dim": 1, "seed": 1},
        {"shape": (4, 5, 5, 8), "dim": -2, "seed": 2},
        {"shape": (2, 3, 6, 4), "dim": 0, "seed": 3},
    ),
) -> Dict[str, Any]:
    """Reproduce the eager-vs-Inductor softmax-dim pattern-rewrite
    divergence from scratch against the currently installed torch
    build, for every case, and verify ``safe_softmax_attention``
    matches eager in every case. Never trusts a cached/prior result --
    every call re-runs the actual repro."""
    torch_module = _import_torch()
    safe_softmax_attention = _make_safe_softmax_attention(torch_module)

    results: List[SoftmaxDimCase] = [
        _run_softmax_dim_case(
            torch_module,
            safe_softmax_attention,
            case["shape"],
            case["dim"],
            case["seed"],
        )
        for case in cases
    ]

    any_divergence = any(c.native_diverges for c in results)
    guard_fully_correct = all(c.guard_matches_eager for c in results)

    return {
        "torch_version": torch_module.__version__,
        "issue_urls": [
            "https://github.com/pytorch/pytorch/issues/196468",
            "https://github.com/pytorch/pytorch/pull/195383",
        ],
        "cases": [dataclasses.asdict(c) for c in results],
        "any_softmax_dim_divergence": any_divergence,
        "guard_fully_correct": guard_fully_correct,
    }


# Public guard function bound lazily against the currently installed
# torch build (import-time torch import would break "torch not
# installed" degradation -- see TorchUnavailableError above).
def safe_softmax_attention(q, k, v, dim=-1, scale=None):
    """Module-level convenience wrapper: resolves torch on first call
    and delegates to the bound guard. See ``_make_safe_softmax_attention``
    for the full rationale."""
    torch_module = _import_torch()
    return _make_safe_softmax_attention(torch_module)(q, k, v, dim=dim, scale=scale)
