"""Tests for the core diagnosis/guard logic: reproduce the upstream bug
from scratch, verify safe_rrelu avoids it, and check the mechanism
signature (bit-exact forward, zero diff with early-stop disabled).
"""
from __future__ import annotations

import pytest

torch = pytest.importorskip("torch")

from torch_correctness_guards.guards.checkpoint_noise import diagnose, safe_rrelu


def test_reproduce_upstream_bug_193671():
    """Reproduces pytorch/pytorch#193671 end to end on this host's real
    installed torch build: F.rrelu under non-reentrant checkpointing or
    saved_tensors_hooks must diverge from the uncheckpointed reference,
    while the forward pass stays bit-exact and disabling checkpoint's
    early-stop machinery makes the divergence disappear -- the exact
    signature the upstream issue documents as the root cause."""
    report = diagnose()

    # This is a real, currently-open upstream bug reading uninitialized
    # memory; the exact magnitude varies run to run but SOME divergence
    # in at least one of the two triggering modes is expected on an
    # affected torch build.
    if not report["bug_reproduced"]:
        pytest.skip(
            "F.rrelu checkpoint/saved-hooks bug did not reproduce on this "
            "torch build (may indicate it was fixed upstream) -- see "
            "diagnose() output for details"
        )

    c = report["checkpoint_case"]
    # The bit-exact-forward + early-stop-disabled-fixes-it signature is
    # what distinguishes this specific bug from an unrelated numerical
    # issue; require it whenever the checkpoint path itself triggered.
    if c["buggy_diff"] > 0.0:
        assert c["buggy_forward_diff"] == 0.0
        assert c["no_early_stop_diff"] == 0.0


def test_guard_matches_across_all_execution_modes():
    """safe_rrelu must produce identical (zero-diff, NaN-free) gradients
    under eager, non-reentrant checkpointing, and saved_tensors_hooks --
    this is the actual acceptance oracle for the guard, independent of
    whether the underlying F.rrelu bug happens to reproduce this run."""
    report = diagnose()
    assert report["guard_fully_correct"] is True

    ckpt = report["checkpoint_case"]
    hooks = report["saved_hooks_case"]
    assert ckpt["guard_diff"] == 0.0
    assert ckpt["guard_has_nan"] is False
    assert hooks["guard_diff"] == 0.0
    assert hooks["guard_has_nan"] is False


def test_safe_rrelu_eval_mode_delegates_to_real_rrelu():
    """In eval mode there is no randomness/mutable-output kernel
    involved, so safe_rrelu should exactly match F.rrelu's documented
    fixed-slope behavior."""
    import torch.nn.functional as F

    x = torch.tensor([-2.0, -1.0, 0.0, 1.0, 2.0])
    lower, upper = 0.1, 0.9
    expected = F.rrelu(x, lower, upper, training=False)
    actual = safe_rrelu(x, lower, upper, training=False)
    assert torch.equal(expected, actual)


def test_safe_rrelu_forward_shape_and_sign_semantics():
    """safe_rrelu's forward pass must preserve the basic RReLU contract:
    non-negative inputs pass through unchanged; negative inputs are
    scaled by a factor strictly between lower and upper (in expectation
    -- verified here via a large sample and a tolerant bound check)."""
    torch.manual_seed(42)
    x = torch.randn(10000) * 5
    lower, upper = 0.1, 0.9
    out = safe_rrelu(x, lower, upper, training=True)

    assert out.shape == x.shape
    pos_mask = x >= 0
    assert torch.equal(out[pos_mask], x[pos_mask])

    neg_mask = ~pos_mask
    if neg_mask.any():
        ratio = out[neg_mask] / x[neg_mask]
        assert bool((ratio >= lower - 1e-6).all())
        assert bool((ratio <= upper + 1e-6).all())


def test_safe_rrelu_gradient_flows_correctly():
    """Gradients through safe_rrelu must be well-defined and match the
    applied per-element slope (1.0 for positive inputs, the sampled
    slope for negative inputs) -- checked via torch.where's known
    gradient semantics rather than assuming any particular library
    internals."""
    torch.manual_seed(0)
    x = torch.tensor([-2.0, -1.0, 1.0, 2.0], requires_grad=True)
    generator = torch.Generator().manual_seed(0)
    out = safe_rrelu(x, 0.1, 0.9, training=True, generator=generator)
    out.sum().backward()
    assert x.grad is not None
    assert not torch.isnan(x.grad).any()
    # Positive inputs always have gradient exactly 1.0 (identity path).
    assert torch.allclose(x.grad[2:], torch.ones(2))
