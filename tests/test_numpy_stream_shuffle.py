"""Correctness tests for safe_row_shuffle/safe_row_shuffle_ and the
diagnose() function, against real torch -- not mocked.

Design note: since the guard's whole point is "always a true permutation
of the input rows", every guard test checks the ROW MULTISET invariant
(sorted(rows_after) == sorted(rows_before)) rather than any specific
output ordering -- the exact permutation is randomized by design.
"""
from __future__ import annotations

import pytest

torch = pytest.importorskip("torch")

from torch_correctness_guards.guards.numpy_stream_shuffle import (  # noqa: E402
    TorchUnavailableError,
    diagnose,
    safe_row_shuffle,
    safe_row_shuffle_,
)


def _row_multiset(tensor):
    return sorted(tuple(row) for row in tensor.tolist())


# ---------------------------------------------------------------------------
# safe_row_shuffle / safe_row_shuffle_ correctness: row multiset preserved,
# actual reordering happens (not a no-op), reproducible with a generator.
# ---------------------------------------------------------------------------


def test_safe_row_shuffle_preserves_row_multiset():
    x = torch.arange(20).reshape(10, 2)
    before = _row_multiset(x)
    result = safe_row_shuffle(x)
    after = _row_multiset(result)
    assert before == after


def test_safe_row_shuffle_does_not_mutate_input():
    x = torch.arange(20).reshape(10, 2)
    original = x.clone()
    safe_row_shuffle(x)
    assert torch.equal(x, original)


def test_safe_row_shuffle_returns_a_permutation_not_identity_with_high_probability():
    # Not deterministic by nature, but with 10 rows the chance that a
    # genuine random permutation equals the identity is 1/10! -- if this
    # ever fails it is not a false-positive-prone flake in practice.
    x = torch.arange(20).reshape(10, 2)
    result = safe_row_shuffle(x, generator=torch.Generator().manual_seed(3))
    assert not torch.equal(result, x)
    assert _row_multiset(result) == _row_multiset(x)


def test_safe_row_shuffle_is_reproducible_with_a_seeded_generator():
    x = torch.arange(20).reshape(10, 2)
    g1 = torch.Generator().manual_seed(42)
    g2 = torch.Generator().manual_seed(42)
    r1 = safe_row_shuffle(x, generator=g1)
    r2 = safe_row_shuffle(x, generator=g2)
    assert torch.equal(r1, r2)


def test_safe_row_shuffle_inplace_mutates_and_preserves_multiset():
    x = torch.arange(20).reshape(10, 2)
    before = _row_multiset(x)
    ref = safe_row_shuffle_(x, generator=torch.Generator().manual_seed(5))
    assert ref is x
    after = _row_multiset(x)
    assert before == after


@pytest.mark.parametrize("n_rows,n_cols", [(1, 1), (2, 3), (5, 1), (50, 4)])
def test_safe_row_shuffle_preserves_multiset_at_various_shapes(n_rows, n_cols):
    x = torch.arange(n_rows * n_cols).reshape(n_rows, n_cols)
    before = _row_multiset(x)
    result = safe_row_shuffle(x)
    assert _row_multiset(result) == before


def test_safe_row_shuffle_rejects_0d_tensor():
    x = torch.tensor(5)
    with pytest.raises(ValueError):
        safe_row_shuffle(x)


# ---------------------------------------------------------------------------
# Regression test that would have FAILED before this repo's guard existed:
# demonstrates the real bug via the unguarded torch._numpy.random.shuffle
# stream-mode path (proves this is a genuine, currently-reproducible
# upstream defect, not a hypothetical), then proves the guard closes it
# via diagnose().
# ---------------------------------------------------------------------------


def test_unguarded_stream_mode_shuffle_corrupts_row_multiset_on_this_host():
    """This is the ACTUAL upstream bug (pytorch/pytorch#197795),
    reproduced with the real torch._numpy.random.shuffle under
    use_numpy_random_stream=True -- no guard involved.

    If this ever fails (multiset preserved across ALL scanned
    seeds/shapes), it means upstream PyTorch has silently fixed the
    stream-mode shuffle's row-aliasing corruption, which should be
    investigated and reflected in this test/README, not silenced."""
    import warnings

    import torch._numpy as tnp

    found_corruption = False
    for seed in (0, 1, 2, 3, 4, 5, 6, 7):
        x = torch.arange(12).reshape(6, 2)
        before = _row_multiset(x)
        tnp.random.seed(seed)
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            with torch._dynamo.config.patch(use_numpy_random_stream=True):
                tnp.random.shuffle(x)
        after = _row_multiset(x)
        if before != after:
            found_corruption = True
            break

    assert found_corruption, (
        "Expected the documented row-multiset corruption "
        "(pytorch/pytorch#197795) at at least one of seeds 0-7 on this "
        "host's torch build; if this assertion now fails, the upstream "
        "bug may be fixed on this host's torch version -- verify against "
        "the currently installed torch version and update this "
        "test/README rather than deleting the assertion."
    )


def test_diagnose_reproduces_bug_and_confirms_guard_effective():
    report = diagnose(seeds=(0, 1, 2, 7), shapes=((6, 2),))
    assert report["torch_version"]
    assert len(report["issue_urls"]) == 1
    assert "197795" in report["issue_urls"][0]
    # On any host affected by the upstream bug, at least one case must show
    # the row-multiset corruption in the UNGUARDED path.
    assert report["any_bug_present"] is True
    # The guard must preserve the row multiset in EVERY tested case.
    assert report["guard_fully_effective"] is True
    assert len(report["cases"]) == 4  # 4 seeds x 1 shape


def test_diagnose_default_covers_multiple_seeds_and_shapes():
    report = diagnose()
    shapes = sorted({(c["n_rows"], c["n_cols"]) for c in report["cases"]})
    seeds = sorted({c["seed"] for c in report["cases"]})
    assert shapes == [(6, 2), (10, 3)]
    assert seeds == [0, 1, 2, 7]


def test_torch_unavailable_error_is_distinct_type(monkeypatch):
    import torch_correctness_guards.guards.numpy_stream_shuffle as core_module

    def _boom():
        raise TorchUnavailableError("torch is required for diagnosis and guarding; install the 'torch' extra.")

    monkeypatch.setattr(core_module, "_import_torch", _boom)
    with pytest.raises(TorchUnavailableError):
        core_module.diagnose()
