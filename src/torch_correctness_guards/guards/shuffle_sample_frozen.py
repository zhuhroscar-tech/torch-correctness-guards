"""torch-compile-shuffle-sample-frozen-guard core: detect and guard a real
Dynamo tracing defect where ``random.shuffle(lst)`` and
``random.sample(population, k)``, called from inside a
``torch.compile``'d function, are baked into the graph as trace-time
constants instead of being re-executed on every call.

Upstream reference: pytorch/pytorch#197085 ("[dynamo] random.shuffle /
random.sample inside a compiled function return the same permutation
on every call (eager advances the RNG)"), open and triaged (labels:
triaged, module: correctness (silent), oncall: pt2, module: dynamo,
release triage) as of this guard's creation (2026-09-19) --
independently re-checked via `gh api`, never trusted from a cached
issue summary alone.

The bug, reproduced from scratch on this host (torch 2.14.0, macOS
arm64 CPU, backend="eager"; see README for the exact commands): every
sibling ``random`` function Dynamo supports (``random.random``,
``randint``, ``randrange``, ``uniform``, ``choice``, ``choices``,
``gauss``) correctly re-executes on each compiled call via Dynamo's
``call_random_fn`` path, advancing the global RNG exactly like eager.
Only ``shuffle`` and ``sample`` have no run-time counterpart in
Dynamo's ``RandomVariable``: the permutation / sampled indices are
computed once, at trace time, against a copy of the RNG state, and
baked into the graph as a constant ``ListVariable``. Every call after
the first (tracing) call silently replays that same frozen result --
no error, no warning, no graph break.

This module's guards, ``safe_shuffle`` and ``safe_sample``, wrap the
real ``random.shuffle``/``random.sample`` in a function decorated with
``torch._dynamo.disable``, forcing Dynamo to graph-break around the
call so it always executes eagerly -- exactly the same call-site
workaround pattern the upstream issue itself names as one of the two
acceptable outcomes ("behave like eager, or Dynamo graph-breaks on
them"). The guard does not patch PyTorch; callers replace their
``random.shuffle``/``random.sample`` call sites explicitly.
"""
from __future__ import annotations

import dataclasses
import random
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


def make_safe_shuffle(torch_module):
    """Build a guard function, bound to a specific torch module, that
    forces ``random.shuffle`` to graph-break out of any enclosing
    ``torch.compile`` region via ``torch._dynamo.disable``, so it
    always executes eagerly (fresh permutation per call, real RNG
    advance) instead of being frozen at trace time. Returns a callable
    ``(lst) -> lst`` matching ``random.shuffle``'s own in-place
    signature and return value (``None``... but this guard returns the
    list for convenience; see README for the exact contract)."""

    def _safe_shuffle(lst):
        random.shuffle(lst)
        return lst

    return torch_module._dynamo.disable(_safe_shuffle)


def make_safe_sample(torch_module):
    """Build a guard function, bound to a specific torch module, that
    forces ``random.sample`` to graph-break out of any enclosing
    ``torch.compile`` region via ``torch._dynamo.disable``, so it
    always executes eagerly. Returns a callable
    ``(population, k) -> list`` matching ``random.sample``'s own
    signature and return value."""

    def _safe_sample(population, k):
        return random.sample(population, k)

    return torch_module._dynamo.disable(_safe_sample)


@dataclasses.dataclass
class RandomFreezeCase:
    kind: str
    description: str
    eager_sequence: List[Any]
    native_compiled_sequence: List[Any]
    guarded_compiled_sequence: List[Any]
    eager_shows_real_variation: bool
    native_frozen_after_first_call: bool
    guard_matches_eager: bool
    guard_correct: bool


def _run_shuffle_case(torch_module, seed: int, calls: int = 3) -> RandomFreezeCase:
    # NOTE: the compiled functions below take a tensor argument AND
    # perform a real tensor op on it (``x + 1``), exactly like the
    # upstream issue's own reproducer. This is load-bearing: without
    # an actual tensor op, Dynamo has nothing to build a graph around
    # and does NOT reproduce the trace-time freeze on this host
    # (confirmed by hand before writing this function this way) --
    # only this shape matches pytorch/pytorch#197085's reported
    # behavior byte-for-byte.
    template = [1, 2, 3, 4, 5]

    def native_fn(x):
        lst = list(template)
        random.shuffle(lst)
        _ = x + 1
        return lst

    safe_shuffle = make_safe_shuffle(torch_module)

    def guarded_fn(x):
        lst = list(template)
        safe_shuffle(lst)
        _ = x + 1
        return lst

    compiled_native = torch_module.compile(native_fn, backend="eager")
    compiled_guarded = torch_module.compile(guarded_fn, backend="eager")
    zero = torch_module.zeros(1)

    random.seed(seed)
    eager_sequence = [native_fn(zero) for _ in range(calls)]

    random.seed(seed)
    native_compiled_sequence = [compiled_native(zero) for _ in range(calls)]

    random.seed(seed)
    guarded_compiled_sequence = [compiled_guarded(zero) for _ in range(calls)]

    return _build_case(
        kind="shuffle",
        description="random.shuffle(list-of-5) inside torch.compile(backend='eager')",
        eager_sequence=eager_sequence,
        native_compiled_sequence=native_compiled_sequence,
        guarded_compiled_sequence=guarded_compiled_sequence,
    )


def _run_sample_case(torch_module, seed: int, calls: int = 3) -> RandomFreezeCase:
    # See the tensor-op note in _run_shuffle_case -- the same
    # constraint applies here.
    population = list(range(100))

    def native_fn(x):
        s = random.sample(population, 2)
        _ = x + 1
        return s

    safe_sample = make_safe_sample(torch_module)

    def guarded_fn(x):
        s = safe_sample(population, 2)
        _ = x + 1
        return s

    compiled_native = torch_module.compile(native_fn, backend="eager")
    compiled_guarded = torch_module.compile(guarded_fn, backend="eager")
    zero = torch_module.zeros(1)

    random.seed(seed)
    eager_sequence = [native_fn(zero) for _ in range(calls)]

    random.seed(seed)
    native_compiled_sequence = [compiled_native(zero) for _ in range(calls)]

    random.seed(seed)
    guarded_compiled_sequence = [compiled_guarded(zero) for _ in range(calls)]

    return _build_case(
        kind="sample",
        description="random.sample(range(100), 2) inside torch.compile(backend='eager')",
        eager_sequence=eager_sequence,
        native_compiled_sequence=native_compiled_sequence,
        guarded_compiled_sequence=guarded_compiled_sequence,
    )


def _build_case(
    kind: str,
    description: str,
    eager_sequence: Sequence[Any],
    native_compiled_sequence: Sequence[Any],
    guarded_compiled_sequence: Sequence[Any],
) -> RandomFreezeCase:
    eager_list = list(eager_sequence)
    native_list = list(native_compiled_sequence)
    guarded_list = list(guarded_compiled_sequence)

    eager_shows_real_variation = len({tuple(v) for v in eager_list}) > 1
    # "Frozen after first call": every call from the second onward is
    # byte-identical to the first (the trace-time constant being
    # replayed), even though the first (tracing) call is correct.
    native_frozen_after_first_call = len(native_list) > 1 and all(
        v == native_list[0] for v in native_list[1:]
    )
    guard_matches_eager = guarded_list == eager_list
    # The guard is only "correct" if it actually matches eager on a
    # case where the native path would otherwise have frozen -- a
    # guard that trivially matches on a non-varying case proves
    # nothing, so guard_correct also requires eager to show real
    # variation (ruling out a no-op guard coincidentally passing).
    guard_correct = guard_matches_eager and eager_shows_real_variation

    return RandomFreezeCase(
        kind=kind,
        description=description,
        eager_sequence=eager_list,
        native_compiled_sequence=native_list,
        guarded_compiled_sequence=guarded_list,
        eager_shows_real_variation=eager_shows_real_variation,
        native_frozen_after_first_call=native_frozen_after_first_call,
        guard_matches_eager=guard_matches_eager,
        guard_correct=guard_correct,
    )


def diagnose(seed: int = 7, calls: int = 3) -> Dict[str, Any]:
    """Reproduce the shuffle/sample trace-time-freeze bug from scratch
    against the currently installed torch build (backend='eager'), and
    verify ``make_safe_shuffle``/``make_safe_sample``'s
    ``torch._dynamo.disable``-based guards restore eager semantics.
    Never trusts a cached/prior result -- every call re-runs the
    actual repro, byte-for-byte comparable to the upstream issue's own
    reported output when reproduced with the same seed."""
    torch_module = _import_torch()

    cases = [
        _run_shuffle_case(torch_module, seed=seed, calls=calls),
        _run_sample_case(torch_module, seed=seed, calls=calls),
    ]

    any_native_frozen = any(c.native_frozen_after_first_call for c in cases)
    guard_fully_correct = all(c.guard_correct for c in cases)

    return {
        "torch_version": torch_module.__version__,
        "issue_url": "https://github.com/pytorch/pytorch/issues/197085",
        "seed": seed,
        "calls": calls,
        "cases": [dataclasses.asdict(c) for c in cases],
        "any_native_frozen": any_native_frozen,
        "guard_fully_correct": guard_fully_correct,
    }


# Public convenience wrappers: resolve torch lazily so importing this
# module without torch installed doesn't crash (matching the sibling
# guard repos' degradation pattern).
def safe_shuffle(lst):
    """Module-level convenience wrapper around ``make_safe_shuffle``:
    resolves torch on first call, shuffles ``lst`` in place, and
    returns it. Forces a Dynamo graph-break when called from inside a
    ``torch.compile``'d region so the shuffle always executes eagerly.
    See ``make_safe_shuffle``'s docstring for the full rationale."""
    torch_module = _import_torch()
    return make_safe_shuffle(torch_module)(lst)


def safe_sample(population, k):
    """Module-level convenience wrapper around ``make_safe_sample``:
    resolves torch on first call. Forces a Dynamo graph-break when
    called from inside a ``torch.compile``'d region so the sample
    always executes eagerly. See ``make_safe_sample``'s docstring for
    the full rationale."""
    torch_module = _import_torch()
    return make_safe_sample(torch_module)(population, k)
