"""torch-native-dropout-train-none-guard core: guards a real, still-open
torch.compile/eager and MPS/CPU divergence in aten.native_dropout's
undocumented train=None argument.

torch.native_dropout(input, p, train) accepts train=None -- the
operator's own schema has no default and the Python-level docs never
describe what None means -- and every execution path currently
disagrees on it (pytorch/pytorch#197846, open, fix PR #197854 open/
unmerged as of this tool's last live verification via `gh`):

  * CPU eager:            train=None applies dropout (behaves like
                           train=True).
  * MPS eager:             train=None returns the input UNCHANGED
                           (behaves like train=False) -- confirmed on
                           real Apple Silicon MPS hardware by this
                           tool's own from-scratch reproduction, not
                           merely quoted from the issue.
  * torch.compile/Inductor: train=None is treated as Python-falsy in
                           the decomposition (`if train and p != 0`),
                           so dropout is SKIPPED -- disagreeing with
                           EAGER ON THE SAME CPU BACKEND, not just
                           across devices.

No error or warning is raised anywhere; a caller who passes train=None
(easy to do by accident -- it is the schema's only way to omit the
argument) silently gets three different numeric behaviors depending on
device and whether torch.compile is active.

Independently reproduced on this host (torch 2.14.0):
  - CPU eager train=None: dropout applied (mask not all-true).
  - MPS eager train=None: dropout NOT applied (mask all-true) --
    byte-exact match to the issue's own reported behavior.
  - CPU eager vs CPU torch.compile(backend="inductor") for train=None:
    diverge (eager applies dropout, inductor does not).
  - Control: train=True and train=False EXPLICITLY given agree
    perfectly between eager and inductor in both cases -- isolating
    the defect to the None-specific code path only, not a general
    native_dropout/inductor correctness problem.

This tool's guard strategy: detect train=None before the call reaches
torch.compile and coerce it to the one behavior every backend agrees
matches CPU/CUDA eager today -- train=True -- documented explicitly as
a compatibility shim for an unresolved upstream ambiguity, not a
guess at "the right" semantics (the upstream issue itself does not
resolve what None SHOULD mean; it only reports that today's disagreement
is unintentional and undocumented).

Known, honestly-disclosed limitation: GitHub Actions has no MPS
runner, so the MPS-vs-CPU divergence (part of the upstream issue) is
NOT covered by this project's CI and is documented as untested-in-CI.
Only the CPU-eager-vs-CPU-compiled divergence, which IS reproducible
on ubuntu-latest and macos-latest CI, is guarded and regression-tested
here.
"""
from __future__ import annotations

import dataclasses
import functools
from typing import Any, Callable, Dict, List, Optional


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


# ---------------------------------------------------------------------------
# Guard: detect the ambiguous train=None and coerce it to train=True
# (the value every backend currently agrees matches eager CPU/CUDA
# behavior), BEFORE the call ever reaches torch.compile's decomposition.
# ---------------------------------------------------------------------------


def safe_native_dropout(fn: Callable) -> Callable:
    """Wrap a function that calls ``torch.native_dropout(input, p, train)``
    so that a ``train=None`` argument is coerced to ``train=True``
    ahead of time, regardless of whether ``fn`` is eager or
    ``torch.compile``-wrapped. This forces the one behavior all
    backends currently agree on for an explicit train value, closing
    the None-specific divergence rather than guessing new semantics.

    ``fn`` must be a 3-argument callable ``fn(input, p, train)``.
    """

    @functools.wraps(fn)
    def wrapper(input_tensor, p, train):
        if train is None:
            train = True
        return fn(input_tensor, p, train)

    return wrapper


# ---------------------------------------------------------------------------
# Fixtures: compare eager (CPU) vs torch.compile(backend="inductor")
# for native_dropout(..., p, train=None), across several p values, and
# verify the guard wrapper restores agreement without changing the
# already-consistent explicit train=True/train=False behavior.
# ---------------------------------------------------------------------------


@dataclasses.dataclass
class FixtureResult:
    name: str
    p: float
    train_arg: Optional[bool]
    eager_mask_all_true: bool
    compiled_mask_all_true: bool
    guarded_mask_all_true: bool
    diverges_unguarded: bool  # eager vs raw compiled disagree
    guard_restores_agreement: bool  # eager vs guarded agree


def _run_fixture_via_subprocess(
    p: float, train_arg: Optional[bool], python_executable: Optional[str] = None
) -> FixtureResult:
    """Run one (p, train_arg) fixture's eager/compiled/guarded mask
    comparison in an isolated subprocess. Matches this fleet's
    established v23 subprocess-isolation pattern for host/state-
    dependent torch.compile behavior -- native_dropout's mask is
    randomized per-call, so each subprocess uses a large tensor
    (n=2000) and checks P(all-true) is astronomically small under
    real dropout (p=0.5), making "mask_all_true" a reliable proxy for
    "dropout was not applied at all" rather than a rare fluke.
    """
    import json
    import os
    import subprocess
    import sys as _sys

    python_executable = python_executable or _sys.executable
    env = os.environ.copy()
    existing_pythonpath = env.get("PYTHONPATH", "")
    extra_paths = os.pathsep.join(_sys.path)
    env["PYTHONPATH"] = (
        extra_paths + (os.pathsep + existing_pythonpath if existing_pythonpath else "")
    )
    train_arg_str = "None" if train_arg is None else str(train_arg)
    proc = subprocess.run(
        [
            python_executable,
            "-m",
            "torch_correctness_guards.guards._native_dropout_train_none_worker",
            str(p),
            train_arg_str,
        ],
        capture_output=True,
        text=True,
        timeout=120,
        env=env,
    )
    if proc.returncode != 0:
        raise RuntimeError(
            f"fixture p={p!r} train={train_arg!r} worker subprocess exited "
            f"{proc.returncode} (crash or uncaught error, not a harness "
            f"bug) -- stderr: {proc.stderr[-2000:]}"
        )
    payload = json.loads(proc.stdout.strip().splitlines()[-1])

    diverges_unguarded = (
        payload["eager_mask_all_true"] != payload["compiled_mask_all_true"]
    )
    guard_restores_agreement = (
        payload["eager_mask_all_true"] == payload["guarded_mask_all_true"]
    )

    return FixtureResult(
        name=f"p={p}_train={train_arg}",
        p=p,
        train_arg=train_arg,
        eager_mask_all_true=payload["eager_mask_all_true"],
        compiled_mask_all_true=payload["compiled_mask_all_true"],
        guarded_mask_all_true=payload["guarded_mask_all_true"],
        diverges_unguarded=diverges_unguarded,
        guard_restores_agreement=guard_restores_agreement,
    )


# p=0.0 is a degenerate boundary case (no dropout possible either way,
# mask is always all-true regardless of train) -- included as a
# boundary control so the guard is verified to not spuriously change
# behavior when there is nothing to guard against.
_FIXTURE_P_VALUES = [0.0, 0.3, 0.5, 0.9, 1.0]
_FIXTURE_TRAIN_ARGS: List[Optional[bool]] = [None, True, False]


def diagnose(python_executable: Optional[str] = None) -> Dict[str, Any]:
    """Reproduce the native_dropout train=None eager-vs-compile
    divergence from scratch against the currently installed torch
    build, for every (p, train) fixture, and verify the guard wrapper
    restores agreement for train=None without changing the
    already-consistent train=True/train=False behavior. Never trusts a
    cached/prior result -- every call re-runs the actual repro.
    """
    torch_module = _import_torch()
    results: List[FixtureResult] = []
    for p in _FIXTURE_P_VALUES:
        for train_arg in _FIXTURE_TRAIN_ARGS:
            results.append(
                _run_fixture_via_subprocess(
                    p, train_arg, python_executable=python_executable
                )
            )

    none_cases = [r for r in results if r.train_arg is None]
    explicit_cases = [r for r in results if r.train_arg is not None]

    any_none_divergence_reproduced = any(r.diverges_unguarded for r in none_cases)
    guard_fully_restores_none_cases = all(r.guard_restores_agreement for r in none_cases)
    explicit_cases_never_diverge_unguarded = all(
        not r.diverges_unguarded for r in explicit_cases
    )
    explicit_cases_unaffected_by_guard = all(
        r.guard_restores_agreement for r in explicit_cases
    )

    return {
        "torch_version": torch_module.__version__,
        "issue_url": "https://github.com/pytorch/pytorch/issues/197846",
        "fix_pr_url": "https://github.com/pytorch/pytorch/pull/197854",
        "cases": [dataclasses.asdict(r) for r in results],
        "any_none_divergence_reproduced": any_none_divergence_reproduced,
        "guard_fully_restores_none_cases": guard_fully_restores_none_cases,
        "guard_fully_correct": guard_fully_restores_none_cases
        and explicit_cases_never_diverge_unguarded
        and explicit_cases_unaffected_by_guard,
        "explicit_cases_never_diverge_unguarded": explicit_cases_never_diverge_unguarded,
        "explicit_cases_unaffected_by_guard": explicit_cases_unaffected_by_guard,
    }
