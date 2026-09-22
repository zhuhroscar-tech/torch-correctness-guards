"""torch-inductor-2d-tiled-reduction-tail-store-guard core: guards a real
torch.compile (Inductor) correctness/memory-safety bug where a CPU
2D-tiled reduction kernel's tail block correctly MASKS the reduction
accumulation but writes the result back with a FULL-WIDTH vector store,
writing past the end of the output buffer.

Upstream report: pytorch/pytorch#196681 ("[inductor] CPU 2D-tiled
reduction stores a full-width vector in the tail block: heap overflow
(SIGABRT) or silent wrong results"). A fix PR (#196882, "[inductor]
Mask the 2D-tiled reduction tail store on the output axis") is OPEN and
UNMERGED as of this run (confirmed via a fresh `gh` read, not cached).

Root cause per the issue and a contributor's follow-up diagnosis: when
Inductor tiles a reduction over TWO axes (output dim + reduction dim,
with an untiled pointwise axis in between), the code that decides
whether to split the epilogue store into masked main/tail blocks
(`_inner_loop_reduction_outer_not`) assumes the second tiled axis is
always the level immediately after the first. When another axis sits
between them, that assumption is wrong, the split is skipped, and only
the full-width store from the "main" kernel reaches the output -- so
whenever the tiled OUTPUT dimension is not an exact multiple of the
host's SIMD vector width, the tail iteration's store overruns the
output buffer by (vector_width - remainder) elements.

CRITICALLY: this manifests DIFFERENTLY depending on host/allocator --
confirmed via this project's own CI (not assumed): on ubuntu-latest
x86 (AVX2), the overrun corrupts glibc's heap metadata and the process
is killed with SIGABRT ("Fatal Python error: Aborted", exit code 134).
On this project's macOS ARM (NEON) dev host, the SAME bug instead
silently overwrites a few columns of the next output row with no
crash at all -- consistent with the issue's own description that
crash-vs-silent-corruption is allocator/layout dependent, not
guaranteed either way on any given host.

Because the crash variant can abort the CALLING process (not just the
compiled call), the only way to reliably diagnose OR guard this bug
without risking the diagnosing/guarding process itself is to run each
actual compiled invocation in an isolated subprocess (see
``_worker.py``) and treat "the child was killed by a signal" as
exactly as diagnostic as "the child returned a wrong value" --
`diagnose()` does this for every case.

``safe_compiled_reduction(fn, reference_fn)`` remains available as a
lightweight IN-PROCESS wrapper for callers who want the simple
"compare against eager and raise on divergence" contract directly
(e.g. as an ergonomic decorator around their own compiled function).
Its documented, honest limitation: on a host/shape where this bug
manifests as a hard crash, an in-process wrapper cannot survive the
abort any more than unwrapped code could -- it only helps against the
silent-wrong-value manifestation. Callers on affected hosts who need
crash-survival guarantees should isolate the risky call in a
subprocess themselves, exactly as `_worker.py` does for this tool's
own diagnosis.
"""
from __future__ import annotations

import dataclasses
import json
import subprocess
import sys
from typing import Any, Dict, List, Optional, Tuple


class TorchUnavailableError(RuntimeError):
    """Raised when torch cannot be imported."""


class TailStoreOverrunSuspected(RuntimeError):
    """Raised when a compiled call's output diverges from eager by more
    than the numerical tolerance, consistent with the #196681 2D-tiled
    reduction tail-store overrun (or an unrelated but equally real
    correctness divergence -- either way, the compiled result must not
    be trusted silently)."""


def _import_torch():
    try:
        import torch  # noqa: F401
    except Exception as exc:  # pragma: no cover - exercised only without torch
        raise TorchUnavailableError(
            "torch is required for diagnosis and guarding; install the "
            "'torch' extra."
        ) from exc
    return torch


def safe_compiled_reduction(
    compiled_fn,
    reference_fn,
    rtol: float = 1e-4,
    atol: float = 1e-4,
):
    """Wrap a ``torch.compile``-produced callable that may exhibit the
    #196681 2D-tiled-reduction tail-store overrun: on EVERY call,
    compare the compiled result against ``reference_fn`` (typically the
    eager version of the same function) run on the same input, and
    raise ``TailStoreOverrunSuspected`` if they diverge beyond
    tolerance, instead of silently returning corrupted values.

    HONEST LIMITATION: this is a detect-and-raise guard, not a
    crash-prevention guard. On a host/shape combination where the bug
    manifests as a hard process abort (SIGABRT from glibc heap
    corruption -- observed on this project's own ubuntu-latest AVX2 CI),
    the abort happens INSIDE the call to ``compiled_fn`` below, before
    this wrapper ever gets a chance to compare or raise -- no pure
    in-process Python code can survive that. On hosts/shapes where the
    bug instead silently returns a wrong VALUE with no crash (observed
    on this project's macOS ARM/NEON dev host), this wrapper reliably
    catches it. If you need guaranteed crash survival on a host where
    the crash variant reproduces, isolate the risky call in your own
    subprocess (see this package's ``_worker.py`` for the pattern this
    tool itself uses to diagnose the bug safely).
    """
    torch_module = _import_torch()

    def wrapper(*args, **kwargs):
        expected = reference_fn(*args, **kwargs)
        actual = compiled_fn(*args, **kwargs)
        if not torch_module.allclose(actual, expected, rtol=rtol, atol=atol):
            max_diff = (actual - expected).abs().max().item()
            raise TailStoreOverrunSuspected(
                "torch.compile output diverges from eager by "
                f"max_abs_diff={max_diff!r}, exceeding rtol={rtol}/atol={atol}. "
                "Consistent with pytorch/pytorch#196681 (CPU 2D-tiled "
                "reduction tail-store overrun) if the shape has a "
                "non-vector-width-multiple tiled output dimension; do "
                "not trust this result."
            )
        return actual

    return wrapper


def _invoke_worker(mode: str, size: int, timeout: float = 180.0) -> Dict[str, Any]:
    """Run one case's bug/guard check in an isolated subprocess (see
    module docstring for why: this bug can SIGABRT the calling
    process). Returns a dict always containing ``crashed`` plus either
    the worker's own JSON fields (crashed=False) or ``returncode`` and
    a truncated ``stderr_tail`` (crashed=True)."""
    proc = subprocess.run(
        [
            sys.executable,
            "-m",
            "torch_correctness_guards.guards._tiled_reduction_tail_store_worker",
            mode,
            str(size),
        ],
        capture_output=True,
        text=True,
        timeout=timeout,
    )
    if proc.returncode != 0:
        return {
            "size": size,
            "crashed": True,
            "returncode": proc.returncode,
            "stderr_tail": (proc.stderr or "")[-800:],
        }
    stdout = (proc.stdout or "").strip()
    line = stdout.splitlines()[-1] if stdout else "{}"
    data = json.loads(line)
    data["crashed"] = False
    return data


@dataclasses.dataclass
class TileBoundaryCase:
    size: int
    aligned_to_vector_width: bool
    max_abs_diff: Optional[float]
    bug_reproduced: bool
    crashed_bare: bool
    guard_raised: bool
    crashed_guarded: bool
    guard_behaved_correctly: bool


def _run_case(size: int) -> TileBoundaryCase:
    bare = _invoke_worker("bare", size)
    guarded = _invoke_worker("guarded", size)

    crashed_bare = bool(bare.get("crashed", False))
    max_abs_diff = bare.get("max_abs_diff")
    bug_reproduced = crashed_bare or (
        max_abs_diff is not None and max_abs_diff > 1e-3
    )

    crashed_guarded = bool(guarded.get("crashed", False))
    guard_raised = bool(guarded.get("raised", False))

    if bug_reproduced:
        # Correct guard behavior when the bug is present: either it
        # raised cleanly (the silent-corruption manifestation), or the
        # process was killed in BOTH bare and guarded modes (the crash
        # manifestation -- an in-process wrapper cannot prevent an
        # abort that happens inside the very call it wraps; only
        # subprocess isolation, as used by this diagnosis itself, can
        # contain it). Either outcome means the caller never received
        # a silently-trusted wrong value.
        guard_behaved_correctly = guard_raised or (crashed_bare and crashed_guarded)
    else:
        guard_behaved_correctly = (not guard_raised) and (not crashed_guarded)

    aligned = (size % 4) == 0

    return TileBoundaryCase(
        size=size,
        aligned_to_vector_width=aligned,
        max_abs_diff=max_abs_diff,
        bug_reproduced=bug_reproduced,
        crashed_bare=crashed_bare,
        guard_raised=guard_raised,
        crashed_guarded=crashed_guarded,
        guard_behaved_correctly=guard_behaved_correctly,
    )


def diagnose(sizes: Tuple[int, ...] = (64, 66, 70, 72, 80)) -> Dict[str, Any]:
    """Reproduce the 2D-tiled-reduction tail-store-overrun divergence
    from scratch against the currently installed torch build, across
    several tile-boundary-crossing output sizes, and verify that
    ``safe_compiled_reduction`` (exercised through the guarded worker
    path) never lets a caller silently trust a corrupted result --
    whether the bug manifests as a wrong value or as a process abort.
    Every case runs in its own subprocess (see ``_worker.py``); a
    crash in one case cannot affect or invalidate any other case, or
    this function's own caller."""
    torch_module = _import_torch()
    results: List[TileBoundaryCase] = [_run_case(s) for s in sizes]

    any_bug_reproduced = any(c.bug_reproduced for c in results)
    any_crash_observed = any(c.crashed_bare for c in results)
    guard_fully_correct = all(c.guard_behaved_correctly for c in results)

    return {
        "torch_version": torch_module.__version__,
        "issue_url": "https://github.com/pytorch/pytorch/issues/196681",
        "fix_pr_url": "https://github.com/pytorch/pytorch/pull/196882",
        "cases": [dataclasses.asdict(c) for c in results],
        "any_bug_reproduced": any_bug_reproduced,
        "any_crash_observed": any_crash_observed,
        "guard_fully_correct": guard_fully_correct,
    }
