"""Subprocess worker: run ONE case's real bug/guard check in an
isolated child process.

pytorch/pytorch#196681 can manifest as a hard process abort (SIGABRT,
glibc heap corruption) depending on allocator layout / SIMD vector
width / host -- observed as an actual crash (exit code 134, core
dumped) on ubuntu-latest x86 AVX2 CI for this project, and as a silent
wrong VALUE with no crash on this project's macOS ARM/NEON dev host.
Running each case in its own subprocess means a crash only kills that
subprocess -- this is the only reliable way to diagnose or test a bug
that can corrupt process memory without taking down the caller
(``diagnose()``, the CLI, or an entire pytest session) with it.

Invoked as ``python -m
torch_correctness_guards.guards._tiled_reduction_tail_store_worker <bare|guarded> <size>``.
Never import this module's functions expecting them to be crash-safe
when called directly in-process -- the whole point is that they run in
a throwaway child.
"""
from __future__ import annotations

import json
import sys


def _reference_fn(torch_module):
    """The issue's own minimal reproducer: GELU on a channels_last
    input, a permute, two mismatched-shape pads, and a matmul -- fused
    by Inductor into one 2D-tiled kernel whose output (tiled) dimension
    can land on a non-vector-width-multiple size."""
    F = torch_module.nn.functional

    def f(x):
        v = (x * 0.5) * (torch_module.erf(x * 0.7071067811865476) + 1)
        size = x.shape[2]
        complement = size - x.shape[1]
        a = v.permute(3, 2, 1, 0)
        p = F.pad(a, (0, 0, 0, complement), value=0.5)
        q = F.pad(v, (0, 0, 0, 0, 0, complement), value=0.5)
        return torch_module.matmul(q, p)

    return f


def _make_input(torch_module, size: int):
    torch_module.manual_seed(0)
    return torch_module.randn(1, 8, size, size).contiguous(
        memory_format=torch_module.channels_last
    )


def _run_bare(size: int) -> dict:
    import torch

    fn = _reference_fn(torch)
    x = _make_input(torch, size)
    expected = fn(x)
    compiled = torch.compile(fn, fullgraph=True)
    actual = compiled(x)  # may SIGABRT the process -- isolated by design
    max_abs_diff = (actual - expected).abs().max().item()
    return {"size": size, "max_abs_diff": max_abs_diff}


def _run_guarded(size: int) -> dict:
    import torch

    from .tiled_reduction_tail_store import TailStoreOverrunSuspected, safe_compiled_reduction

    fn = _reference_fn(torch)
    x = _make_input(torch, size)
    compiled = torch.compile(fn, fullgraph=True)
    guarded = safe_compiled_reduction(compiled, fn)
    try:
        guarded(x)  # may also SIGABRT -- isolated by design
        return {"size": size, "raised": False}
    except TailStoreOverrunSuspected as exc:
        return {"size": size, "raised": True, "message": str(exc)}


def main(argv=None) -> int:
    import os
    import tempfile

    argv = sys.argv[1:] if argv is None else argv
    os.environ["TORCHINDUCTOR_CACHE_DIR"] = tempfile.mkdtemp(
        prefix="tiled-reduction-guard-worker-"
    )
    mode, size_str = argv[0], argv[1]
    size = int(size_str)
    if mode == "bare":
        result = _run_bare(size)
    elif mode == "guarded":
        result = _run_guarded(size)
    else:
        raise SystemExit(f"unknown mode: {mode!r} (expected 'bare' or 'guarded')")
    print(json.dumps(result))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
