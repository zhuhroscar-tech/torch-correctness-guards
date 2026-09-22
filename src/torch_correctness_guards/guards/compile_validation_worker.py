"""Subprocess worker: run ONE fixture's eager/compiled/guarded check in
an isolated child process.

Discovered while building this tool: tracing a
``torch.distributions``-based function through ``torch.compile`` can
flip ``Distribution._validate_args`` -- a PROCESS-GLOBAL flag -- as a
side effect, silently disabling eager mode's own domain validation
(e.g. Categorical's zero-batch reshape-ambiguity check) for every
*subsequent* eager call in that same process, not just the compiled
one. Running each fixture in its own subprocess means one fixture's
``torch.compile`` call can never contaminate another fixture's eager
baseline -- this is the only reliable way to measure 10 independent
eager/compiled/guarded comparisons without cross-fixture state leaks
(matching this fleet's established subprocess-isolation pattern for
host/state-dependent torch.compile behavior).

Invoked as ``python -m torch_correctness_guards.guards.compile_validation_worker <fixture_name>``.
"""
from __future__ import annotations

import json
import sys


def _build_fixture(torch_module, fixture_name: str):
    from .compile_validation import _fixture_specs

    specs = {spec["name"]: spec for spec in _fixture_specs(torch_module)}
    if fixture_name not in specs:
        raise SystemExit(f"unknown fixture: {fixture_name!r} (known: {sorted(specs)})")
    return specs[fixture_name]


def _call_capturing(fn, *args):
    try:
        fn(*args)
        return False, None
    except Exception as exc:  # noqa: BLE001 - intentionally broad, this IS the check
        return True, f"{type(exc).__name__}: {exc}"


def run_fixture_isolated(fixture_name: str) -> dict:
    import torch

    spec = _build_fixture(torch, fixture_name)
    fn = spec["eager_fn_factory"](torch)
    args = spec["args"]

    eager_raised, eager_error = _call_capturing(fn, *args)

    torch._dynamo.reset()
    compiled_fn = torch.compile(fn, fullgraph=True)
    compiled_raised, compiled_error = _call_capturing(compiled_fn, *args)

    torch._dynamo.reset()
    compiled_fn_for_guard = torch.compile(fn, fullgraph=True)
    guarded_fn = spec["safe_wrapper_factory"](compiled_fn_for_guard)
    guarded_raised, guarded_error = _call_capturing(guarded_fn, *args)

    return {
        "name": fixture_name,
        "issue_url": spec["issue_url"],
        "is_boundary_case": spec["is_boundary_case"],
        "torch_version": torch.__version__,
        "eager_raised": eager_raised,
        "eager_error": eager_error,
        "compiled_raised": compiled_raised,
        "compiled_error": compiled_error,
        "guarded_raised": guarded_raised,
        "guarded_error": guarded_error,
    }


def main(argv=None) -> int:
    import os
    import tempfile

    argv = sys.argv[1:] if argv is None else argv
    if not argv:
        raise SystemExit("usage: python -m torch_compile_validation_guard._worker <fixture_name>")
    os.environ["TORCHINDUCTOR_CACHE_DIR"] = tempfile.mkdtemp(
        prefix="validation-guard-worker-"
    )
    result = run_fixture_isolated(argv[0])
    print(json.dumps(result))
    return 0


if __name__ == "__main__":
    sys.exit(main())
