"""Subprocess worker: run ONE (p, train_arg) fixture's eager/compiled/
guarded native_dropout mask comparison in an isolated child process.

Matches this fleet's established subprocess-isolation pattern (see
torch-compile-validation-guard/_worker.py's module docstring for the
general rationale): running each fixture in its own process means one
fixture's ``torch.compile`` call, or any dynamo/inductor cache state,
can never contaminate a later fixture's eager baseline.

Invoked as:
  python -m torch_correctness_guards.guards._native_dropout_train_none_worker <p> <train_arg>
where <train_arg> is one of the literal strings "None", "True", "False".
"""
from __future__ import annotations

import json
import sys


def _parse_train_arg(raw: str):
    if raw == "None":
        return None
    if raw == "True":
        return True
    if raw == "False":
        return False
    raise SystemExit(f"invalid train arg: {raw!r} (expected None/True/False)")


def run_fixture_isolated(p: float, train_arg) -> dict:
    import torch

    from .native_dropout_train_none import safe_native_dropout

    torch.manual_seed(0)
    n = 2000
    x = torch.ones(n)

    def eager_fn(inp):
        return torch.native_dropout(inp, p, train_arg)[1]

    eager_mask = eager_fn(x)
    eager_mask_all_true = bool(eager_mask.all().item())

    torch._dynamo.reset()
    compiled_fn = torch.compile(eager_fn, backend="inductor", fullgraph=True)
    compiled_mask = compiled_fn(x)
    compiled_mask_all_true = bool(compiled_mask.all().item())

    torch._dynamo.reset()

    def raw_compiled_native_dropout(inp, p_, train_):
        return torch.native_dropout(inp, p_, train_)[1]

    guarded_native_dropout = safe_native_dropout(raw_compiled_native_dropout)

    def guarded_fn(inp):
        compiled_guarded = torch.compile(
            lambda t: guarded_native_dropout(t, p, train_arg),
            backend="inductor",
            fullgraph=True,
        )
        return compiled_guarded(inp)

    guarded_mask = guarded_fn(x)
    guarded_mask_all_true = bool(guarded_mask.all().item())

    return {
        "p": p,
        "train_arg": train_arg,
        "torch_version": torch.__version__,
        "eager_mask_all_true": eager_mask_all_true,
        "compiled_mask_all_true": compiled_mask_all_true,
        "guarded_mask_all_true": guarded_mask_all_true,
    }


def main(argv=None) -> int:
    import os
    import tempfile

    argv = sys.argv[1:] if argv is None else argv
    if len(argv) != 2:
        raise SystemExit(
            "usage: python -m torch_correctness_guards.guards._native_dropout_train_none_worker <p> <train_arg>"
        )
    os.environ["TORCHINDUCTOR_CACHE_DIR"] = tempfile.mkdtemp(
        prefix="native-dropout-guard-worker-"
    )
    p = float(argv[0])
    train_arg = _parse_train_arg(argv[1])
    result = run_fixture_isolated(p, train_arg)
    print(json.dumps(result))
    return 0


if __name__ == "__main__":
    sys.exit(main())
