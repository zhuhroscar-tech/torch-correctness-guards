"""Direct unit tests for _worker.py's pure helper functions and CLI
argument handling -- exercised in-process since these paths do not
themselves risk a crash (only the actual torch.compile call inside
_run_bare/_run_guarded does, which is why those are only exercised via
subprocess in test_core.py, never imported and called directly here)."""
from __future__ import annotations

import json

import pytest

torch = pytest.importorskip("torch")

from torch_correctness_guards.guards._tiled_reduction_tail_store_worker import (  # noqa: E402
    _make_input,
    _reference_fn,
    main,
)


def test_reference_fn_matches_expected_shape():
    fn = _reference_fn(torch)
    x = _make_input(torch, 64)
    out = fn(x)
    assert out.shape == (64, 64, 64, 1)


def test_make_input_is_deterministic():
    x1 = _make_input(torch, 66)
    x2 = _make_input(torch, 66)
    assert torch.equal(x1, x2)
    assert x1.is_contiguous(memory_format=torch.channels_last)


def test_main_rejects_unknown_mode():
    with pytest.raises(SystemExit):
        main(["nonsense", "64"])


def test_main_bare_mode_prints_json(capsys):
    main(["bare", "64"])
    out = capsys.readouterr().out.strip()
    payload = json.loads(out.splitlines()[-1])
    assert payload["size"] == 64
    assert "max_abs_diff" in payload


def test_main_guarded_mode_prints_json(capsys):
    main(["guarded", "64"])
    out = capsys.readouterr().out.strip()
    payload = json.loads(out.splitlines()[-1])
    assert payload["size"] == 64
    assert payload["raised"] is False
