"""Regression tests for torch-embeddingbag-freq-scale-guard.

These prove:
  1. The bug is real and reproducible from scratch on this host's
     installed torch build WHEN a functional MPS device is present:
     torch.nn.functional.embedding_bag(..., scale_grad_by_freq=True)
     silently returns the UNSCALED gradient on MPS, while CPU
     correctly divides by index frequency. Skipped (not silently
     passed) when MPS isn't functional on the running host --
     honestly distinguishing "not tested" from "passed".
  2. make_safe_embedding_bag is an independently-verified fix: its
     gradient matches the CPU oracle on every device it runs on,
     including MPS when available, and including the CPU path itself
     (no regression on the already-correct backend).
  3. The scale_grad_by_freq=False path is delegated unchanged (no
     double-guarding of an already-correct code path) on every
     device.
  4. mps_is_functional() distinguishes a genuine usable MPS device
     from `torch.backends.mps.is_available()` alone (which is known
     to be True-but-non-functional on virtualized CI runners).
"""
from __future__ import annotations

import pytest

torch = pytest.importorskip("torch")

from torch_correctness_guards.guards.embeddingbag_freq_scale import (
    diagnose,
    make_safe_embedding_bag,
    mps_is_functional,
    safe_embedding_bag,
)


def _mps_functional() -> bool:
    return mps_is_functional(torch)


class TestMpsFunctionalProbe:
    def test_mps_is_functional_returns_bool(self):
        result = mps_is_functional(torch)
        assert isinstance(result, bool)

    def test_mps_is_functional_false_when_backend_reports_unavailable(self, monkeypatch):
        class FakeMpsBackend:
            @staticmethod
            def is_available():
                return False

        monkeypatch.setattr(torch.backends, "mps", FakeMpsBackend(), raising=False)
        assert mps_is_functional(torch) is False

    def test_mps_is_functional_false_when_allocation_probe_raises(self, monkeypatch):
        class FakeMpsBackend:
            @staticmethod
            def is_available():
                return True

        def _raise_tensor(*args, **kwargs):
            if kwargs.get("device") == "mps":
                raise RuntimeError("simulated virtualized-runner allocation failure")
            return torch.tensor.__wrapped__(*args, **kwargs) if hasattr(torch.tensor, "__wrapped__") else None

        monkeypatch.setattr(torch.backends, "mps", FakeMpsBackend(), raising=False)
        real_tensor = torch.tensor

        def _fake_tensor(data, device=None, **kwargs):
            if device == "mps":
                raise RuntimeError("simulated virtualized-runner allocation failure")
            return real_tensor(data, device=device, **kwargs)

        monkeypatch.setattr(torch, "tensor", _fake_tensor)
        assert mps_is_functional(torch) is False


class TestNativeBugReproductionOnMps:
    @pytest.mark.skipif(not _mps_functional(), reason="MPS not functional on this host")
    def test_native_mps_silently_ignores_scale_grad_by_freq(self):
        # Real repro of pytorch/pytorch#190061 on this host's actual MPS
        # device. Not asserted unconditionally true forever: if a future
        # torch release merges PR #190062 and fixes this upstream, this
        # test will start failing with a clear message -- update the
        # README/ledger accordingly rather than treating that flip as a
        # regression in this guard.
        weight_cpu = torch.zeros(4, 3, requires_grad=True)
        idx_cpu = torch.tensor([1, 1, 0, 2])
        offsets_cpu = torch.tensor([0])
        out_cpu = torch.nn.functional.embedding_bag(
            idx_cpu, weight_cpu, offsets_cpu, mode="sum", scale_grad_by_freq=True
        )
        out_cpu.sum().backward()
        cpu_row1 = weight_cpu.grad[1].tolist()

        weight_mps = torch.zeros(4, 3, device="mps", requires_grad=True)
        idx_mps = torch.tensor([1, 1, 0, 2], device="mps")
        offsets_mps = torch.tensor([0], device="mps")
        out_mps = torch.nn.functional.embedding_bag(
            idx_mps, weight_mps, offsets_mps, mode="sum", scale_grad_by_freq=True
        )
        out_mps.sum().backward()
        mps_row1 = weight_mps.grad[1].detach().cpu().tolist()

        assert cpu_row1 == pytest.approx([1.0, 1.0, 1.0]), (
            "CPU should correctly halve the gradient for index 1 (frequency 2)"
        )
        assert mps_row1 == pytest.approx([2.0, 2.0, 2.0]), (
            "expected the native MPS silent-ignore bug (unscaled gradient == 2.0); "
            "if this now matches CPU (1.0), pytorch/pytorch#190061 may be fixed "
            "upstream -- update the README/ledger accordingly"
        )
        assert cpu_row1 != mps_row1, (
            "this test's whole point is that CPU and MPS silently disagree; "
            "if they now match, the bug is fixed upstream"
        )


class TestGuardCorrectness:
    def test_guard_matches_cpu_oracle_on_cpu(self):
        weight = torch.zeros(4, 3, requires_grad=True)
        idx = torch.tensor([1, 1, 0, 2])
        offsets = torch.tensor([0])
        safe_fn = make_safe_embedding_bag(torch)
        out = safe_fn(idx, weight, offsets, mode="sum", scale_grad_by_freq=True)
        out.sum().backward()
        assert weight.grad[1].tolist() == pytest.approx([1.0, 1.0, 1.0])
        # Index 0 and index 2 each occur once: unscaled (divide by 1).
        assert weight.grad[0].tolist() == pytest.approx([1.0, 1.0, 1.0])
        assert weight.grad[2].tolist() == pytest.approx([1.0, 1.0, 1.0])

    @pytest.mark.skipif(not _mps_functional(), reason="MPS not functional on this host")
    def test_guard_matches_cpu_oracle_on_mps(self):
        weight_cpu = torch.zeros(4, 3, requires_grad=True)
        idx_cpu = torch.tensor([1, 1, 0, 2])
        offsets_cpu = torch.tensor([0])
        safe_fn_cpu = make_safe_embedding_bag(torch)
        out_cpu = safe_fn_cpu(idx_cpu, weight_cpu, offsets_cpu, mode="sum", scale_grad_by_freq=True)
        out_cpu.sum().backward()
        cpu_row1 = weight_cpu.grad[1].tolist()

        weight_mps = torch.zeros(4, 3, device="mps", requires_grad=True)
        idx_mps = torch.tensor([1, 1, 0, 2], device="mps")
        offsets_mps = torch.tensor([0], device="mps")
        safe_fn_mps = make_safe_embedding_bag(torch)
        out_mps = safe_fn_mps(idx_mps, weight_mps, offsets_mps, mode="sum", scale_grad_by_freq=True)
        out_mps.sum().backward()
        mps_row1 = weight_mps.grad[1].detach().cpu().tolist()

        assert mps_row1 == pytest.approx(cpu_row1), (
            "guard must produce the SAME (correct) gradient on MPS as on CPU, "
            "unlike the native op which silently disagrees"
        )
        assert mps_row1 == pytest.approx([1.0, 1.0, 1.0])

    def test_guard_delegates_unchanged_when_scale_grad_by_freq_false(self):
        weight = torch.zeros(4, 3, requires_grad=True)
        idx = torch.tensor([1, 1, 0, 2])
        offsets = torch.tensor([0])
        safe_fn = make_safe_embedding_bag(torch)
        guard_out = safe_fn(idx, weight, offsets, mode="sum", scale_grad_by_freq=False)

        weight2 = torch.zeros(4, 3, requires_grad=True)
        native_out = torch.nn.functional.embedding_bag(
            idx, weight2, offsets, mode="sum", scale_grad_by_freq=False
        )
        assert torch.equal(guard_out, native_out)

    def test_module_level_safe_embedding_bag_wrapper(self):
        weight = torch.zeros(4, 3, requires_grad=True)
        idx = torch.tensor([1, 1, 0, 2])
        offsets = torch.tensor([0])
        out = safe_embedding_bag(idx, weight, offsets, mode="sum", scale_grad_by_freq=True)
        out.sum().backward()
        assert weight.grad[1].tolist() == pytest.approx([1.0, 1.0, 1.0])


class TestDiagnose:
    def test_diagnose_returns_expected_shape(self):
        report = diagnose()
        assert "torch_version" in report
        assert "mps_functional" in report
        assert "cases" in report
        assert "any_native_silently_wrong" in report
        assert "guard_fully_correct" in report
        assert report["guard_fully_correct"] is True, (
            "the guard must be correct on every device this run actually exercised"
        )

    def test_diagnose_cpu_case_always_present_and_ran(self):
        report = diagnose()
        cpu_cases = [c for c in report["cases"] if c["device"] == "cpu"]
        assert len(cpu_cases) == 1
        assert cpu_cases[0]["ran"] is True
        assert cpu_cases[0]["native_matches_oracle"] is True

    def test_diagnose_mps_case_present_with_ran_and_skip_reason_consistent(self):
        report = diagnose()
        mps_cases = [c for c in report["cases"] if c["device"] == "mps"]
        assert len(mps_cases) == 1
        mps_case = mps_cases[0]
        if report["mps_functional"]:
            assert mps_case["ran"] is True
            assert mps_case["skip_reason"] is None
        else:
            assert mps_case["ran"] is False
            assert mps_case["skip_reason"] is not None

    @pytest.mark.skipif(not _mps_functional(), reason="MPS not functional on this host")
    def test_diagnose_reports_native_silently_wrong_when_mps_functional(self):
        # On a genuinely functional MPS host, the native bug must show up
        # in diagnose()'s own summary flag -- this is the same repro as
        # TestNativeBugReproductionOnMps but verified through the public
        # diagnose() API surface instead of calling torch directly.
        report = diagnose()
        assert report["any_native_silently_wrong"] is True
