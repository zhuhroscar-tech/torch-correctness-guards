"""Regression tests for torch-mps-copy-dtype-guard.

These prove:
  1. The bug is real and reproducible from scratch on this host's
     installed torch build WHEN a functional MPS device is present:
     transferring an MPS tensor into a float64 or complex128 CPU
     destination via .to()/.copy_() silently zeros the result (or
     leaves it unchanged for a pre-filled .copy_() destination).
     Skipped (not silently passed) when MPS isn't functional on the
     running host -- honestly distinguishing "not tested" from
     "passed".
  2. safe_to()/safe_copy_() are independently-verified fixes: their
     result matches a CPU-only oracle on every device they run on,
     including MPS when available.
  3. Unaffected dtype/device combinations delegate straight to the
     native op unchanged (no double-guarding of an already-correct
     code path).
  4. mps_is_functional() distinguishes a genuine usable MPS device
     from `torch.backends.mps.is_available()` alone (which is known
     to be True-but-non-functional on virtualized CI runners).
"""
from __future__ import annotations

import pytest

torch = pytest.importorskip("torch")

from torch_correctness_guards.guards.mps_copy_dtype import (
    diagnose,
    mps_is_functional,
    safe_copy_,
    safe_to,
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
    def test_native_to_silently_zeros_float64(self):
        # Real repro of pytorch/pytorch#197715 on this host's actual
        # MPS device. Not asserted unconditionally true forever: if a
        # future torch release fixes this upstream, this test will
        # start failing with a clear message -- update the
        # README/ledger accordingly rather than treating that flip as
        # a regression in this guard.
        src_cpu = torch.arange(1, 9, dtype=torch.float32) / 3
        src_mps = src_cpu.to("mps")
        native_result = src_mps.to("cpu", torch.float64)
        oracle = src_cpu.to(torch.float64)

        assert native_result.tolist() == pytest.approx([0.0] * 8), (
            "expected the native silent-zero bug; if this now matches the "
            "oracle, pytorch/pytorch#197715 may be fixed upstream -- "
            "update the README/ledger accordingly"
        )
        assert not torch.allclose(native_result, oracle), (
            "this test's whole point is that the native result silently "
            "disagrees with the oracle; if they now match, the bug is "
            "fixed upstream"
        )

    @pytest.mark.skipif(not _mps_functional(), reason="MPS not functional on this host")
    def test_native_to_silently_zeros_complex128(self):
        src_cpu_values = torch.arange(1, 9, dtype=torch.float32) / 3
        src_cpu = torch.complex(src_cpu_values, src_cpu_values)
        src_mps = src_cpu.to("mps")
        native_result = src_mps.to("cpu", torch.complex128)
        oracle = src_cpu.to(torch.complex128)

        assert not torch.allclose(native_result, oracle), (
            "expected the native silent-zero bug for complex128; if this "
            "now matches the oracle, the bug may be fixed upstream"
        )

    @pytest.mark.skipif(not _mps_functional(), reason="MPS not functional on this host")
    def test_native_copy_leaves_destination_unchanged(self):
        # The .copy_() variant is even sneakier: a pre-filled
        # destination isn't even zeroed, it's left completely
        # untouched -- the strongest form of "silent" failure.
        src_cpu = torch.arange(1, 9, dtype=torch.float32) / 3
        src_mps = src_cpu.to("mps")
        dst = torch.full((8,), 9.0, dtype=torch.float64)
        dst.copy_(src_mps)
        assert dst.tolist() == pytest.approx([9.0] * 8), (
            "expected the native .copy_() to silently leave the "
            "pre-filled destination untouched; if this now updates "
            "the destination, the bug may be fixed upstream"
        )


class TestGuardCorrectness:
    def test_safe_to_matches_oracle_on_cpu_source(self):
        src_cpu = torch.arange(1, 9, dtype=torch.float32) / 3
        oracle = src_cpu.to(torch.float64)
        result = safe_to(src_cpu, "cpu", torch.float64)
        assert torch.allclose(result, oracle)

    @pytest.mark.skipif(not _mps_functional(), reason="MPS not functional on this host")
    def test_safe_to_matches_oracle_on_mps_source_float64(self):
        src_cpu = torch.arange(1, 9, dtype=torch.float32) / 3
        src_mps = src_cpu.to("mps")
        oracle = src_cpu.to(torch.float64)
        result = safe_to(src_mps, "cpu", torch.float64)
        assert torch.allclose(result, oracle), (
            "guard must produce the SAME (correct) result as the CPU "
            "oracle, unlike the native op which silently disagrees"
        )

    @pytest.mark.skipif(not _mps_functional(), reason="MPS not functional on this host")
    def test_safe_to_matches_oracle_on_mps_source_complex128(self):
        src_cpu_values = torch.arange(1, 9, dtype=torch.float32) / 3
        src_cpu = torch.complex(src_cpu_values, src_cpu_values)
        src_mps = src_cpu.to("mps")
        oracle = src_cpu.to(torch.complex128)
        result = safe_to(src_mps, "cpu", torch.complex128)
        assert torch.allclose(result, oracle)

    @pytest.mark.skipif(not _mps_functional(), reason="MPS not functional on this host")
    def test_safe_copy_matches_oracle_on_mps_source(self):
        src_cpu = torch.arange(1, 9, dtype=torch.float32) / 3
        src_mps = src_cpu.to("mps")
        oracle = src_cpu.to(torch.float64)
        dst = torch.full((8,), 9.0, dtype=torch.float64)
        safe_copy_(dst, src_mps)
        assert torch.allclose(dst, oracle)

    def test_safe_to_delegates_unchanged_for_unaffected_dtype(self):
        # float32 destination isn't in the affected set: guard must
        # not alter behavior for it.
        src_cpu = torch.arange(1, 9, dtype=torch.float32) / 3
        guard_result = safe_to(src_cpu, "cpu", torch.float32)
        native_result = src_cpu.to("cpu", torch.float32)
        assert torch.equal(guard_result, native_result)

    def test_safe_to_delegates_unchanged_when_dtype_is_none(self):
        src_cpu = torch.arange(1, 9, dtype=torch.float32) / 3
        guard_result = safe_to(src_cpu, "cpu")
        native_result = src_cpu.to("cpu")
        assert torch.equal(guard_result, native_result)

    def test_safe_copy_delegates_unchanged_for_unaffected_dtype(self):
        src_cpu = torch.arange(1, 9, dtype=torch.float32) / 3
        dst_guard = torch.zeros(8, dtype=torch.float32)
        dst_native = torch.zeros(8, dtype=torch.float32)
        safe_copy_(dst_guard, src_cpu)
        dst_native.copy_(src_cpu)
        assert torch.equal(dst_guard, dst_native)


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

    def test_diagnose_has_both_dtype_cases(self):
        report = diagnose()
        names = {c["dtype_name"] for c in report["cases"]}
        assert names == {"float64", "complex128"}

    def test_diagnose_case_skip_reason_consistent_with_mps_functional(self):
        report = diagnose()
        for c in report["cases"]:
            if report["mps_functional"]:
                assert c["ran"] is True
                assert c["skip_reason"] is None
            else:
                assert c["ran"] is False
                assert c["skip_reason"] is not None

    @pytest.mark.skipif(not _mps_functional(), reason="MPS not functional on this host")
    def test_diagnose_reports_native_silently_wrong_when_mps_functional(self):
        # On a genuinely functional MPS host, the native bug must show
        # up in diagnose()'s own summary flag -- this is the same
        # repro as TestNativeBugReproductionOnMps but verified through
        # the public diagnose() API surface instead of calling torch
        # directly.
        report = diagnose()
        assert report["any_native_silently_wrong"] is True
