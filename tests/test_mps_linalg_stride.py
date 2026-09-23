"""Regression tests for torch-mps-linalg-stride-guard.

These prove:
  1. The bug is real and reproducible from scratch on this host's
     installed torch build WHEN a functional MPS device is present:
     linalg.solve_triangular / cholesky_solve / linalg.solve return a
     row-major result on MPS while CPU returns column-major, for
     mathematically identical inputs. Skipped (not silently passed)
     when MPS isn't functional on the running host -- honestly
     distinguishing "not tested" from "passed".
  2. safe_solve_triangular/safe_cholesky_solve/safe_solve are
     independently-verified fixes: their result matches the CPU
     column-major layout AND the correct values on every device they
     run on, including MPS when available.
  3. The guard never alters VALUES, only layout -- verified against a
     CPU-computed oracle for both native and guarded paths.
  4. mps_is_functional() distinguishes a genuine usable MPS device
     from `torch.backends.mps.is_available()` alone (known to be
     True-but-non-functional on virtualized CI runners).
"""
from __future__ import annotations

import pytest

torch = pytest.importorskip("torch")

from torch_correctness_guards.guards.mps_linalg_stride import (
    diagnose,
    mps_is_functional,
    safe_cholesky_solve,
    safe_solve,
    safe_solve_triangular,
)


def _mps_functional() -> bool:
    return mps_is_functional(torch)


def _spd_inputs(device):
    torch.manual_seed(0)
    A = torch.randn(5, 5, dtype=torch.float32)
    A = A @ A.T + 5 * torch.eye(5)
    B = torch.randn(5, 5, dtype=torch.float32)
    return A.to(device), B.to(device)


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

    def test_mps_is_functional_false_when_cholesky_probe_raises(self, monkeypatch):
        # Regression test for the exact failure mode observed on live
        # GitHub Actions macos-latest CI (2026-09-21): the device
        # passes a plain elementwise probe but the Metal compiler
        # rejects the matmul/cooperative-tensor kernel that
        # torch.linalg.cholesky (and by extension cholesky_solve,
        # solve, solve_triangular) depends on. Before this fix,
        # mps_is_functional() only probed a simple multiply and would
        # have wrongly reported True, letting the guarded ops crash
        # mid-test-suite instead of being honestly skipped.
        class FakeMpsBackend:
            @staticmethod
            def is_available():
                return True

        monkeypatch.setattr(torch.backends, "mps", FakeMpsBackend(), raising=False)
        real_cholesky = torch.linalg.cholesky

        def _fake_cholesky(t, *args, **kwargs):
            if getattr(t, "device", None) is not None and t.device.type == "mps":
                raise RuntimeError(
                    "simulated Metal compiler error: unsupported "
                    "deferred-static-alloca-size in cooperative_tensor "
                    "matmul kernel (observed on live macos-latest CI)"
                )
            return real_cholesky(t, *args, **kwargs)

        monkeypatch.setattr(torch.linalg, "cholesky", _fake_cholesky)
        assert mps_is_functional(torch) is False


class TestNativeBugReproductionOnMps:
    @pytest.mark.skipif(not _mps_functional(), reason="MPS not functional on this host")
    def test_native_solve_triangular_row_major_on_mps(self):
        # Real repro of pytorch/pytorch#197236 on this host's actual MPS
        # device, using the issue's own exact input construction.
        A_cpu, B_cpu = _spd_inputs("cpu")
        A_mps, B_mps = _spd_inputs("mps")
        A_cpu_tri = torch.triu(A_cpu)
        A_mps_tri = torch.triu(A_mps)

        cpu_result = torch.linalg.solve_triangular(A_cpu_tri, B_cpu, upper=True)
        mps_result = torch.linalg.solve_triangular(A_mps_tri, B_mps, upper=True)

        assert cpu_result.stride() == (1, 5), (
            "expected CPU column-major (1, 5); if this changed, the "
            "oracle assumption of this test needs updating"
        )
        assert mps_result.stride() != cpu_result.stride(), (
            "this test's whole point is that native MPS output has a "
            "DIFFERENT layout than CPU; if they now match, "
            "pytorch/pytorch#197236 may be fixed upstream -- update the "
            "README/ledger accordingly"
        )
        # Values must still agree even though layout differs.
        assert torch.allclose(mps_result.cpu(), cpu_result, atol=1e-5, rtol=1e-5)

    @pytest.mark.skipif(not _mps_functional(), reason="MPS not functional on this host")
    def test_native_cholesky_solve_row_major_on_mps(self):
        A_cpu, B_cpu = _spd_inputs("cpu")
        A_mps, B_mps = _spd_inputs("mps")
        L_cpu = torch.linalg.cholesky(A_cpu)
        L_mps = torch.linalg.cholesky(A_mps)

        cpu_result = torch.cholesky_solve(B_cpu, L_cpu)
        mps_result = torch.cholesky_solve(B_mps, L_mps)

        assert mps_result.stride() != cpu_result.stride(), (
            "expected the native layout mismatch; if this now matches "
            "CPU, the bug may be fixed upstream"
        )
        assert torch.allclose(mps_result.cpu(), cpu_result, atol=1e-5, rtol=1e-5)

    @pytest.mark.skipif(not _mps_functional(), reason="MPS not functional on this host")
    def test_native_solve_row_major_on_mps(self):
        A_cpu, B_cpu = _spd_inputs("cpu")
        A_mps, B_mps = _spd_inputs("mps")

        cpu_result = torch.linalg.solve(A_cpu, B_cpu)
        mps_result = torch.linalg.solve(A_mps, B_mps)

        assert mps_result.stride() != cpu_result.stride(), (
            "expected the native layout mismatch; if this now matches "
            "CPU, the bug may be fixed upstream"
        )
        assert torch.allclose(mps_result.cpu(), cpu_result, atol=1e-5, rtol=1e-5)


class TestGuardCorrectness:
    def test_safe_solve_delegates_unchanged_on_cpu(self):
        A_cpu, B_cpu = _spd_inputs("cpu")
        native = torch.linalg.solve(A_cpu, B_cpu)
        guarded = safe_solve(A_cpu, B_cpu)
        assert torch.equal(native, guarded)
        assert native.stride() == guarded.stride()

    @pytest.mark.skipif(not _mps_functional(), reason="MPS not functional on this host")
    def test_safe_solve_triangular_matches_cpu_layout_and_values_on_mps(self):
        A_cpu, B_cpu = _spd_inputs("cpu")
        A_mps, B_mps = _spd_inputs("mps")
        A_cpu_tri = torch.triu(A_cpu)
        A_mps_tri = torch.triu(A_mps)

        cpu_result = torch.linalg.solve_triangular(A_cpu_tri, B_cpu, upper=True)
        guard_result = safe_solve_triangular(A_mps_tri, B_mps, upper=True)

        assert guard_result.stride() == cpu_result.stride(), (
            "guard must produce the SAME (column-major) layout as CPU, "
            "unlike the native op which returns row-major"
        )
        assert torch.allclose(guard_result.cpu(), cpu_result, atol=1e-5, rtol=1e-5)

    @pytest.mark.skipif(not _mps_functional(), reason="MPS not functional on this host")
    def test_safe_cholesky_solve_matches_cpu_layout_and_values_on_mps(self):
        A_cpu, B_cpu = _spd_inputs("cpu")
        A_mps, B_mps = _spd_inputs("mps")
        L_cpu = torch.linalg.cholesky(A_cpu)
        L_mps = torch.linalg.cholesky(A_mps)

        cpu_result = torch.cholesky_solve(B_cpu, L_cpu)
        guard_result = safe_cholesky_solve(B_mps, L_mps)

        assert guard_result.stride() == cpu_result.stride()
        assert torch.allclose(guard_result.cpu(), cpu_result, atol=1e-5, rtol=1e-5)

    @pytest.mark.skipif(not _mps_functional(), reason="MPS not functional on this host")
    def test_safe_solve_matches_cpu_layout_and_values_on_mps(self):
        A_cpu, B_cpu = _spd_inputs("cpu")
        A_mps, B_mps = _spd_inputs("mps")

        cpu_result = torch.linalg.solve(A_cpu, B_cpu)
        guard_result = safe_solve(A_mps, B_mps)

        assert guard_result.stride() == cpu_result.stride()
        assert torch.allclose(guard_result.cpu(), cpu_result, atol=1e-5, rtol=1e-5)

    @pytest.mark.skipif(not _mps_functional(), reason="MPS not functional on this host")
    def test_guard_does_not_alter_already_column_major_mps_result(self, monkeypatch):
        # If a future torch build fixes the upstream bug (MPS already
        # returns column-major), the guard must be a true no-op: same
        # tensor identity-equivalent values and layout, no wasted
        # normalization round trip changing anything observable.
        import torch_correctness_guards.guards.mps_linalg_stride as core_mod

        A_cpu, B_cpu = _spd_inputs("cpu")
        A_mps, B_mps = _spd_inputs("mps")

        native_result = torch.linalg.solve(A_mps, B_mps)
        already_column_major = native_result.transpose(-1, -2).contiguous().transpose(-1, -2)
        monkeypatch.setattr(torch.linalg, "solve", lambda a, b, **kw: already_column_major)
        guarded = safe_solve(A_mps, B_mps)
        assert guarded.stride() == already_column_major.stride()
        assert torch.equal(guarded, already_column_major)


class TestDiagnose:
    def test_diagnose_returns_expected_shape(self):
        report = diagnose()
        assert "torch_version" in report
        assert "mps_functional" in report
        assert "cases" in report
        assert "any_native_layout_mismatch" in report
        assert "guard_fully_correct" in report
        assert report["guard_fully_correct"] is True, (
            "the guard must be correct on every device this run actually exercised"
        )

    def test_diagnose_has_all_three_op_cases(self):
        report = diagnose()
        names = {c["op_name"] for c in report["cases"]}
        assert names == {"solve_triangular", "cholesky_solve", "solve"}

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
    def test_diagnose_reports_native_layout_mismatch_when_mps_functional(self):
        # On a genuinely functional MPS host, the native bug must show
        # up in diagnose()'s own summary flag -- verified through the
        # public diagnose() API surface instead of calling torch
        # directly.
        report = diagnose()
        assert report["any_native_layout_mismatch"] is True
