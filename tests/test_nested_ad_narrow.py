"""Regression tests for torch-nested-ad-narrow-guard.

These prove:
  1. All three bugs are real and reproducible from scratch on this
     host's installed torch build, using the EXACT repro code published
     in the upstream issues (pytorch/pytorch#196697, #196708, #145837)
     -- not a paraphrase.
  2. safe_nested_slogdet_second_order_jvp, safe_jagged_narrow_unbind,
     and safe_jagged_padded_transform are independently verified fixes:
     they match the mathematically correct expected value (computed by
     hand from the closed-form derivative / selection in the issue, or
     an independent no-grad reference, never derived from the buggy
     code path itself).
  3. A bug-injection test proves the "guard" logic is non-tautological:
     forward-over-reverse differentiation (torch.func.grad of
     torch.func.jvp) is ALSO silently wrong for the slogdet case (also
     returns 0.0), so merely swapping which torch.func transform runs
     first is not sufficient -- only reverse-mode-first approaches
     (autograd.grad/create_graph or jvp-of-grad) are correct. This
     guards against a future "fix" that swaps forward.func.jvp/grad
     order without actually using reverse differentiation first.
"""
from __future__ import annotations

import pytest

torch = pytest.importorskip("torch")

from torch_correctness_guards.guards.nested_ad_narrow import (
    diagnose,
    safe_nested_slogdet_second_order_jvp,
    safe_nested_householder_product_second_order_jvp,
    safe_layer_norm_second_order_jvp,
    safe_jagged_narrow_unbind,
    safe_jagged_padded_transform,
    safe_autograd_function_higher_order_derivative,
)


EXPECTED_SECOND_ORDER = -0.17853600476096013
EXPECTED_NARROW_SUM = 7 * 0.7
EXPECTED_HOUSEHOLDER_SECOND_ORDER = 1.556251320682393
EXPECTED_LAYER_NORM_SECOND_ORDER = 0.7749142079590942


def _slogdet_f(t):
    c = torch.tensor([1, 2, 3], dtype=t.dtype)
    a = torch.stack((t + c[1], c[0], c[0], c[2])).reshape(2, 2)
    sign, value = torch.linalg.slogdet(a)
    return value


def _householder_f(t):
    c = torch.ones((), dtype=t.dtype)
    a = torch.stack((c, t)).reshape(2, 1)
    tau = (2 / (1 + t * t)).reshape(1)
    return torch.linalg.householder_product(a, tau).sum()


def _layer_norm_f(t):
    zero = torch.zeros((), dtype=t.dtype)
    x = torch.stack((t, zero)).reshape(1, 2)
    weight = torch.tensor([1, 2], dtype=t.dtype)
    bias = torch.zeros((2,), dtype=t.dtype)
    return torch.nn.functional.layer_norm(
        x, normalized_shape=[2], weight=weight, bias=bias, eps=0.25
    ).sum()


class TestNativeBugReproduction:
    def test_slogdet_nested_jvp_silently_returns_zero_on_this_host(self):
        # Exact repro from pytorch/pytorch#196697's issue body.
        import torch.func as tfunc

        dtype = torch.float64

        def jvp1(t):
            return tfunc.jvp(_slogdet_f, (t,), (torch.ones_like(t),))[1]

        t = torch.tensor(0.7, dtype=dtype)
        _, actual = tfunc.jvp(jvp1, (t,), (torch.ones_like(t),))

        assert abs(actual.item()) < 1e-10, (
            "expected the known bug (nested forward-mode JVP collapses to "
            "0.0); if this now fails, the bug may be fixed upstream "
            "(pytorch/pytorch#196697) -- update the README/ledger "
            "accordingly rather than treating this as a regression"
        )
        assert abs(EXPECTED_SECOND_ORDER) > 1e-3, "sanity: expected value is genuinely nonzero"

    def test_narrow_unbind_silently_includes_unselected_element_on_this_host(self):
        # Exact repro from pytorch/pytorch#196708's issue body.
        dtype = torch.float64
        t = torch.tensor(0.7, dtype=dtype)
        x = torch.stack((t, 2 * t, 3 * t, 4 * t)).reshape(2, 2)
        starts = torch.tensor([0, 1], dtype=torch.int64)
        lengths = torch.tensor([2, 1], dtype=torch.int64)

        nt = torch.nested.narrow(x, 1, starts, lengths, layout=torch.jagged)
        parts = nt.unbind()
        actual = (parts[0].sum() + parts[1].sum()).item()

        assert abs(actual - EXPECTED_NARROW_SUM) > 1e-3, (
            "expected the known bug (narrow+unbind includes an unselected "
            "element, giving 10*t instead of 7*t); if this now fails, the "
            "bug may be fixed upstream (pytorch/pytorch#196708) -- update "
            "the README/ledger accordingly rather than treating this as a "
            "regression"
        )

    def test_padded_jagged_roundtrip_breaks_backward_on_this_host(self):
        # Exact repro pattern from pytorch/pytorch#145837's issue body:
        # pad -> transform -> torch.nested.narrow back to jagged -> backward.
        torch.manual_seed(0)
        dim = 4
        offsets = torch.tensor([0, 3, 5, 9])
        pos_emb = torch.randn(1, 10, dim)

        values = torch.randn(9, dim, requires_grad=True)
        x = torch.nested.nested_tensor_from_jagged(values, offsets)

        offsets_orig = x.offsets()
        padded = torch.nested.to_padded_tensor(x, padding=0.0)
        seq_len = padded.shape[1]
        padded = padded + pos_emb[:, :seq_len, :]
        jagged = torch.nested.narrow(
            padded, dim=1, start=0, length=offsets_orig.diff(), layout=torch.jagged
        )

        with pytest.raises(RuntimeError, match="invalid gradient"):
            jagged.contiguous().values().mean().backward()

    def test_householder_product_nested_jvp_silently_wrong_on_this_host(self):
        # Exact repro from pytorch/pytorch#196698's issue body.
        import torch.func as tfunc

        dtype = torch.float64

        def jvp1(t):
            return tfunc.jvp(_householder_f, (t,), (torch.ones_like(t),))[1]

        t = torch.tensor(0.7, dtype=dtype)
        _, actual = tfunc.jvp(jvp1, (t,), (torch.ones_like(t),))

        assert abs(actual.item() - EXPECTED_HOUSEHOLDER_SECOND_ORDER) > 1e-3, (
            "expected the known bug (nested forward-mode JVP returns the "
            "wrong second derivative for householder_product); if this "
            "now passes, the bug may be fixed upstream "
            "(pytorch/pytorch#196698) -- update the README/ledger "
            "accordingly rather than treating this as a regression"
        )


class TestBugInjectionNonTautological:
    """Prove forward-over-reverse is ALSO silently wrong, so a fix must
    specifically use reverse-mode-first differentiation, not just any
    reordering of torch.func transforms."""

    def test_forward_over_reverse_is_also_silently_wrong(self):
        import torch.func as tfunc

        dtype = torch.float64
        t0 = torch.tensor(0.7, dtype=dtype)

        def jvp_f(t):
            return tfunc.jvp(_slogdet_f, (t,), (torch.ones_like(t),))[1]

        second_for = tfunc.grad(jvp_f)(t0)
        assert abs(second_for.item()) < 1e-10, (
            "sanity check: forward-over-reverse (grad of jvp) really is "
            "ALSO silently wrong (0.0) on this host -- this is why the "
            "guard specifically requires reverse-mode-first "
            "differentiation, not just 'use torch.func differently'"
        )

    def test_householder_forward_over_reverse_is_also_silently_wrong(self):
        # Same non-tautology check for the householder_product case
        # (#196698): forward-over-reverse is ALSO wrong here, just to a
        # different (nonzero, but still incorrect) value than
        # forward-over-forward -- proving the guard's reverse-over-
        # reverse requirement isn't just "avoid nesting jvp twice".
        import torch.func as tfunc

        dtype = torch.float64
        t0 = torch.tensor(0.7, dtype=dtype)

        def jvp_f(t):
            return tfunc.jvp(_householder_f, (t,), (torch.ones_like(t),))[1]

        second_for = tfunc.grad(jvp_f)(t0)
        assert abs(second_for.item() - EXPECTED_HOUSEHOLDER_SECOND_ORDER) > 1e-3, (
            "sanity check: forward-over-reverse (grad of jvp) is ALSO "
            "silently wrong for householder_product on this host"
        )

    def test_householder_independent_finite_difference_oracle_confirms_expected(self):
        # Independent oracle computed via central finite differences on
        # the closed-form column sum f(t) = 1 - 2*(1+t)/(1+t**2) -- never
        # calls torch.func, torch.linalg, or any guard code path, so it
        # cannot share a bug with what it is checking.
        def f_closed(t):
            return 1 - 2 * (1 + t) / (1 + t * t)

        h = 1e-5
        t0 = 0.7
        d2 = (f_closed(t0 + h) - 2 * f_closed(t0) + f_closed(t0 - h)) / (h * h)
        assert abs(d2 - EXPECTED_HOUSEHOLDER_SECOND_ORDER) < 1e-4

    def test_layer_norm_nested_jvp_silently_wrong_sign_on_this_host(self):
        # Exact repro from pytorch/pytorch#196700's issue body.
        import torch.func as tfunc

        t0 = torch.tensor(0.7, dtype=torch.float64)

        def jvp1(t):
            return tfunc.jvp(_layer_norm_f, (t,), (torch.ones_like(t),))[1]

        _, actual = tfunc.jvp(jvp1, (t0,), (torch.ones_like(t0),))
        # This is a genuine defect check: if a future torch release
        # fixes #196700, this assertion should start failing loudly
        # (not silently), signaling the guard workaround can be
        # retired for this op.
        assert abs(actual.item() - EXPECTED_LAYER_NORM_SECOND_ORDER) > 1e-3, (
            "forward-over-forward nested JVP now matches the expected "
            "value -- pytorch/pytorch#196700 may be fixed upstream; "
            "re-evaluate whether safe_layer_norm_second_order_jvp is "
            "still needed."
        )

    def _make_square_autograd_function(self):
        class Square(torch.autograd.Function):
            generate_vmap_rule = True

            @staticmethod
            def forward(x):
                return x * x

            @staticmethod
            def setup_context(ctx, inputs, output):
                ctx.save_for_backward(*inputs)
                ctx.save_for_forward(*inputs)

            @staticmethod
            def backward(ctx, g):
                (x,) = ctx.saved_tensors
                return 2 * x * g

            @staticmethod
            def jvp(ctx, dx):
                (x,) = ctx.saved_tensors
                return 2 * x * dx

        return Square

    def test_jacfwd_chain_through_custom_function_silently_zeroes_on_this_host(self):
        # Exact repro from pytorch/pytorch#197867's issue body: x**4 via
        # a custom autograd.Function with a correct analytic jvp.
        import torch.func as tfunc

        Square = self._make_square_autograd_function()

        def custom(x):
            return Square.apply(Square.apply(x)).sum()

        x0 = torch.tensor([3.0], dtype=torch.float64)
        derivative = custom
        values = []
        for _ in range(3):
            derivative = tfunc.jacfwd(derivative)
            values.append(derivative(x0).item())

        assert values[0] == pytest.approx(108.0, abs=1e-6), (
            "sanity: first derivative should be correct on this host"
        )
        assert abs(values[1]) < 1e-9 and abs(values[2]) < 1e-9, (
            "expected the known bug (jacfwd chained through a custom "
            "autograd.Function silently zeroes the 2nd/3rd derivative); "
            "if this now fails, the bug may be fixed upstream "
            "(pytorch/pytorch#197867) -- update the README/ledger "
            "accordingly rather than treating this as a regression"
        )

    def test_layer_norm_forward_over_reverse_is_also_silently_wrong(self):
        # Same non-tautology check as the slogdet/householder cases:
        # merely swapping torch.func transform order (grad-of-jvp
        # instead of jvp-of-jvp) does NOT fix layer_norm either -- both
        # wrong paths return the identical wrong-sign value.
        import torch.func as tfunc

        t0 = torch.tensor(0.7, dtype=torch.float64)

        def jvp1(t):
            return tfunc.jvp(_layer_norm_f, (t,), (torch.ones_like(t),))[1]

        for_rev = tfunc.grad(jvp1)(t0)
        assert abs(for_rev.item() - EXPECTED_LAYER_NORM_SECOND_ORDER) > 1e-3, (
            "forward-over-reverse now matches expected -- re-evaluate "
            "whether swapping transform order alone would suffice"
        )

    def test_layer_norm_independent_finite_difference_oracle_confirms_expected(self):
        # Independent oracle computed via central finite differences on
        # the closed-form scalar expression f(t) = -t / sqrt(1 + t**2)
        # -- never calls torch.func, torch.nn.functional, or any guard
        # code path, so it cannot share a bug with what it is checking.
        def f_closed(t):
            return -t / (1 + t * t) ** 0.5

        h = 1e-4
        t0 = 0.7
        d2 = (f_closed(t0 + h) - 2 * f_closed(t0) + f_closed(t0 - h)) / (h * h)
        assert abs(d2 - EXPECTED_LAYER_NORM_SECOND_ORDER) < 1e-4


class TestGuardMatchesExpected:
    def test_safe_nested_slogdet_second_order_jvp_matches_expected(self):
        t0 = torch.tensor(0.7, dtype=torch.float64)
        result = safe_nested_slogdet_second_order_jvp(_slogdet_f, t0)
        assert abs(result.item() - EXPECTED_SECOND_ORDER) < 1e-9

    def test_safe_nested_slogdet_matches_reverse_over_forward_independent_method(self):
        # Independent oracle: reverse-over-forward (jvp of grad), a
        # DIFFERENT differentiation order than the guard's
        # reverse-over-reverse, computed independently in this test
        # rather than reusing the guard's own code path.
        import torch.func as tfunc

        t0 = torch.tensor(0.7, dtype=torch.float64)

        def grad_f(t):
            return tfunc.grad(_slogdet_f)(t)

        _, oracle = tfunc.jvp(grad_f, (t0,), (torch.ones_like(t0),))

        guard_result = safe_nested_slogdet_second_order_jvp(_slogdet_f, t0)
        assert abs(guard_result.item() - oracle.item()) < 1e-9

    def test_safe_nested_householder_product_second_order_jvp_matches_expected(self):
        t0 = torch.tensor(0.7, dtype=torch.float64)
        result = safe_nested_householder_product_second_order_jvp(_householder_f, t0)
        assert abs(result.item() - EXPECTED_HOUSEHOLDER_SECOND_ORDER) < 1e-9

    def test_safe_nested_householder_product_matches_finite_difference_oracle(self):
        # Independent oracle: central finite difference on the
        # closed-form expression, computed without any torch.func or
        # torch.linalg call, so it cannot share a bug with the guard.
        def f_closed(t):
            return 1 - 2 * (1 + t) / (1 + t * t)

        h = 1e-5
        t0_val = 0.7
        oracle = (f_closed(t0_val + h) - 2 * f_closed(t0_val) + f_closed(t0_val - h)) / (h * h)

        t0 = torch.tensor(t0_val, dtype=torch.float64)
        guard_result = safe_nested_householder_product_second_order_jvp(_householder_f, t0)
        assert abs(guard_result.item() - oracle) < 1e-4

    def test_safe_layer_norm_second_order_jvp_matches_expected(self):
        t0 = torch.tensor(0.7, dtype=torch.float64)
        result = safe_layer_norm_second_order_jvp(_layer_norm_f, t0)
        assert abs(result.item() - EXPECTED_LAYER_NORM_SECOND_ORDER) < 1e-9

    def test_safe_layer_norm_matches_finite_difference_oracle(self):
        # Independent oracle: central finite difference on the
        # closed-form scalar expression, computed without any
        # torch.func or torch.nn.functional call, so it cannot share a
        # bug with the guard.
        def f_closed(t):
            return -t / (1 + t * t) ** 0.5

        h = 1e-4
        t0_val = 0.7
        oracle = (f_closed(t0_val + h) - 2 * f_closed(t0_val) + f_closed(t0_val - h)) / (h * h)

        t0 = torch.tensor(t0_val, dtype=torch.float64)
        guard_result = safe_layer_norm_second_order_jvp(_layer_norm_f, t0)
        assert abs(guard_result.item() - oracle) < 1e-4

    def test_safe_jagged_narrow_unbind_matches_expected(self):
        dtype = torch.float64
        t = torch.tensor(0.7, dtype=dtype)
        x = torch.stack((t, 2 * t, 3 * t, 4 * t)).reshape(2, 2)
        starts = torch.tensor([0, 1], dtype=torch.int64)
        lengths = torch.tensor([2, 1], dtype=torch.int64)

        parts = safe_jagged_narrow_unbind(x, 1, starts, lengths)
        total = sum(p.sum() for p in parts).item()
        assert abs(total - EXPECTED_NARROW_SUM) < 1e-9

    def test_safe_jagged_narrow_unbind_selects_correct_individual_rows(self):
        dtype = torch.float64
        t = torch.tensor(0.7, dtype=dtype)
        x = torch.stack((t, 2 * t, 3 * t, 4 * t)).reshape(2, 2)
        starts = torch.tensor([0, 1], dtype=torch.int64)
        lengths = torch.tensor([2, 1], dtype=torch.int64)

        parts = safe_jagged_narrow_unbind(x, 1, starts, lengths)
        assert parts[0].numel() == 2
        assert parts[1].numel() == 1
        assert torch.allclose(parts[0], torch.stack((t, 2 * t)))
        assert torch.allclose(parts[1], torch.stack((4 * t,)))

    def test_safe_jagged_narrow_unbind_rejects_mismatched_row_count(self):
        x = torch.zeros(2, 4, dtype=torch.float64)
        starts = torch.tensor([0, 1, 2], dtype=torch.int64)  # wrong length
        lengths = torch.tensor([2, 1, 1], dtype=torch.int64)
        with pytest.raises(ValueError):
            safe_jagged_narrow_unbind(x, 1, starts, lengths)

    def test_safe_jagged_narrow_unbind_rejects_dim_zero(self):
        x = torch.zeros(2, 4, dtype=torch.float64)
        starts = torch.tensor([0, 0], dtype=torch.int64)
        lengths = torch.tensor([1, 1], dtype=torch.int64)
        with pytest.raises(NotImplementedError):
            safe_jagged_narrow_unbind(x, 0, starts, lengths)

    def test_safe_jagged_padded_transform_backward_succeeds(self):
        torch.manual_seed(0)
        dim = 4
        offsets = torch.tensor([0, 3, 5, 9])
        pos_emb = torch.randn(1, 10, dim)

        def transform(padded):
            seq_len = padded.shape[1]
            return padded + pos_emb[:, :seq_len, :]

        values = torch.randn(9, dim, requires_grad=True)
        x = torch.nested.nested_tensor_from_jagged(values, offsets)

        out = safe_jagged_padded_transform(x, transform)
        loss = out.values().mean()
        loss.backward()  # must not raise -- proves the fix for #145837

        assert values.grad is not None
        assert not torch.isnan(values.grad).any()

    def test_safe_jagged_padded_transform_forward_matches_reference(self):
        torch.manual_seed(1)
        dim = 3
        offsets = torch.tensor([0, 2, 5, 6])
        pos_emb = torch.randn(1, 8, dim)

        def transform(padded):
            seq_len = padded.shape[1]
            return padded + pos_emb[:, :seq_len, :]

        values = torch.randn(6, dim)
        x = torch.nested.nested_tensor_from_jagged(values, offsets)

        out = safe_jagged_padded_transform(x, transform)

        # Independent reference: manually pad, transform, and slice back
        # without going through the guard function at all.
        offs = offsets.tolist()
        padded_ref = torch.nested.to_padded_tensor(x, padding=0.0)
        padded_ref = transform(padded_ref)
        rows = [padded_ref[i, : offs[i + 1] - offs[i]] for i in range(len(offs) - 1)]
        expected = torch.cat(rows, dim=0)

        assert torch.allclose(out.values(), expected, atol=1e-6)

    def test_safe_autograd_function_higher_order_derivative_matches_expected(self):
        Square = self._make_square_autograd_function_module_level()

        def custom(x):
            return Square.apply(Square.apply(x)).sum()

        x0 = torch.tensor([3.0], dtype=torch.float64)
        expected = [108.0, 108.0, 72.0]
        for order, exp in zip((1, 2, 3), expected):
            result = safe_autograd_function_higher_order_derivative(custom, x0, order)
            assert abs(result.item() - exp) < 1e-6

    def test_safe_autograd_function_higher_order_derivative_matches_plain_ops_reference(self):
        # Independent oracle: the same derivative order computed via
        # reverse-mode chaining on a PLAIN tensor-ops function (no
        # custom autograd.Function at all), so it cannot share the
        # #197867 bug with what it is checking.
        def pure(x):
            return ((x * x) * (x * x)).sum()

        Square = self._make_square_autograd_function_module_level()

        def custom(x):
            return Square.apply(Square.apply(x)).sum()

        x0 = torch.tensor([3.0], dtype=torch.float64)
        for order in (1, 2, 3):
            pure_result = safe_autograd_function_higher_order_derivative(pure, x0, order)
            custom_result = safe_autograd_function_higher_order_derivative(custom, x0, order)
            assert abs(pure_result.item() - custom_result.item()) < 1e-6

    def test_safe_autograd_function_higher_order_derivative_rejects_order_zero(self):
        def pure(x):
            return (x * x).sum()

        x0 = torch.tensor([3.0], dtype=torch.float64)
        with pytest.raises(ValueError):
            safe_autograd_function_higher_order_derivative(pure, x0, 0)

    @staticmethod
    def _make_square_autograd_function_module_level():
        class Square(torch.autograd.Function):
            generate_vmap_rule = True

            @staticmethod
            def forward(x):
                return x * x

            @staticmethod
            def setup_context(ctx, inputs, output):
                ctx.save_for_backward(*inputs)
                ctx.save_for_forward(*inputs)

            @staticmethod
            def backward(ctx, g):
                (x,) = ctx.saved_tensors
                return 2 * x * g

            @staticmethod
            def jvp(ctx, dx):
                (x,) = ctx.saved_tensors
                return 2 * x * dx

        return Square


class TestDiagnose:
    def test_diagnose_runs_and_reports_consistent_structure(self):
        report = diagnose()
        assert isinstance(report["torch_version"], str)
        assert "slogdet_second_order_case" in report
        assert "householder_product_second_order_case" in report
        assert "layer_norm_second_order_case" in report
        assert "jagged_narrow_unbind_case" in report
        assert "jagged_padded_transform_case" in report
        assert "autograd_function_higher_order_case" in report

    def test_diagnose_reports_all_bugs_reproduced_on_this_host(self):
        report = diagnose()
        assert report["slogdet_bug_reproduced"] is True, (
            "expected the slogdet nested-JVP bug to reproduce on "
            f"torch {report['torch_version']}"
        )
        assert report["householder_bug_reproduced"] is True, (
            "expected the householder_product nested-JVP bug to "
            f"reproduce on torch {report['torch_version']}"
        )
        assert report["layer_norm_bug_reproduced"] is True, (
            "expected the layer_norm nested-JVP bug to "
            f"reproduce on torch {report['torch_version']}"
        )
        assert report["narrow_bug_reproduced"] is True, (
            "expected the narrow+unbind bug to reproduce on "
            f"torch {report['torch_version']}"
        )
        assert report["padded_transform_bug_reproduced"] is True, (
            "expected the padded<->jagged roundtrip backward bug to "
            f"reproduce on torch {report['torch_version']}"
        )
        assert report["autograd_function_higher_order_bug_reproduced"] is True, (
            "expected the jacfwd-chain-through-custom-Function bug to "
            f"reproduce on torch {report['torch_version']}"
        )

    def test_diagnose_reports_guards_fully_correct(self):
        report = diagnose()
        assert report["guards_fully_correct"] is True
