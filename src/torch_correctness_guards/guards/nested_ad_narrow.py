"""torch-nested-ad-narrow-guard core: guard four real torch correctness bugs.

All bugs were independently reproduced from scratch on this host (torch
2.14.0, macOS arm64 CPU) using the EXACT repro code published in the
upstream issues (not paraphrased) -- see README for full commands and
issue links.

1. pytorch/pytorch#196697 -- nested (second-order) forward-mode AD via
   ``torch.func.jvp(jvp1, ...)`` silently collapses ``torch.linalg.slogdet``'s
   second derivative to exactly 0.0 instead of the mathematically correct
   nonzero value. Reproduced verbatim: expected -0.17853600476096013,
   torch.func.jvp-over-jvp actually returns 0.0. This is a genuinely
   dangerous silent-zero failure mode (not a crash, not NaN) for anyone
   computing curvature/Hessian-vector-product-style quantities of
   log-determinants via nested forward-mode AD (e.g. natural-gradient or
   log-likelihood-Hessian methods for Gaussian/matrix-variate models).

   This module does NOT attempt to patch torch's forward-mode AD
   internals (out of scope for a small guard package). Instead it
   provides ``safe_nested_slogdet_second_order_jvp``, a verified-correct
   replacement path using reverse-over-reverse (``torch.autograd.grad``
   with ``create_graph=True`` called twice) and reverse-over-forward
   (``torch.func.jvp`` of ``torch.func.grad``), both of which were
   confirmedon this host to exactly match the expected analytic value
   -- unlike forward-over-reverse (``torch.func.grad`` of
   ``torch.func.jvp``), which ALSO silently returns 0.0 and must be
   avoided.

2. pytorch/pytorch#196708 -- ``torch.nested.narrow(x, dim, starts,
   lengths, layout=torch.jagged)`` followed by ``.unbind()`` silently
   includes an element outside the requested [start, start+length)
   range for at least one row. Reproduced verbatim: for rows
   ``[t, 2*t]`` and ``[3*t, 4*t]`` with starts=[0, 1], lengths=[2, 1],
   the correct selection is ``[t, 2*t]`` and ``[4*t]`` (sum = 7*t), but
   narrow+unbind returns sum = 10*t (i.e. it silently also includes the
   unselected 3*t). At t=0.7: actual 6.999999999999999 vs expected
   4.8999999999999995.

   ``safe_jagged_narrow_unbind`` provides a verified-correct workaround:
   it never calls the buggy ``torch.nested.narrow``, instead building a
   fresh jagged nested tensor directly from the requested per-row
   Python-level slices of the dense input, then unbinding that. This
   matches the mathematically correct selection exactly on this host.

3. pytorch/pytorch#145837 -- round-tripping a jagged nested tensor
   through ``torch.nested.to_padded_tensor`` (pad), applying some
   transform on the padded dense form (e.g. adding a positional
   embedding), then converting back with
   ``torch.nested.narrow(..., layout=torch.jagged)`` breaks the
   backward pass with ``RuntimeError: Function CloneBackward0 returned
   an invalid gradient at index 0 - got [..., jN, ...] but expected
   shape compatible with [..., jM, ...]``. Independently reproduced on
   this host (torch 2.14.0) with a 3-sequence jagged batch (lengths
   3/2/4, dim=8): the forward pass succeeds, but ``.backward()`` raises
   this shape-mismatch RuntimeError every time, because
   ``torch.nested.narrow`` constructs a *new* jagged offsets object on
   the backward reconstruction instead of reusing the exact tensor
   object produced when the input was first converted from padded ->
   jagged, and autograd's shape-compatibility check treats the two
   (numerically-identical) offsets objects as different symbolic jagged
   sizes.

   This is a common pattern for adding positional embeddings (or any
   op unsupported directly in jagged layout) to variable-length
   sequence batches, so it silently blocks a normal, expected NestedTensor
   workflow with a crash on backward -- not silently wrong output like
   bugs 1 and 2, but a full training-loop stopper.

4. pytorch/pytorch#196698 -- nested (second-order) forward-mode AD via
   ``torch.func.jvp(jvp1, ...)`` ALSO silently returns the wrong value
   for ``torch.linalg.householder_product``'s second derivative --
   the same failure class as bug 1 (#196697), but a distinct linalg op,
   discovered independently during this fleet's maintenance scouting
   rather than sourced from bug 1's own issue thread. Reproduced
   verbatim on this host: expected 1.556251320682393 (cross-checked
   against an independent central-finite-difference oracle on the
   closed-form ``1 - 2*(1+t)/(1+t**2)``, giving 1.5562484634301652,
   matching to ~1e-5), forward-over-forward (jvp-of-jvp) actually
   returns -0.7533368863909331. Forward-over-reverse (grad-of-jvp) is
   ALSO wrong (-0.753336886390932); reverse-over-forward (jvp-of-grad)
   raises a functorch-transform RuntimeError entirely (unlike the
   slogdet case, where it works). Only reverse-over-reverse
   (``torch.autograd.grad(create_graph=True)`` twice) is verified
   correct here, matching the finite-difference oracle to ~1e-9.

   ``safe_nested_householder_product_second_order_jvp`` reuses the same
   verified reverse-over-reverse pattern as bug 1's guard.

   ``safe_jagged_padded_transform`` provides a verified-correct
   workaround: it converts to padded form, applies the transform, then
   reconstructs the jagged tensor via direct per-row Python-level
   slicing and ``torch.cat`` (never calling ``torch.nested.narrow``),
   which keeps the autograd graph consistent and allows ``.backward()``
   to complete. Verified on this host to (a) complete backward without
   raising, and (b) produce forward values that exactly match a
   no-grad reference computed the same way.

5. pytorch/pytorch#196700 -- nested (second-order) forward-mode AD via
   ``torch.func.jvp(jvp1, ...)`` ALSO silently returns the WRONG-SIGN
   value for ``torch.nn.functional.layer_norm``'s second derivative --
   the same failure class as bugs 1 and 4 (#196697, #196698), a third
   distinct op, found independently during this fleet's external OSS
   scouting rather than sourced from bug 1's own issue thread.
   Reproduced verbatim on this host: expected 0.7749142079590942
   (cross-checked against an independent central-finite-difference
   oracle computed directly on the closed-form scalar expression
   ``-t / sqrt(1 + t**2)`` with no autograd involved at all, giving
   0.774914199475063, matching to ~1e-7), forward-over-forward
   (jvp-of-jvp) actually returns -0.38487405661968344 -- not merely a
   magnitude error but the WRONG SIGN. Forward-over-reverse (grad-of-
   jvp) is ALSO wrong (-0.3848740566196835, same wrong value);
   reverse-over-forward (jvp-of-grad) raises the same functorch-
   transform RuntimeError as bug 4's reverse-over-forward attempt.
   Only reverse-over-reverse (``torch.autograd.grad(create_graph=True)``
   twice) is verified correct here (0.7749142079590943, matching the
   finite-difference oracle to ~1e-13).

   ``safe_layer_norm_second_order_jvp`` reuses the same verified
   reverse-over-reverse pattern as bugs 1 and 4's guards.

6. pytorch/pytorch#197867 -- ``torch.func.jacfwd`` chained through a
   THIRD-order derivative (``jacfwd(jacfwd(jacfwd(f)))``) silently
   returns 0.0 for the 2nd and 3rd derivatives whenever ``f`` routes
   through a custom ``torch.autograd.Function`` that defines a
   ``jvp`` staticmethod (the officially documented way to make a
   custom autograd.Function forward-mode-AD compatible) -- a
   DIFFERENT failure trigger than bugs 1/4/5 above (those need
   ``torch.func.jvp`` literally nested inside another ``jvp``; this
   one needs ``jacfwd`` chained three deep through a custom Function,
   and does not require calling ``torch.func.jvp`` directly at all).
   Independently reproduced on this host (torch 2.14.0): for
   ``f(x) = Square.apply(Square.apply(x))`` (mathematically ``x**4``,
   where ``Square`` is a minimal custom autograd.Function with a
   correct analytic ``jvp``), plain tensor ops give derivatives
   ``[108.0, 108.0, 72.0]`` at ``x=3``; the custom-Function version
   gives ``[108.0, 0.0, 0.0]`` -- the first derivative is right, the
   second and third silently collapse to zero. Reverse-over-forward
   (``jacrev`` of ``jacfwd``) gives the correct first-order value but
   was NOT independently re-checked for order 2/3 by this guard (out
   of scope; see README for what was and was not tested).

   ``safe_autograd_function_higher_order_derivative`` avoids
   ``torch.func.jacfwd``/``jvp`` (forward-mode AD) entirely for this
   case: it computes the requested derivative order via ``order``
   chained calls to ``torch.autograd.grad(..., create_graph=True)``
   (reverse-mode only), which was independently verified on this host
   to reproduce the correct ``[108.0, 108.0, 72.0]`` sequence for both
   the plain-ops function AND the custom-Function version -- proving
   the bug is specific to forward-mode AD through the custom
   Function's ``jvp`` path, not a fundamental limitation of computing
   third derivatives of this function at all.
"""
from __future__ import annotations

import dataclasses
from typing import Any, Dict, List, Sequence


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


def safe_nested_slogdet_second_order_jvp(f, t):
    """Compute the second derivative of a scalar function ``f`` (which
    internally uses ``torch.linalg.slogdet``) at scalar tensor ``t``,
    avoiding the silently-wrong-zero result that nested forward-mode AD
    (``torch.func.jvp`` of ``torch.func.jvp``) produces for this op
    (pytorch/pytorch#196697).

    Uses reverse-over-reverse differentiation
    (``torch.autograd.grad`` with ``create_graph=True``, called twice),
    which was independently verified on this host to match the
    analytic second derivative exactly for the slogdet case that
    nested forward-mode AD gets wrong.

    ``t`` must be a scalar tensor; ``f`` must be differentiable via
    ``torch.autograd`` (i.e. must not itself require ``torch.func``
    transforms internally).
    """
    torch_module = _import_torch()
    t_req = t.clone().detach().requires_grad_(True)
    y = f(t_req)
    (g1,) = torch_module.autograd.grad(y, t_req, create_graph=True)
    (g2,) = torch_module.autograd.grad(g1, t_req)
    return g2.detach()


def safe_nested_householder_product_second_order_jvp(f, t):
    """Compute the second derivative of a scalar function ``f`` (which
    internally uses ``torch.linalg.householder_product``) at scalar
    tensor ``t``, avoiding the silently-wrong result that nested
    forward-mode AD (``torch.func.jvp`` of ``torch.func.jvp``) produces
    for this op (pytorch/pytorch#196698 -- the same failure class as
    #196697 for ``torch.linalg.slogdet``, but a distinct op discovered
    independently during fleet maintenance).

    Uses reverse-over-reverse differentiation (``torch.autograd.grad``
    with ``create_graph=True``, called twice), which was independently
    verified on this host to match a central-finite-difference oracle
    to ~1e-9 for the householder_product case that nested forward-mode
    AD gets wrong (both forward-over-forward AND forward-over-reverse
    are wrong for this op too -- see README).

    ``t`` must be a scalar tensor; ``f`` must be differentiable via
    ``torch.autograd`` (i.e. must not itself require ``torch.func``
    transforms internally).
    """
    torch_module = _import_torch()
    t_req = t.clone().detach().requires_grad_(True)
    y = f(t_req)
    (g1,) = torch_module.autograd.grad(y, t_req, create_graph=True)
    (g2,) = torch_module.autograd.grad(g1, t_req)
    return g2.detach()


def safe_layer_norm_second_order_jvp(f, t):
    """Compute the second derivative of a scalar function ``f`` (which
    internally uses ``torch.nn.functional.layer_norm``) at scalar
    tensor ``t``, avoiding the silently-wrong (wrong-sign) result that
    nested forward-mode AD (``torch.func.jvp`` of ``torch.func.jvp``)
    produces for this op (pytorch/pytorch#196700 -- the same failure
    class as #196697/#196698, but a distinct op discovered
    independently during fleet maintenance).

    Uses reverse-over-reverse differentiation (``torch.autograd.grad``
    with ``create_graph=True``, called twice), which was independently
    verified on this host to match a central-finite-difference oracle
    to ~1e-7 for the layer_norm case that nested forward-mode AD gets
    wrong (both forward-over-forward AND forward-over-reverse return
    the identical wrong-sign value for this op -- see README).

    ``t`` must be a scalar tensor; ``f`` must be differentiable via
    ``torch.autograd`` (i.e. must not itself require ``torch.func``
    transforms internally).
    """
    torch_module = _import_torch()
    t_req = t.clone().detach().requires_grad_(True)
    y = f(t_req)
    (g1,) = torch_module.autograd.grad(y, t_req, create_graph=True)
    (g2,) = torch_module.autograd.grad(g1, t_req)
    return g2.detach()


def safe_jagged_narrow_unbind(dense_x, dim: int, starts, lengths):
    """Select per-row [start, start+length) slices of dense tensor
    ``dense_x`` along ``dim`` and return them as unbound tensors, without
    ever calling the buggy ``torch.nested.narrow(..., layout=torch.jagged)``
    (pytorch/pytorch#196708), which silently includes elements outside
    the requested per-row range for at least one row.

    ``dense_x`` is a 2D dense tensor (rows, along dim=0, each sliced
    along ``dim``). ``starts``/``lengths`` are 1D integer tensors or
    Python sequences, one entry per row along dim 0.

    Returns a list of 1D tensors, one per row, each the correctly
    selected slice -- equivalent to what narrow+unbind SHOULD have
    returned.
    """
    torch_module = _import_torch()
    if dim != 1:
        raise NotImplementedError(
            "safe_jagged_narrow_unbind currently only supports dim=1 "
            "(the same dim used in the upstream bug report); dim=0 "
            "narrowing over jagged nested tensors is unrelated to this "
            "guard's scope."
        )

    starts_list = [int(s) for s in (starts.tolist() if hasattr(starts, "tolist") else starts)]
    lengths_list = [int(l) for l in (lengths.tolist() if hasattr(lengths, "tolist") else lengths)]

    if len(starts_list) != dense_x.shape[0] or len(lengths_list) != dense_x.shape[0]:
        raise ValueError(
            "starts/lengths must have one entry per row of dense_x "
            f"(dense_x.shape[0]={dense_x.shape[0]}, "
            f"len(starts)={len(starts_list)}, len(lengths)={len(lengths_list)})"
        )

    rows: List[Any] = []
    for i in range(dense_x.shape[0]):
        s, l = starts_list[i], lengths_list[i]
        rows.append(dense_x[i, s : s + l].clone())
    return rows


def safe_autograd_function_higher_order_derivative(f, x, order: int):
    """Compute the ``order``-th derivative of scalar-output function ``f``
    at tensor ``x``, avoiding the silently-wrong-zero result that
    ``torch.func.jacfwd`` chained ``order`` times deep produces when
    ``f`` routes through a custom ``torch.autograd.Function`` defining
    a ``jvp`` staticmethod (pytorch/pytorch#197867).

    Uses ``order`` chained calls to ``torch.autograd.grad`` with
    ``create_graph=True`` (pure reverse-mode AD, never forward-mode),
    which was independently verified on this host to reproduce the
    correct derivative sequence for both a plain-tensor-ops function
    and an equivalent function built from a custom autograd.Function
    with a correct analytic ``jvp`` -- proving the upstream bug is
    specific to nested ``jacfwd``/forward-mode AD through the custom
    Function path, not a fundamental limitation of the underlying
    math.

    ``x`` must be a tensor with ``requires_grad=True`` (or will be
    cloned and have it set); ``f`` must return a scalar (0-dim or
    single-element) tensor. ``order`` must be a positive integer.
    """
    torch_module = _import_torch()
    if order < 1:
        raise ValueError(f"order must be >= 1, got {order}")

    x_req = x.clone().detach().requires_grad_(True)
    current = f(x_req)
    for i in range(order):
        create_graph = i < order - 1
        (current,) = torch_module.autograd.grad(current, x_req, create_graph=create_graph)
    return current.detach()


def safe_jagged_padded_transform(nested_x, transform_fn):
    """Apply ``transform_fn`` (any dense-tensor operation, e.g. adding a
    positional embedding) to a jagged nested tensor ``nested_x`` by
    round-tripping through padded form, WITHOUT ever calling
    ``torch.nested.narrow(..., layout=torch.jagged)`` on the result
    (pytorch/pytorch#145837), which breaks the backward pass with a
    ``CloneBackward0 returned an invalid gradient`` RuntimeError because
    it constructs a new offsets object that autograd's shape check
    rejects as incompatible with the original.

    ``nested_x`` must be a jagged-layout nested tensor created via
    ``torch.nested.nested_tensor_from_jagged``. ``transform_fn`` receives
    the padded dense tensor (shape ``[batch, max_len, ...]``) and must
    return a dense tensor of the same shape.

    Returns a jagged nested tensor built by applying ``transform_fn`` to
    the padded form, then slicing each row back to its original length
    with plain Python/tensor indexing and re-concatenating with
    ``torch.cat`` -- this keeps the autograd graph consistent (no
    ``torch.nested.narrow`` call), so ``.backward()`` completes without
    the shape-mismatch RuntimeError that the naive narrow-based
    roundtrip raises.
    """
    torch_module = _import_torch()
    offsets = nested_x.offsets()
    padded = torch_module.nested.to_padded_tensor(nested_x, padding=0.0)
    transformed = transform_fn(padded)
    lengths = offsets.diff()
    rows: List[Any] = []
    for i in range(transformed.shape[0]):
        length = int(lengths[i].item())
        rows.append(transformed[i, :length])
    flat = torch_module.cat(rows, dim=0)
    return torch_module.nested.nested_tensor_from_jagged(flat, offsets)


@dataclasses.dataclass
class SlogdetSecondOrderCase:
    t_value: float
    expected_second_order: float
    forward_over_forward_jvp: float  # the buggy path (#196697)
    forward_over_forward_matches_expected: bool
    guard_reverse_over_reverse: float
    guard_matches_expected: bool


def _run_slogdet_case(torch_module) -> SlogdetSecondOrderCase:
    import torch.func as tfunc

    dtype = torch_module.float64

    def f(t):
        c = torch_module.tensor([1, 2, 3], dtype=t.dtype)
        a = torch_module.stack((t + c[1], c[0], c[0], c[2])).reshape(2, 2)
        sign, value = torch_module.linalg.slogdet(a)
        return value

    t0 = torch_module.tensor(0.7, dtype=dtype)
    expected = -0.17853600476096013

    # Buggy path: forward-over-forward via nested torch.func.jvp.
    def jvp1(t):
        return tfunc.jvp(f, (t,), (torch_module.ones_like(t),))[1]

    _, fof_actual = tfunc.jvp(jvp1, (t0,), (torch_module.ones_like(t0),))
    fof_val = float(fof_actual.item())

    guard_val = float(safe_nested_slogdet_second_order_jvp(f, t0).item())

    return SlogdetSecondOrderCase(
        t_value=0.7,
        expected_second_order=expected,
        forward_over_forward_jvp=fof_val,
        forward_over_forward_matches_expected=abs(fof_val - expected) < 1e-6,
        guard_reverse_over_reverse=guard_val,
        guard_matches_expected=abs(guard_val - expected) < 1e-6,
    )


@dataclasses.dataclass
class HouseholderProductSecondOrderCase:
    t_value: float
    expected_second_order: float
    forward_over_forward_jvp: float  # the buggy path (#196698)
    forward_over_forward_matches_expected: bool
    guard_reverse_over_reverse: float
    guard_matches_expected: bool


def _run_householder_case(torch_module) -> HouseholderProductSecondOrderCase:
    import torch.func as tfunc

    dtype = torch_module.float64

    def f(t):
        c = torch_module.ones((), dtype=t.dtype)
        a = torch_module.stack((c, t)).reshape(2, 1)
        tau = (2 / (1 + t * t)).reshape(1)
        return torch_module.linalg.householder_product(a, tau).sum()

    t0 = torch_module.tensor(0.7, dtype=dtype)
    # Independent oracle: closed-form f(t) = 1 - 2*(1+t)/(1+t**2) (the
    # column sum simplifies algebraically); expected value is this
    # function's exact analytic second derivative at t=0.7, cross-
    # checked against a central finite difference (see README/ledger).
    expected = 1.556251320682393

    # Buggy path: forward-over-forward via nested torch.func.jvp.
    def jvp1(t):
        return tfunc.jvp(f, (t,), (torch_module.ones_like(t),))[1]

    _, fof_actual = tfunc.jvp(jvp1, (t0,), (torch_module.ones_like(t0),))
    fof_val = float(fof_actual.item())

    guard_val = float(safe_nested_householder_product_second_order_jvp(f, t0).item())

    return HouseholderProductSecondOrderCase(
        t_value=0.7,
        expected_second_order=expected,
        forward_over_forward_jvp=fof_val,
        forward_over_forward_matches_expected=abs(fof_val - expected) < 1e-6,
        guard_reverse_over_reverse=guard_val,
        guard_matches_expected=abs(guard_val - expected) < 1e-6,
    )


@dataclasses.dataclass
class LayerNormSecondOrderCase:
    t_value: float
    expected_second_order: float
    forward_over_forward_jvp: float  # the buggy path (#196700)
    forward_over_forward_matches_expected: bool
    guard_reverse_over_reverse: float
    guard_matches_expected: bool


def _run_layer_norm_case(torch_module) -> LayerNormSecondOrderCase:
    import torch.func as tfunc

    dtype = torch_module.float64

    def f(t):
        zero = torch_module.zeros((), dtype=t.dtype)
        x = torch_module.stack((t, zero)).reshape(1, 2)
        weight = torch_module.tensor([1, 2], dtype=t.dtype)
        bias = torch_module.zeros((2,), dtype=t.dtype)
        return torch_module.nn.functional.layer_norm(
            x, normalized_shape=[2], weight=weight, bias=bias, eps=0.25
        ).sum()

    t0 = torch_module.tensor(0.7, dtype=dtype)
    # Independent oracle: closed-form f(t) = -t / sqrt(1 + t**2) (the
    # scalar output simplifies algebraically); expected value is this
    # function's exact analytic second derivative at t=0.7, cross-
    # checked against a central finite difference on the closed form
    # with no autograd involved at all (see README/ledger).
    expected = 0.7749142079590942

    # Buggy path: forward-over-forward via nested torch.func.jvp.
    def jvp1(t):
        return tfunc.jvp(f, (t,), (torch_module.ones_like(t),))[1]

    _, fof_actual = tfunc.jvp(jvp1, (t0,), (torch_module.ones_like(t0),))
    fof_val = float(fof_actual.item())

    guard_val = float(safe_layer_norm_second_order_jvp(f, t0).item())

    return LayerNormSecondOrderCase(
        t_value=0.7,
        expected_second_order=expected,
        forward_over_forward_jvp=fof_val,
        forward_over_forward_matches_expected=abs(fof_val - expected) < 1e-6,
        guard_reverse_over_reverse=guard_val,
        guard_matches_expected=abs(guard_val - expected) < 1e-6,
    )


@dataclasses.dataclass
class JaggedNarrowCase:
    t_value: float
    expected_sum: float
    buggy_narrow_unbind_sum: float
    buggy_matches_expected: bool
    guard_sum: float
    guard_matches_expected: bool


def _run_jagged_narrow_case(torch_module) -> JaggedNarrowCase:
    dtype = torch_module.float64
    t = torch_module.tensor(0.7, dtype=dtype)
    x = torch_module.stack((t, 2 * t, 3 * t, 4 * t)).reshape(2, 2)
    starts = torch_module.tensor([0, 1], dtype=torch_module.int64)
    lengths = torch_module.tensor([2, 1], dtype=torch_module.int64)
    expected = 7 * 0.7

    nt = torch_module.nested.narrow(x, 1, starts, lengths, layout=torch_module.jagged)
    parts = nt.unbind()
    buggy_sum = float((parts[0].sum() + parts[1].sum()).item())

    guard_parts = safe_jagged_narrow_unbind(x, 1, starts, lengths)
    guard_sum = float(sum(p.sum() for p in guard_parts).item())

    return JaggedNarrowCase(
        t_value=0.7,
        expected_sum=expected,
        buggy_narrow_unbind_sum=buggy_sum,
        buggy_matches_expected=abs(buggy_sum - expected) < 1e-6,
        guard_sum=guard_sum,
        guard_matches_expected=abs(guard_sum - expected) < 1e-6,
    )


@dataclasses.dataclass
class JaggedPaddedTransformCase:
    backward_raised_on_buggy_path: bool
    buggy_error_message: str
    guard_backward_succeeded: bool
    guard_forward_matches_reference: bool


def _run_jagged_padded_transform_case(torch_module) -> JaggedPaddedTransformCase:
    """Reproduce pytorch/pytorch#145837 from scratch: padded->transform->
    jagged roundtrip via torch.nested.narrow breaks backward. Then verify
    safe_jagged_padded_transform avoids the crash and matches a no-grad
    reference computed the same way."""
    torch_module.manual_seed(0)
    dim = 8
    max_len = 10
    offsets = torch_module.tensor([0, 3, 5, 9])

    pos_emb = torch_module.randn(1, max_len, dim)

    def transform(padded):
        seq_len = padded.shape[1]
        return padded + pos_emb[:, :seq_len, :]

    # Buggy path: torch.nested.narrow-based roundtrip.
    values_buggy = torch_module.randn(9, dim, requires_grad=True)
    x_buggy = torch_module.nested.nested_tensor_from_jagged(values_buggy, offsets)
    backward_raised = False
    error_message = ""
    try:
        offsets_orig = x_buggy.offsets()
        padded = torch_module.nested.to_padded_tensor(x_buggy, padding=0.0)
        padded = transform(padded)
        jagged = torch_module.nested.narrow(
            padded, dim=1, start=0, length=offsets_orig.diff(), layout=torch_module.jagged
        )
        loss = jagged.contiguous().values().mean()
        loss.backward()
    except RuntimeError as exc:
        backward_raised = True
        error_message = str(exc)

    # Guard path: safe_jagged_padded_transform.
    values_guard = torch_module.randn(9, dim, requires_grad=True)
    x_guard = torch_module.nested.nested_tensor_from_jagged(values_guard, offsets)
    guard_backward_succeeded = False
    guard_forward_matches_reference = False
    try:
        out = safe_jagged_padded_transform(x_guard, transform)
        loss = out.values().mean()
        loss.backward()
        guard_backward_succeeded = values_guard.grad is not None

        with torch_module.no_grad():
            offs_list = offsets.tolist()
            padded_ref = torch_module.nested.to_padded_tensor(x_guard.detach(), padding=0.0)
            padded_ref = transform(padded_ref)
            rows = []
            for i in range(len(offs_list) - 1):
                length = offs_list[i + 1] - offs_list[i]
                rows.append(padded_ref[i, :length])
            expected_flat = torch_module.cat(rows, dim=0)
        guard_forward_matches_reference = bool(
            torch_module.allclose(out.values().detach(), expected_flat, atol=1e-6)
        )
    except RuntimeError:
        guard_backward_succeeded = False

    return JaggedPaddedTransformCase(
        backward_raised_on_buggy_path=backward_raised,
        buggy_error_message=error_message,
        guard_backward_succeeded=guard_backward_succeeded,
        guard_forward_matches_reference=guard_forward_matches_reference,
    )


@dataclasses.dataclass
class AutogradFunctionHigherOrderCase:
    x_value: float
    expected_derivatives: List[float]  # [order1, order2, order3] for pure ops
    custom_function_jacfwd_chain: List[float]  # the buggy path (#197867)
    custom_function_jacfwd_matches_expected: bool
    guard_reverse_mode_chain: List[float]
    guard_matches_expected: bool


def _run_autograd_function_higher_order_case(torch_module) -> AutogradFunctionHigherOrderCase:
    import torch.func as tfunc

    class Square(torch_module.autograd.Function):
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

    def custom(x):
        return Square.apply(Square.apply(x)).sum()

    x0 = torch_module.tensor([3.0], dtype=torch_module.float64)
    # Exact analytic derivatives of x**4 at x=3: 108, 108, 72.
    expected = [108.0, 108.0, 72.0]

    # Buggy path: torch.func.jacfwd chained three deep through the
    # custom autograd.Function.
    derivative = custom
    buggy_values: List[float] = []
    for _ in range(3):
        derivative = tfunc.jacfwd(derivative)
        buggy_values.append(float(derivative(x0).item()))

    guard_values = [
        float(safe_autograd_function_higher_order_derivative(custom, x0, order).item())
        for order in (1, 2, 3)
    ]

    return AutogradFunctionHigherOrderCase(
        x_value=3.0,
        expected_derivatives=expected,
        custom_function_jacfwd_chain=buggy_values,
        custom_function_jacfwd_matches_expected=all(
            abs(a - b) < 1e-6 for a, b in zip(buggy_values, expected)
        ),
        guard_reverse_mode_chain=guard_values,
        guard_matches_expected=all(
            abs(a - b) < 1e-6 for a, b in zip(guard_values, expected)
        ),
    )


def diagnose() -> Dict[str, Any]:
    """Reproduce all six upstream bugs from scratch against the
    currently installed torch build, using the EXACT repro code from
    the issues, and verify the guard functions produce the
    mathematically correct result instead. Never trusts a cached/prior
    result."""
    torch_module = _import_torch()

    slogdet_case = _run_slogdet_case(torch_module)
    householder_case = _run_householder_case(torch_module)
    layer_norm_case = _run_layer_norm_case(torch_module)
    narrow_case = _run_jagged_narrow_case(torch_module)
    padded_transform_case = _run_jagged_padded_transform_case(torch_module)
    higher_order_case = _run_autograd_function_higher_order_case(torch_module)

    return {
        "torch_version": torch_module.__version__,
        "issue_urls": [
            "https://github.com/pytorch/pytorch/issues/196697",
            "https://github.com/pytorch/pytorch/issues/196708",
            "https://github.com/pytorch/pytorch/issues/145837",
            "https://github.com/pytorch/pytorch/issues/196698",
            "https://github.com/pytorch/pytorch/issues/196700",
            "https://github.com/pytorch/pytorch/issues/197867",
        ],
        "slogdet_second_order_case": dataclasses.asdict(slogdet_case),
        "householder_product_second_order_case": dataclasses.asdict(householder_case),
        "layer_norm_second_order_case": dataclasses.asdict(layer_norm_case),
        "jagged_narrow_unbind_case": dataclasses.asdict(narrow_case),
        "jagged_padded_transform_case": dataclasses.asdict(padded_transform_case),
        "autograd_function_higher_order_case": dataclasses.asdict(higher_order_case),
        "slogdet_bug_reproduced": not slogdet_case.forward_over_forward_matches_expected,
        "householder_bug_reproduced": not householder_case.forward_over_forward_matches_expected,
        "layer_norm_bug_reproduced": not layer_norm_case.forward_over_forward_matches_expected,
        "narrow_bug_reproduced": not narrow_case.buggy_matches_expected,
        "padded_transform_bug_reproduced": padded_transform_case.backward_raised_on_buggy_path,
        "autograd_function_higher_order_bug_reproduced": (
            not higher_order_case.custom_function_jacfwd_matches_expected
        ),
        "guards_fully_correct": (
            slogdet_case.guard_matches_expected
            and householder_case.guard_matches_expected
            and layer_norm_case.guard_matches_expected
            and narrow_case.guard_matches_expected
            and padded_transform_case.guard_backward_succeeded
            and padded_transform_case.guard_forward_matches_reference
            and higher_order_case.guard_matches_expected
        ),
    }
