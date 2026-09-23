# torch-correctness-guards

A consolidated suite of focused diagnostics and call-site workarounds for real PyTorch correctness bugs.

The package replaces the older one-repo-per-guard pattern with one installable CLI and stable Python import surface.

## Included guards

| Guard | CLI name | Upstream issue | What it checks |
|---|---|---|---|
| addcdiv stale scalar | `addcdiv-stale-scalar` | pytorch/pytorch#185382 | Reproduces an Inductor stale Python-scalar cache bug in Adam-style `addcdiv_`/`addcmul_` arithmetic and verifies the eager call-site guard. |
| as_strided restride OOB | `as-strided-restride-oob` | pytorch/pytorch#197431, pytorch/pytorch#192226 | Reproduces an Inductor storage-span miscalculation for `torch.as_strided` on repeated+sliced restrided views and verifies the eager call-site guard. |
| checkpoint noise | `checkpoint-noise` | pytorch/pytorch#193671 | Reproduces `F.rrelu(..., training=True)` silently wrong gradients under non-reentrant checkpointing or saved-tensors hooks and verifies the `safe_rrelu` replacement. |
| compile validation | `compile-validation` | pytorch/pytorch#185246, pytorch/pytorch#185248, pytorch/pytorch#193757, pytorch/pytorch#194548, pytorch/pytorch#193811 | Reproduces `torch.compile` accepting out-of-domain inputs that eager rejects for bernoulli, normal, BCE, categorical sampling, and bilinear upsampling, then verifies guard wrappers restore eager validation. |
| CPU backward NaN tail | `cpu-backward-nan-tail` | pytorch/pytorch#195075 | Reproduces CPU backward kernels returning different gradients at NaN elements in SIMD vector blocks versus scalar tails, then verifies length-independent `safe_*_backward` guards. |
| Dynamo closure descriptor | `dynamo-closure-descriptor` | pytorch/pytorch#197811, pytorch/pytorch#197860, pytorch/pytorch#197859 | Reproduces Dynamo stale graph reuse for closure-captured Tensor method descriptors, bound-method `__code__` mutation, and `nn.Module` instance-`__dict__` submodule shadowing; verifies graph-break call-site guards restore eager semantics. |
| dtype-view scatter | `dtype-view-scatter` | pytorch/pytorch#197408 | Reproduces Inductor value corruption and return-aliasing drift for dtype-view + custom-op + `diagonal_scatter`, then verifies `safe_compiled_dtype_view_diagonal_scatter` restores eager's values and non-aliasing contract. |
| duplicate index write-order | `duplicate-index-writeorder` | pytorch/pytorch#197582 | Reproduces Inductor miscompiling `v[idx] = v[idx] + delta` when `idx` is a computed duplicate index such as `torch.arange(n) % 2`, then verifies `safe_dup_index_assign` restores eager semantics. |
| dynamic clamp | `dynamic-clamp` | pytorch/pytorch#194976, nvidia/Megatron-LM#6918 | Reproduces Inductor stale reuse of automatically-dynamic Python float bounds in `torch.clamp` and verifies the `safe_clamp` call-site guard. |
| equality fusion | `equality-fusion` | pytorch/pytorch#195214 | Reproduces Inductor fusing a low-precision division into a following equality/argmax comparison without preserving the eager rounding boundary, then verifies `precision_safe_division_compare` restores eager tie counts. |
| embedding_bag frequency scaling | `embeddingbag-freq-scale` | pytorch/pytorch#190061 | Reproduces the MPS backend silently ignoring `scale_grad_by_freq=True` in `embedding_bag` backward and verifies `safe_embedding_bag` matches the CPU-oracle gradient. |
| expand fill | `expand-fill` | pytorch/pytorch#197448 | Reproduces Inductor writing wrong values for `.fill_(scalar)` on a broadcast view created by `Tensor.expand()` and verifies `safe_fill_` matches eager. |
| fp16 LayerNorm tail | `fp16-layernorm-tail` | none filed | Reproduces CPU float16 `layer_norm` returning nonzero output for exact-constant rows and verifies `safe_layer_norm` computes through a float32 upcast. |
| full dtype | `full-dtype` | pytorch/pytorch#194062 | Reproduces Inductor dropping `torch.full(..., dtype=...)` casts for symbolic fills, including skipped narrow-integer overflow checks, and verifies `safe_full` restores eager execution. |
| in-place slice-shift aliasing | `inplace-slice-shift-aliasing` | pytorch/pytorch#197829 | Reproduces Inductor corrupting `x[1:] = x[:-1].clone()` style overlapping row-shift assignments and verifies `safe_slice_shift` restores eager semantics. |
| int64 index truncation | `int64-index-truncation` | pytorch/pytorch#183901 | Reproduces Inductor truncating int64 `arange`-multiply expressions to 32-bit-range arithmetic before storing an int64 result, then verifies eager graph-break guards restore exact values. |
| linalg NaN/Inf guards | `linalg-nan` | pytorch/pytorch#187759, numpy/numpy#20280, pytorch/pytorch#169237, pytorch/pytorch#193006 | Reproduces `torch.linalg.svdvals()` / `eigvalsh()` values-only routines silently swallowing non-finite input and verifies safe finite-input guards; also ships precision/overflow-safe L2 norm wrappers. |
| linalg complex pinv gradients | `linalg-pinv-complex-grad` | pytorch/pytorch#197084 | Reproduces AOTAutograd backends computing wrong gradients for complex `torch.linalg.pinv()` / `matrix_sqrth` under `torch.compile`, then verifies `safe_complex_pinv_grad` graph-breaks the call-site back to eager gradients. |
| memory-budget RNG recompute | `memory-budget-rng` | pytorch/pytorch#190758 | Reproduces AOTAutograd recomputing RNG ops with fresh randomness when `activation_memory_budget < 1.0`, then verifies `safe_compile` temporarily forces budget=1.0 and restores the caller's setting. |
| MPS copy dtype | `mps-copy-dtype` | pytorch/pytorch#197715 | Reproduces Apple Silicon MPS-to-CPU copies into `float64`/`complex128` silently losing data and verifies `safe_to` / `safe_copy_` perform the transfer before the CPU-side dtype conversion. |
| MPS linalg stride | `mps-linalg-stride` | pytorch/pytorch#197236 | Reproduces Apple Silicon MPS linalg solves returning row-major output where CPU/CUDA return column-major, then verifies `safe_solve_triangular` / `safe_cholesky_solve` / `safe_solve` normalize eager output layout without changing values. |
| multi-output aliasing | `multioutput-alias` | pytorch/pytorch#195338 | Reproduces multi-output `out=(t, t)` aliasing for `torch.aminmax` and `torch.linalg.slogdet`, then verifies `safe_aminmax` / `safe_slogdet` raise before a result slot is silently overwritten. |
| native dropout train=None | `native-dropout-train-none` | pytorch/pytorch#197846 | Reproduces `torch.native_dropout(..., train=None)` disagreeing between CPU eager and `torch.compile`, documents the related MPS eager divergence, and verifies `safe_native_dropout` coerces the ambiguous `None` before compile. |
| nested AD / jagged narrow | `nested-ad-narrow` | pytorch/pytorch#196697, #196698, #196700, #196708, #145837, #197867 | Reproduces nested forward-mode AD wrong derivatives, jagged `torch.nested.narrow` selection/backward failures, and custom `autograd.Function` higher-order `jacfwd` zeroing; verifies reverse-mode and per-row-slicing guards. |
| torch._numpy stream shuffle | `numpy-stream-shuffle` | pytorch/pytorch#197795 | Reproduces `torch._numpy.random.shuffle` under `use_numpy_random_stream=True` corrupting a tensor's row multiset, then verifies `safe_row_shuffle` / `safe_row_shuffle_` preserve true permutation semantics. |
| optimizer introspection | `optim-introspection` | pytorch/pytorch#164929 | Reproduces `get_optimizer_state_dict()` mutating optimizer step counters during a supposedly read-only inspection, then verifies `safe_get_optimizer_state_dict` snapshots and restores optimizer state. |
| take_along_dim OOB | `take-along-dim-oob` | pytorch/pytorch#196106 | Reproduces `torch.take_along_dim(input, indices, dim=<int>)` silently wrapping out-of-bounds indices modulo the dimension instead of raising, then verifies `safe_take_along_dim` restores the shared raise-on-OOB contract while preserving valid negative indices. |
| scatter copy-back aliasing | `scatter-copyback-alias` | pytorch/pytorch#195451, pytorch/pytorch#197893 | Reproduces Inductor changing a compiled function's return-value aliasing contract for scatter copy-back and no-op-elimination rewrites, then verifies `safe_compiled_scatter_returning` restores eager's non-aliasing behavior. |
| softmax dim attention rewrite | `softmax-dim` | pytorch/pytorch#196468 | Reproduces Inductor rewriting attention-shaped `matmul -> softmax(dim=non-last) -> matmul` graphs as if `dim=-1`, then verifies `safe_softmax_attention` preserves eager semantics. |
| 2D tiled reduction tail store | `tiled-reduction-tail-store` | pytorch/pytorch#196681 | Reproduces Inductor's CPU 2D-tiled reduction tail-store overrun/corruption in isolated subprocesses and verifies `safe_compiled_reduction` never silently trusts a corrupted result. |
| Normal.sample dtype promotion | `normal-dtype-promotion` | pytorch/pytorch#194547 | Reproduces `torch.compile` silently promoting `torch.distributions.Normal.sample()` output dtype when eager preserves the lower-precision `loc` dtype, and verifies the wrapper restores eager's dtype contract. |
| shuffle/sample frozen RNG | `shuffle-sample-frozen` | pytorch/pytorch#197085 | Reproduces Dynamo baking `random.shuffle()` / `random.sample()` results into a compiled graph as trace-time constants and verifies graph-break wrappers restore eager per-call randomness. |
| std/var precision | `std-precision` | pytorch/pytorch#197089 | Reproduces `torch.compile(backend="inductor")` accumulating `torch.std`/`torch.var`-family reductions in float32 where CPU eager uses double precision, and verifies float64-upcast guards preserve eager outputs and gradients. |
| transpose argmin/argmax | `transpose-argmin` | pytorch/pytorch#197739 | Reproduces Inductor returning a wrong flat `argmin()`/`argmax()` index after `x.t()` plus an intervening op such as `+ 0.5` or `.contiguous()`, and verifies the eager reduction guard. |

## Install

```bash
python -m pip install -e ".[torch]"
```

For development:

```bash
python -m pip install -e ".[dev,torch]"
python -m pytest -q
```

## CLI

```bash
torch-guard list
torch-guard run addcdiv-stale-scalar
torch-guard run as-strided-restride-oob
torch-guard run checkpoint-noise
torch-guard run compile-validation
torch-guard run cpu-backward-nan-tail
torch-guard run dynamo-closure-descriptor
torch-guard run dtype-view-scatter
torch-guard run duplicate-index-writeorder
torch-guard run dynamic-clamp
torch-guard run equality-fusion
torch-guard run embeddingbag-freq-scale
torch-guard run expand-fill
torch-guard run fp16-layernorm-tail
torch-guard run full-dtype
torch-guard run inplace-slice-shift-aliasing
torch-guard run int64-index-truncation
torch-guard run linalg-nan
torch-guard run linalg-pinv-complex-grad
torch-guard run memory-budget-rng
torch-guard run mps-copy-dtype
torch-guard run mps-linalg-stride
torch-guard run multioutput-alias
torch-guard run native-dropout-train-none
torch-guard run nested-ad-narrow
torch-guard run numpy-stream-shuffle
torch-guard run optim-introspection
torch-guard run take-along-dim-oob
torch-guard run scatter-copyback-alias
torch-guard run softmax-dim
torch-guard run tiled-reduction-tail-store
torch-guard run normal-dtype-promotion
torch-guard run shuffle-sample-frozen
torch-guard run std-precision
torch-guard run transpose-argmin
torch-guard run addcdiv-stale-scalar --json
```

Exit codes for `run` describe the guard check: `0` means every guard case matched eager, `1` means a guard check failed, and `2` means torch could not be imported.

## Python API

```python
from torch_correctness_guards import make_safe_stale_scalar_step
from torch_correctness_guards import safe_as_strided
from torch_correctness_guards import safe_rrelu
from torch_correctness_guards import safe_compiled_bernoulli
from torch_correctness_guards import safe_hardtanh_backward, safe_logit_backward
from torch_correctness_guards import safe_bound_method_call, safe_call, safe_module_forward
from torch_correctness_guards import safe_compiled_dtype_view_diagonal_scatter
from torch_correctness_guards import safe_dup_index_assign
from torch_correctness_guards import safe_clamp
from torch_correctness_guards import precision_safe_division_compare
from torch_correctness_guards import safe_embedding_bag
from torch_correctness_guards import safe_fill_
from torch_correctness_guards import safe_layer_norm
from torch_correctness_guards import safe_full
from torch_correctness_guards import safe_slice_shift
from torch_correctness_guards import safe_compiled_reduction
from torch_correctness_guards import safe_int64_arange_mul
from torch_correctness_guards import safe_svdvals, safe_eigvalsh, safe_vector_norm, safe_norm
from torch_correctness_guards import safe_complex_pinv_grad
from torch_correctness_guards import safe_compile
from torch_correctness_guards import safe_copy_, safe_to
from torch_correctness_guards import safe_cholesky_solve, safe_solve, safe_solve_triangular
from torch_correctness_guards import safe_aminmax, safe_slogdet
from torch_correctness_guards import safe_native_dropout
from torch_correctness_guards import safe_jagged_narrow_unbind, safe_jagged_padded_transform
from torch_correctness_guards import safe_nested_slogdet_second_order_jvp
from torch_correctness_guards import safe_row_shuffle, safe_row_shuffle_
from torch_correctness_guards import safe_get_optimizer_state_dict
from torch_correctness_guards import safe_take_along_dim
from torch_correctness_guards import safe_compiled_scatter_returning
from torch_correctness_guards import safe_softmax_attention
from torch_correctness_guards import safe_compiled_normal_sample
from torch_correctness_guards import safe_sample, safe_shuffle
from torch_correctness_guards import safe_std, safe_var, safe_var_mean, safe_std_mean
from torch_correctness_guards import safe_reduce_index

safe_bias_correction = make_safe_stale_scalar_step(torch)
y = safe_as_strided(sliced, size, stride, storage_offset=None)
z = safe_rrelu(x, lower=0.125, upper=1 / 3, training=True)
grad = safe_hardtanh_backward(grad_output, x)
y = safe_call(torch.Tensor.__mul__, a, b)
w = safe_clamp(x, max=limit)
divide = precision_safe_division_compare(lambda x, const: x / const)
emb = safe_embedding_bag(idx, weight, offsets, mode="sum", scale_grad_by_freq=True)
safe_fill_(x.expand(3, -1), 2.0)
ln = safe_layer_norm(x, (x.shape[-1],))
mask = safe_full((2,), fill_value, dtype=torch.bool)
safe_slice_shift(x, shift=1, dim=0)
checked = safe_compiled_reduction(compiled_fn, eager_fn)(x)
products = safe_int64_arange_mul(0, 9, torch.tensor(1500000000, dtype=torch.int64))
vals = safe_svdvals(matrix)
safe_pinv = safe_complex_pinv_grad(lambda a: torch.linalg.pinv(a).sum().abs())
safe_model = safe_compile(model)
cpu64 = safe_to(mps_tensor, "cpu", torch.float64)
safe_copy_(cpu64_buffer, mps_tensor)
sol = safe_solve(A_mps, B_mps)
mn, mx = safe_aminmax(x)
sign, logabsdet = safe_slogdet(matrix)
native_dropout = safe_native_dropout(lambda x, p, train: torch.native_dropout(x, p, train))
second = safe_nested_slogdet_second_order_jvp(f, t0)
parts = safe_jagged_narrow_unbind(dense_x, 1, starts, lengths)
shuffled = safe_row_shuffle(x)
state = safe_get_optimizer_state_dict(model, optimizer)
picked = safe_take_along_dim(x, indices, dim=0)
safe_fn = safe_compiled_scatter_returning(compiled_fn)
attn = safe_softmax_attention(q, k, v, dim=0)
safe_dup_index_assign(v, idx, 1.0)
sample = safe_compiled_normal_sample(compiled_fn, eager_fn)(loc, scale)
safe_shuffle(items)
subset = safe_sample(population, k)
std = safe_std(x)
idx = safe_reduce_index(x.t(), op="add", mode="argmin")
```

## License

MIT. See [LICENSE](LICENSE).
