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
safe_dup_index_assign(v, idx, 1.0)
sample = safe_compiled_normal_sample(compiled_fn, eager_fn)(loc, scale)
safe_shuffle(items)
subset = safe_sample(population, k)
std = safe_std(x)
idx = safe_reduce_index(x.t(), op="add", mode="argmin")
```

## License

MIT. See [LICENSE](LICENSE).
