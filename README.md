# torch-correctness-guards

A consolidated suite of focused diagnostics and call-site workarounds for real PyTorch correctness bugs.

The package replaces the older one-repo-per-guard pattern with one installable CLI and stable Python import surface.

## Included guards

| Guard | CLI name | Upstream issue | What it checks |
|---|---|---|---|
| addcdiv stale scalar | `addcdiv-stale-scalar` | pytorch/pytorch#185382 | Reproduces an Inductor stale Python-scalar cache bug in Adam-style `addcdiv_`/`addcmul_` arithmetic and verifies the eager call-site guard. |
| as_strided restride OOB | `as-strided-restride-oob` | pytorch/pytorch#197431, pytorch/pytorch#192226 | Reproduces an Inductor storage-span miscalculation for `torch.as_strided` on repeated+sliced restrided views and verifies the eager call-site guard. |
| checkpoint noise | `checkpoint-noise` | pytorch/pytorch#193671 | Reproduces `F.rrelu(..., training=True)` silently wrong gradients under non-reentrant checkpointing or saved-tensors hooks and verifies the `safe_rrelu` replacement. |
| dynamic clamp | `dynamic-clamp` | pytorch/pytorch#194976, nvidia/Megatron-LM#6918 | Reproduces Inductor stale reuse of automatically-dynamic Python float bounds in `torch.clamp` and verifies the `safe_clamp` call-site guard. |
| Normal.sample dtype promotion | `normal-dtype-promotion` | pytorch/pytorch#194547 | Reproduces `torch.compile` silently promoting `torch.distributions.Normal.sample()` output dtype when eager preserves the lower-precision `loc` dtype, and verifies the wrapper restores eager's dtype contract. |
| shuffle/sample frozen RNG | `shuffle-sample-frozen` | pytorch/pytorch#197085 | Reproduces Dynamo baking `random.shuffle()` / `random.sample()` results into a compiled graph as trace-time constants and verifies graph-break wrappers restore eager per-call randomness. |
| std/var precision | `std-precision` | pytorch/pytorch#197089 | Reproduces `torch.compile(backend="inductor")` accumulating `torch.std`/`torch.var`-family reductions in float32 where CPU eager uses double precision, and verifies float64-upcast guards preserve eager outputs and gradients. |

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
torch-guard run dynamic-clamp
torch-guard run normal-dtype-promotion
torch-guard run shuffle-sample-frozen
torch-guard run std-precision
torch-guard run addcdiv-stale-scalar --json
```

Exit codes for `run` describe the guard check: `0` means every guard case matched eager, `1` means a guard check failed, and `2` means torch could not be imported.

## Python API

```python
from torch_correctness_guards import make_safe_stale_scalar_step
from torch_correctness_guards import safe_as_strided
from torch_correctness_guards import safe_rrelu
from torch_correctness_guards import safe_clamp
from torch_correctness_guards import safe_compiled_normal_sample
from torch_correctness_guards import safe_sample, safe_shuffle
from torch_correctness_guards import safe_std, safe_var, safe_var_mean, safe_std_mean

safe_bias_correction = make_safe_stale_scalar_step(torch)
y = safe_as_strided(sliced, size, stride, storage_offset=None)
z = safe_rrelu(x, lower=0.125, upper=1 / 3, training=True)
w = safe_clamp(x, max=limit)
sample = safe_compiled_normal_sample(compiled_fn, eager_fn)(loc, scale)
safe_shuffle(items)
subset = safe_sample(population, k)
std = safe_std(x)
```

## License

MIT. See [LICENSE](LICENSE).
