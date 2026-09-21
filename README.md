# torch-correctness-guards

A consolidated suite of focused diagnostics and call-site workarounds for real PyTorch correctness bugs.

The package replaces the older one-repo-per-guard pattern with one installable CLI and stable Python import surface.

## Included guards

| Guard | CLI name | Upstream issue | What it checks |
|---|---|---|---|
| addcdiv stale scalar | `addcdiv-stale-scalar` | pytorch/pytorch#185382 | Reproduces an Inductor stale Python-scalar cache bug in Adam-style `addcdiv_`/`addcmul_` arithmetic and verifies the eager call-site guard. |

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
torch-guard run addcdiv-stale-scalar --json
```

Exit codes for `run` describe the guard check: `0` means every guard case matched eager, `1` means a guard check failed, and `2` means torch could not be imported.

## Python API

```python
from torch_correctness_guards import make_safe_stale_scalar_step

safe_bias_correction = make_safe_stale_scalar_step(torch)
```

## License

MIT. See [LICENSE](LICENSE).
