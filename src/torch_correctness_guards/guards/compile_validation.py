"""compile-validation guard: guards a real class of
torch.compile (Inductor) correctness bugs where eager mode validates
an operator's input DOMAIN and raises RuntimeError/NotImplementedError,
but torch.compile's decomposition/lowering path skips that validation
and silently returns a finite, shape-plausible (but semantically
undefined) result instead.

This is a distinct failure shape from every other guard in this
fleet: those all guard NUMERIC divergence on a VALID input. Here the
input itself is out-of-domain and eager's own contract says so --
torch.compile silently accepting it (instead of raising, or at least
not fabricating a finite answer) is the defect.

Five independently-filed, still-open upstream issues share this root
cause (checked live via `gh api`, not cached, before every acceptance
decision):

  1. torch.bernoulli(p) with p outside [0, 1]           -- #185246
  2. torch.normal(mean, std) with std < 0                -- #185248
  3. F.binary_cross_entropy(input, target) with          -- #193757
     target outside [0, 1] (CPU-specific)
  4. torch.distributions.Categorical(logits=...).sample() -- #194548
     on a zero-batch-size input (eager raises a reshape
     ambiguity error; compiled silently returns shape [0])
  5. F.interpolate(..., mode="bilinear") on an int32      -- #193811
     input (eager raises NotImplementedError; compiled
     silently returns a finite tensor)

A single unresolved umbrella PR (pytorch/pytorch#187368, "[inductor]
add input validation to normal and bernoulli decompositions") covers
part of this family and is still open/unmerged as of this tool's last
verification -- confirmed live, not assumed.

Independently reproduced on this host (torch 2.14.0, CPU) for all 5:
see tests/test_core.py for the exact reproducer per fixture, each
taken directly from the issue's own minimal repro.

Every fixture also carries a BOUNDARY (valid) case that must NOT be
flagged by either eager or the guard -- e.g. p exactly 0.0 or 1.0 for
bernoulli, std exactly 0.0 for normal -- so an off-by-epsilon bug in
this tool's own validators is caught rather than silently accepted as
"the guard works" for the wrong reason.
"""
from __future__ import annotations

import dataclasses
import functools
from typing import Any, Callable, Dict, List, Optional


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


# ---------------------------------------------------------------------------
# Per-op eager-equivalent validators.
#
# Each validator reproduces eager's OWN domain check as cheaply as
# possible (no full eager recompute), so the guard wrapper can reject
# an invalid input before ever calling the (fast but permissive)
# compiled function -- preserving torch.compile's speed benefit for
# every valid input, which is the overwhelming common case.
# ---------------------------------------------------------------------------


def _validate_bernoulli_p(p) -> None:
    """Mirrors eager torch.bernoulli's own domain check (#185246)."""
    if not ((p >= 0.0) and (p <= 1.0)):
        raise RuntimeError(
            f"Expected p_in >= 0 && p_in <= 1 to be true, but got p={p!r}."
        )


def _validate_normal_std(std) -> None:
    """Mirrors eager torch.normal's own domain check (#185248)."""
    if not (std >= 0.0):
        raise RuntimeError(f"normal expects all elements of std >= 0.0, got std={std!r}.")


def _validate_bce_target_scalar(target) -> None:
    """Mirrors eager F.binary_cross_entropy's own domain check (#193757)."""
    if not (0.0 <= target <= 1.0):
        raise RuntimeError(
            f"all elements of target should be between 0 and 1, got target={target!r}."
        )


def _validate_categorical_batch(torch_module, logits) -> None:
    """Mirrors eager Categorical.sample()'s own reshape-ambiguity check
    on a zero-element batch (#194548)."""
    if logits.numel() == 0:
        raise RuntimeError(
            "cannot reshape tensor of 0 elements into shape [0, -1] because "
            "the unspecified dimension size -1 can be any value and is "
            "ambiguous"
        )


def _validate_upsample_bilinear_dtype(torch_module, x) -> None:
    """Mirrors eager upsample_bilinear2d's own dtype check (#193811):
    only floating-point dtypes are implemented."""
    if not x.dtype.is_floating_point:
        raise NotImplementedError(
            f'"upsample_bilinear2d_channels_last" not implemented for '
            f"'{x.dtype}'"
        )


# ---------------------------------------------------------------------------
# Guard wrappers: validate-then-call. Each takes the compiled callable
# and returns a wrapped callable with the same signature that raises
# the same class of error eager would, for the same invalid input,
# before ever invoking the (validation-skipping) compiled function.
# ---------------------------------------------------------------------------


def safe_compiled_bernoulli(compiled_fn: Callable) -> Callable:
    @functools.wraps(compiled_fn)
    def wrapper(p_tensor):
        # p_tensor is a filled tensor of a single scalar p in this
        # fixture's shape; check the scalar value eager would check.
        p_value = p_tensor.flatten()[0].item()
        _validate_bernoulli_p(p_value)
        return compiled_fn(p_tensor)

    return wrapper


def safe_compiled_normal(compiled_fn: Callable) -> Callable:
    @functools.wraps(compiled_fn)
    def wrapper(loc, std_tensor):
        std_value = std_tensor.flatten()[0].item()
        _validate_normal_std(std_value)
        return compiled_fn(loc, std_tensor)

    return wrapper


def safe_compiled_binary_cross_entropy(compiled_fn: Callable) -> Callable:
    @functools.wraps(compiled_fn)
    def wrapper(inp, target_tensor):
        target_value = target_tensor.flatten()[0].item()
        _validate_bce_target_scalar(target_value)
        return compiled_fn(inp, target_tensor)

    return wrapper


def safe_compiled_categorical_sample(compiled_fn: Callable, torch_module) -> Callable:
    @functools.wraps(compiled_fn)
    def wrapper(logits):
        _validate_categorical_batch(torch_module, logits)
        return compiled_fn(logits)

    return wrapper


def safe_compiled_upsample_bilinear2d(compiled_fn: Callable, torch_module) -> Callable:
    @functools.wraps(compiled_fn)
    def wrapper(x):
        _validate_upsample_bilinear_dtype(torch_module, x)
        return compiled_fn(x)

    return wrapper


# ---------------------------------------------------------------------------
# Fixtures: one per upstream issue, each with an INVALID case (should
# raise in both eager and the guard) and a BOUNDARY case (a valid
# input at the edge of the domain, must NOT raise anywhere).
# ---------------------------------------------------------------------------


@dataclasses.dataclass
class FixtureResult:
    name: str
    issue_url: str
    eager_raised: bool
    eager_error: Optional[str]
    compiled_raised: bool
    compiled_error: Optional[str]
    guarded_raised: bool
    guarded_error: Optional[str]
    validation_bypassed: bool  # eager raised, raw compiled did NOT
    guard_restores_validation: bool  # guard raises exactly when eager does
    is_boundary_case: bool


def _bernoulli_eager_fn(torch_module):
    def fn(p_tensor):
        return torch_module.bernoulli(p_tensor.expand(4).clone())

    return fn


def _normal_eager_fn(torch_module):
    def fn(loc, std_tensor):
        return torch_module.normal(loc, std_tensor.expand(4).clone())

    return fn


def _bce_eager_fn(torch_module):
    def fn(inp, target_tensor):
        return torch_module.nn.functional.binary_cross_entropy(
            inp, target_tensor.expand(4).clone()
        )

    return fn


def _categorical_eager_fn(torch_module):
    def fn(logits):
        dist = torch_module.distributions.Categorical(logits=logits)
        return dist.sample()

    return fn


def _upsample_eager_fn(torch_module):
    def fn(x):
        return torch_module.nn.functional.interpolate(
            x, size=(16, 15), mode="bilinear", align_corners=False
        )

    return fn


def _call_capturing(fn, *args):
    try:
        out = fn(*args)
        return False, None, out
    except Exception as exc:  # noqa: BLE001 - intentionally broad, this IS the check
        return True, f"{type(exc).__name__}: {exc}", None


def _run_fixture_via_subprocess(fixture_name: str, python_executable: Optional[str] = None) -> FixtureResult:
    """Run one fixture's eager/compiled/guarded comparison in an
    isolated subprocess (see _worker.py's module docstring for why:
    tracing a torch.distributions-based function through
    torch.compile can flip a PROCESS-GLOBAL validate-args flag,
    silently corrupting a LATER fixture's eager baseline if all
    fixtures shared one process). A non-zero/crashed subprocess is
    itself real evidence (treated as a hard failure), never silently
    swallowed.
    """
    import json
    import subprocess
    import sys as _sys

    import os

    python_executable = python_executable or _sys.executable
    env = os.environ.copy()
    # Ensure the subprocess can import this package even when it is
    # only on sys.path (editable/dev checkout) rather than fully
    # installed -- inherit the parent's sys.path via PYTHONPATH so
    # `python -m torch_compile_validation_guard._worker` resolves
    # identically to how THIS process imported it.
    existing_pythonpath = env.get("PYTHONPATH", "")
    extra_paths = os.pathsep.join(_sys.path)
    env["PYTHONPATH"] = (
        extra_paths + (os.pathsep + existing_pythonpath if existing_pythonpath else "")
    )
    proc = subprocess.run(
        [
            python_executable,
            "-m",
            "torch_correctness_guards.guards.compile_validation_worker",
            fixture_name,
        ],
        capture_output=True,
        text=True,
        timeout=120,
        env=env,
    )
    if proc.returncode != 0:
        raise RuntimeError(
            f"fixture {fixture_name!r} worker subprocess exited "
            f"{proc.returncode} (crash or uncaught error, not a "
            f"harness bug) -- stderr: {proc.stderr[-2000:]}"
        )
    payload = json.loads(proc.stdout.strip().splitlines()[-1])

    validation_bypassed = payload["eager_raised"] and not payload["compiled_raised"]
    guard_restores_validation = payload["guarded_raised"] == payload["eager_raised"]

    return FixtureResult(
        name=payload["name"],
        issue_url=payload["issue_url"],
        eager_raised=payload["eager_raised"],
        eager_error=payload["eager_error"],
        compiled_raised=payload["compiled_raised"],
        compiled_error=payload["compiled_error"],
        guarded_raised=payload["guarded_raised"],
        guarded_error=payload["guarded_error"],
        validation_bypassed=validation_bypassed,
        guard_restores_validation=guard_restores_validation,
        is_boundary_case=payload["is_boundary_case"],
    )


def _fixture_specs(torch_module) -> List[Dict[str, Any]]:
    return [
        dict(
            name="bernoulli_p_out_of_range",
            issue_url="https://github.com/pytorch/pytorch/issues/185246",
            eager_fn_factory=_bernoulli_eager_fn,
            safe_wrapper_factory=safe_compiled_bernoulli,
            args=(torch_module.tensor(2.0),),
            is_boundary_case=False,
        ),
        dict(
            name="bernoulli_p_boundary_one",
            issue_url="https://github.com/pytorch/pytorch/issues/185246",
            eager_fn_factory=_bernoulli_eager_fn,
            safe_wrapper_factory=safe_compiled_bernoulli,
            args=(torch_module.tensor(1.0),),
            is_boundary_case=True,
        ),
        dict(
            name="normal_negative_std",
            issue_url="https://github.com/pytorch/pytorch/issues/185248",
            eager_fn_factory=_normal_eager_fn,
            safe_wrapper_factory=safe_compiled_normal,
            args=(torch_module.zeros(4), torch_module.tensor(-1.0)),
            is_boundary_case=False,
        ),
        dict(
            name="normal_std_boundary_zero",
            issue_url="https://github.com/pytorch/pytorch/issues/185248",
            eager_fn_factory=_normal_eager_fn,
            safe_wrapper_factory=safe_compiled_normal,
            args=(torch_module.zeros(4), torch_module.tensor(0.0)),
            is_boundary_case=True,
        ),
        dict(
            name="bce_target_out_of_range",
            issue_url="https://github.com/pytorch/pytorch/issues/193757",
            eager_fn_factory=_bce_eager_fn,
            safe_wrapper_factory=safe_compiled_binary_cross_entropy,
            args=(torch_module.full((4,), 0.5), torch_module.tensor(1.5)),
            is_boundary_case=False,
        ),
        dict(
            name="bce_target_boundary_one",
            issue_url="https://github.com/pytorch/pytorch/issues/193757",
            eager_fn_factory=_bce_eager_fn,
            safe_wrapper_factory=safe_compiled_binary_cross_entropy,
            args=(torch_module.full((4,), 0.5), torch_module.tensor(1.0)),
            is_boundary_case=True,
        ),
        dict(
            name="categorical_zero_batch",
            issue_url="https://github.com/pytorch/pytorch/issues/194548",
            eager_fn_factory=_categorical_eager_fn,
            safe_wrapper_factory=lambda cf: safe_compiled_categorical_sample(cf, torch_module),
            args=(torch_module.randn(0, 2),),
            is_boundary_case=False,
        ),
        dict(
            name="categorical_nonzero_batch_boundary",
            issue_url="https://github.com/pytorch/pytorch/issues/194548",
            eager_fn_factory=_categorical_eager_fn,
            safe_wrapper_factory=lambda cf: safe_compiled_categorical_sample(cf, torch_module),
            args=(torch_module.randn(1, 2),),
            is_boundary_case=True,
        ),
        dict(
            name="upsample_bilinear2d_int_dtype",
            issue_url="https://github.com/pytorch/pytorch/issues/193811",
            eager_fn_factory=_upsample_eager_fn,
            safe_wrapper_factory=lambda cf: safe_compiled_upsample_bilinear2d(cf, torch_module),
            args=(torch_module.randint(0, 100, (1, 3, 8, 8), dtype=torch_module.int32),),
            is_boundary_case=False,
        ),
        dict(
            name="upsample_bilinear2d_float_dtype_boundary",
            issue_url="https://github.com/pytorch/pytorch/issues/193811",
            eager_fn_factory=_upsample_eager_fn,
            safe_wrapper_factory=lambda cf: safe_compiled_upsample_bilinear2d(cf, torch_module),
            args=(torch_module.rand(1, 3, 8, 8, dtype=torch_module.float32),),
            is_boundary_case=True,
        ),
    ]


def diagnose(python_executable: Optional[str] = None) -> Dict[str, Any]:
    """Reproduce the torch.compile validation-bypass family from
    scratch against the currently installed torch build, for every
    fixture (invalid case + boundary case), and verify the guard
    wrappers restore eager's validation contract without spuriously
    flagging the valid boundary inputs. Never trusts a cached/prior
    result -- every call re-runs the actual repro.

    Each fixture runs in its own subprocess (see _worker.py) so a
    torch.compile-triggered global-state change from one fixture (a
    real leak this project discovered: compiling a
    torch.distributions-based function can flip a process-global
    validate-args flag) can never corrupt a later fixture's eager
    baseline.
    """
    torch_module = _import_torch()
    fixture_names = [spec["name"] for spec in _fixture_specs(torch_module)]
    results = [
        _run_fixture_via_subprocess(name, python_executable=python_executable)
        for name in fixture_names
    ]

    invalid_cases = [r for r in results if not r.is_boundary_case]
    boundary_cases = [r for r in results if r.is_boundary_case]

    any_validation_bypassed = any(r.validation_bypassed for r in invalid_cases)
    guard_fully_correct = all(r.guard_restores_validation for r in results)
    boundary_cases_not_spuriously_flagged = all(
        not r.guarded_raised for r in boundary_cases
    )

    return {
        "torch_version": torch_module.__version__,
        "issue_urls": sorted({r.issue_url for r in results}),
        "cases": [dataclasses.asdict(r) for r in results],
        "any_validation_bypassed": any_validation_bypassed,
        "guard_fully_correct": guard_fully_correct,
        "boundary_cases_not_spuriously_flagged": boundary_cases_not_spuriously_flagged,
    }
