"""Consolidated PyTorch correctness diagnostics and guards."""

__version__ = "0.1.0"

from .guards.addcdiv_stale_scalar import (  # noqa: F401
    TorchUnavailableError,
    diagnose as diagnose_addcdiv_stale_scalar,
    make_safe_stale_scalar_step,
    safe_stale_scalar_step,
)
