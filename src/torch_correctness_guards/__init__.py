"""Consolidated PyTorch correctness diagnostics and guards."""

__version__ = "0.1.0"

from .guards.addcdiv_stale_scalar import (  # noqa: F401
    TorchUnavailableError,
    diagnose as diagnose_addcdiv_stale_scalar,
    make_safe_stale_scalar_step,
    safe_stale_scalar_step,
)
from .guards.as_strided_restride_oob import (  # noqa: F401
    diagnose as diagnose_as_strided_restride_oob,
    safe_as_strided,
)
from .guards.checkpoint_noise import (  # noqa: F401
    diagnose as diagnose_checkpoint_noise,
    safe_rrelu,
)
from .guards.dynamic_clamp import (  # noqa: F401
    diagnose as diagnose_dynamic_clamp,
    safe_clamp,
)

