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
from .guards.compile_validation import (  # noqa: F401
    diagnose as diagnose_compile_validation,
    safe_compiled_bernoulli,
    safe_compiled_binary_cross_entropy,
    safe_compiled_categorical_sample,
    safe_compiled_normal,
    safe_compiled_upsample_bilinear2d,
)
from .guards.cpu_backward_nan_tail import (  # noqa: F401
    diagnose as diagnose_cpu_backward_nan_tail,
    safe_celu_backward,
    safe_elu_backward,
    safe_hardswish_backward,
    safe_hardtanh_backward,
    safe_logit_backward,
    safe_shrink_backward,
)
from .guards.dynamo_closure_descriptor import (  # noqa: F401
    diagnose as diagnose_dynamo_closure_descriptor,
    diagnose_bound_method_guard,
    diagnose_module_shadow_guard,
    safe_bound_method_call,
    safe_call,
    safe_module_forward,
)
from .guards.dtype_view_scatter import (  # noqa: F401
    diagnose as diagnose_dtype_view_scatter,
    safe_compiled_dtype_view_diagonal_scatter,
)
from .guards.duplicate_index_writeorder import (  # noqa: F401
    diagnose as diagnose_duplicate_index_writeorder,
    make_safe_dup_index_assign,
    safe_dup_index_assign,
)
from .guards.dynamic_clamp import (  # noqa: F401
    diagnose as diagnose_dynamic_clamp,
    safe_clamp,
)
from .guards.equality_fusion import (  # noqa: F401
    diagnose as diagnose_equality_fusion,
    precision_safe_division_compare,
)
from .guards.embeddingbag_freq_scale import (  # noqa: F401
    diagnose as diagnose_embeddingbag_freq_scale,
    make_safe_embedding_bag,
    mps_is_functional,
    safe_embedding_bag,
)
from .guards.expand_fill import (  # noqa: F401
    diagnose as diagnose_expand_fill,
    safe_fill_,
)
from .guards.fp16_layernorm_tail import (  # noqa: F401
    diagnose as diagnose_fp16_layernorm_tail,
    safe_layer_norm,
)
from .guards.full_dtype import (  # noqa: F401
    diagnose as diagnose_full_dtype,
    safe_full,
)
from .guards.inplace_slice_shift_aliasing import (  # noqa: F401
    diagnose as diagnose_inplace_slice_shift_aliasing,
    safe_slice_shift,
)
from .guards.int64_index_truncation import (  # noqa: F401
    diagnose as diagnose_int64_index_truncation,
    guard_int64_op,
    make_safe_int64_arange_mul,
    safe_int64_arange_mul,
)
from .guards.linalg_nan import (  # noqa: F401
    diagnose as diagnose_linalg_nan,
    diagnose_norm_precision,
    safe_eigvalsh,
    safe_norm,
    safe_svdvals,
    safe_vector_norm,
)
from .guards.linalg_pinv_complex_grad import (  # noqa: F401
    diagnose as diagnose_linalg_pinv_complex_grad,
    safe_complex_pinv_grad,
)
from .guards.scatter_copyback_alias import (  # noqa: F401
    diagnose as diagnose_scatter_copyback_alias,
    safe_compiled_scatter_returning,
)
from .guards.softmax_dim import (  # noqa: F401
    diagnose as diagnose_softmax_dim,
    safe_softmax_attention,
)
from .guards.normal_dtype_promotion import (  # noqa: F401
    diagnose as diagnose_normal_dtype_promotion,
    safe_compiled_normal_sample,
)
from .guards.shuffle_sample_frozen import (  # noqa: F401
    diagnose as diagnose_shuffle_sample_frozen,
    make_safe_sample,
    make_safe_shuffle,
    safe_sample,
    safe_shuffle,
)
from .guards.std_precision import (  # noqa: F401
    diagnose as diagnose_std_precision,
    safe_std,
    safe_std_mean,
    safe_var,
    safe_var_mean,
)
from .guards.tiled_reduction_tail_store import (  # noqa: F401
    TailStoreOverrunSuspected,
    diagnose as diagnose_tiled_reduction_tail_store,
    safe_compiled_reduction,
)
from .guards.transpose_argmin import (  # noqa: F401
    diagnose as diagnose_transpose_argmin,
    safe_reduce_index,
)

