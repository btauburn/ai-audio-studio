try:
    from .accel_engine import AccelInferenceEngine  # noqa: F401
    from .attention import (  # noqa: F401
        Attention,
        get_forward_context,
        reset_forward_context,
        set_forward_context,
    )
    from .gpt2_accel import GPT2AccelAttention, GPT2AccelModel  # noqa: F401
    from .kv_manager import KVCacheManager, Seq  # noqa: F401
except ModuleNotFoundError as exc:
    if getattr(exc, "name", None) != "triton":
        raise

    _triton_exc = exc

    def _missing(*args, **kwargs):
        raise ImportError(
            "Optional dependency 'triton' is required for indextts.accel. "
            "Install triton (and compatible torch/cuda) or avoid using the accel backend."
        ) from _triton_exc

    class _MissingTriton:
        def __init__(self, *args, **kwargs):
            _missing()

    AccelInferenceEngine = _MissingTriton
    Attention = _MissingTriton
    GPT2AccelAttention = _MissingTriton
    GPT2AccelModel = _MissingTriton
    KVCacheManager = _MissingTriton
    Seq = _MissingTriton

    get_forward_context = _missing
    reset_forward_context = _missing
    set_forward_context = _missing
