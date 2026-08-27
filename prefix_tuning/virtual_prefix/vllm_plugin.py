"""Process-safe vLLM general-plugin registration."""

from vllm import ModelRegistry


DELTA_VIRTUAL_PREFIX_ARCHITECTURE = (
    "Qwen3_5DeltaVirtualPrefixForConditionalGeneration"
)
DELTA_VIRTUAL_PREFIX_MODEL_CLASS = (
    "prefix_tuning.virtual_prefix.vllm_model:"
    "Qwen3_5DeltaVirtualPrefixForConditionalGeneration"
)
INDEPENDENT_DELTA_KV_PREFIX_ARCHITECTURE = (
    "Qwen3_5IndependentDeltaKVPrefixForConditionalGeneration"
)
INDEPENDENT_DELTA_KV_PREFIX_MODEL_CLASS = (
    "prefix_tuning.virtual_prefix.vllm_independent_delta_kv_prefix_model:"
    "Qwen3_5IndependentDeltaKVPrefixForConditionalGeneration"
)
HYBRID_DELTA_RESIDUAL_ATTENTION_PREFIX_ARCHITECTURE = (
    "Qwen3_5HybridDeltaResidualAttentionPrefixForConditionalGeneration"
)
HYBRID_DELTA_RESIDUAL_ATTENTION_PREFIX_MODEL_CLASS = (
    "prefix_tuning.virtual_prefix."
    "vllm_hybrid_delta_residual_attention_prefix_model:"
    "Qwen3_5HybridDeltaResidualAttentionPrefixForConditionalGeneration"
)
# Preserve the public names used by the original single-model plugin.
ARCHITECTURE = DELTA_VIRTUAL_PREFIX_ARCHITECTURE
MODEL_CLASS = DELTA_VIRTUAL_PREFIX_MODEL_CLASS


def register() -> None:
    ModelRegistry.register_model(
        DELTA_VIRTUAL_PREFIX_ARCHITECTURE,
        DELTA_VIRTUAL_PREFIX_MODEL_CLASS,
    )
    ModelRegistry.register_model(
        INDEPENDENT_DELTA_KV_PREFIX_ARCHITECTURE,
        INDEPENDENT_DELTA_KV_PREFIX_MODEL_CLASS,
    )
    ModelRegistry.register_model(
        HYBRID_DELTA_RESIDUAL_ATTENTION_PREFIX_ARCHITECTURE,
        HYBRID_DELTA_RESIDUAL_ATTENTION_PREFIX_MODEL_CLASS,
    )
