"""Process-safe vLLM general-plugin registration."""

from vllm import ModelRegistry


ARCHITECTURE = "Qwen3_5DeltaVirtualPrefixForConditionalGeneration"
MODEL_CLASS = (
    "prefix_tuning.virtual_prefix.vllm_model:"
    "Qwen3_5DeltaVirtualPrefixForConditionalGeneration"
)


def register() -> None:
    ModelRegistry.register_model(ARCHITECTURE, MODEL_CLASS)
