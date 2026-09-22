# Hybrid virtual prefix implementation

This directory contains the implementation used by the current three-domain experiment.

User-facing training, serving, and evaluation commands are defined in the repository-level `experiment.sh`. Files under `internal/` are launcher implementation details.

The active model path consists of:

- `modeling_qwen3_5_hybrid_delta_residual_attention_prefix.py`: Hugging Face model implementation.
- `vllm_hybrid_delta_residual_attention_prefix_model.py`: vLLM runtime implementation.
- `prepare_hybrid_delta_residual_attention_prefix_model.py`: initial prepared model creation.
- `export_hybrid_delta_residual_attention_prefix_checkpoint.py`: training-checkpoint export.
- `compose_hybrid_prefixes.py`: prefix composition.
- `merge_hybrid_prefix_partitions.py` and `reshard_prefix_training_checkpoint.py`: checkpoint utilities.
- `verify_hybrid_delta_residual_attention_prefix_verl_fsdp2.py`, `verify_hf_prefix_reference.py`, and `verify_vllm_compiled_prefix_updates.py`: implementation verification.

Experiment artifacts are stored under `outputs/prefix_composition_3domain_20260917` by default.
