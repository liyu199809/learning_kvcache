# Qwen3.5 DeltaNet virtual-prefix tuning

The active implementation is in `virtual_prefix/`. It adds 256 independent
continuous virtual tokens to each of Qwen3.5-4B's 24 `linear_attention`
mixers. Rejected direct-state and soft-token prototypes have been removed.

See `virtual_prefix/README.md` for architecture, training, vLLM registration,
checkpoint generation, constraints, and validation details.
