# Qwen3.5 per-layer DeltaNet virtual prefix

## Definition

For each linear-attention layer `l`, the trainable object is a table of M
continuous tokens at the input of that layer's GDN mixer:

```text
P_l: [M, hidden_size]
[p_l,0, ..., p_l,M-1, x_l,0, ..., x_l,T-1]
```

Here `X_l` is the real hidden sequence after the decoder layer's
`input_layernorm`. Qwen3.5-4B uses:

```text
linear-attention layers: 24
M:                       256
hidden_size:             2560
tensor per layer:        [256, 2560]
trainable parameters:    24 * 256 * 2560 = 15,728,640
```

The virtual-token outputs are discarded. Their causal effect remains in the
depthwise-convolution and DeltaNet recurrent states seen by the real tokens.
Every layer owns a different `P_l`. The eight full-attention layers are not
modified by this implementation.

Transformers executes the logical sequence as two consecutive causal chunks:
first `P_l`, then `X_l`. This is equivalent to concatenation, keeps the user
chunk's FLA tiling stable, and leaves gradients from the user loss to `P_l`.
For a padded batch, valid tokens are compacted so padding never occurs between
the virtual prefix and the user sequence.

vLLM does not expose virtual positions to its scheduler. For each new request,
the native subclass executes the same `P_l` with the layer's frozen projection,
convolution, gating, and Delta rule, writes the resulting convolution/recurrent
states into the native cache, and then calls the stock Qwen3.5 GDN core for the
user tokens. Decode reuses that cache and never applies `P_l` again.

## Server paths

```text
project: /mnt/storage/disk3/self_evolver
base:    /mnt/storage/disk1/verl_data/base_model/Qwen3.5-4B
model:   /mnt/storage/disk1/verl_data/base_model/Qwen3.5-4B-DeltaVirtualPrefix-256
```

The checkpoint keeps `model_type: qwen3_5`. Its only new weights are:

```text
model.language_model.layers.<linear-layer>.linear_attn.prefix_tokens
```

The base safetensors are hardlinked; the new BF16 virtual-token shard is 30
MiB. The virtual tokens are zero-initialized, so the initial model exactly
matches the base model in the tested Transformers forward.

## vLLM plugin

vLLM engine and worker processes use `spawn`, so the ModelRegistry entry must
be installed in the environment rather than registered only in the launcher:

```bash
cd /mnt/storage/disk3/self_evolver
/root/.local/bin/uv pip install --python .venv/bin/python -e prefix_tuning
```

Then standard native vLLM loading works:

```bash
CUDA_VISIBLE_DEVICES=7 .venv/bin/vllm serve \
  /mnt/storage/disk1/verl_data/base_model/Qwen3.5-4B-DeltaVirtualPrefix-256 \
  --model-impl vllm \
  --trust-remote-code \
  --max-model-len 32768
```

Do not request `--model-impl transformers` on vLLM 0.18: upstream Qwen3.5's
hybrid GDN cache is not compatible with that generic backend. The registered
class extends vLLM's native Qwen3.5 implementation and supports its normal
torch.compile/CUDA-graph path.

## Training

```bash
cd /mnt/storage/disk3/self_evolver
bash prefix_tuning/virtual_prefix/run_opsd_delta_virtual_prefix.sh
```

Useful overrides:

```bash
PREFIX_LR=5e-4 \
TRAIN_GPUS=0,1,2,3,4,5,6,7 \
OUTPUT_DIR=/mnt/storage/disk3/self_evolver/output/my_virtual_prefix \
bash prefix_tuning/virtual_prefix/run_opsd_delta_virtual_prefix.sh
```

The `delta_virtual_prefix_only` callback freezes everything except the 24
`prefix_tokens` tensors and checks their count and shape. Keep weight decay at
zero. Transformers packed/padding-free and sequence-parallel GDN training are
intentionally disabled because their `cu_seqlens` would need to include a
different virtual prefix at every linear layer. Ordinary padded data parallel
and DeepSpeed ZeRO-2 are supported by the launcher.

## Rebuild

The prepared checkpoint already exists. To build another prefix length or
output path:

```bash
cd /mnt/storage/disk3/self_evolver
.venv/bin/python prefix_tuning/virtual_prefix/prepare_delta_virtual_prefix_model.py \
  --source /mnt/storage/disk1/verl_data/base_model/Qwen3.5-4B \
  --output /mnt/storage/disk1/verl_data/base_model/Qwen3.5-4B-DeltaVirtualPrefix-new \
  --modeling-file prefix_tuning/virtual_prefix/modeling_qwen3_5_delta_virtual_prefix.py \
  --num-virtual-tokens 256
```

The builder refuses to overwrite an existing output directory.

## Files

- `modeling_qwen3_5_delta_virtual_prefix.py`: Transformers virtual-token path.
- `vllm_model.py`: native vLLM prefix-state construction and cache injection.
- `vllm_plugin.py`: process-safe vLLM ModelRegistry entry point.
- `plugin.py`: ms-swift model registration and prefix-only callback.
- `prepare_delta_virtual_prefix_model.py`: checkpoint builder.
- `run_opsd_delta_virtual_prefix.sh`: OPSD/GRPO launcher.
- `verify_vllm_smoke.py`: eager or compiled native-vLLM smoke test.

## Validation performed

- zero virtual prefix vs base full-model logits: max difference 0;
- nonzero virtual prefix changes full-model logits;
- left and right padding vs unpadded GDN: max difference 0;
- full GDN forward vs prefill plus decode: BF16 max difference below 0.002;
- all 24 full-checkpoint prefix tensors receive nonzero gradients;
- ms-swift loader and freeze callback load exactly 24 `[256, 2560]` tensors;
- native vLLM eager prefill/decode passes for zero and nonzero checkpoints;
- native vLLM torch.compile and CUDA-graph prefill/decode passes.
