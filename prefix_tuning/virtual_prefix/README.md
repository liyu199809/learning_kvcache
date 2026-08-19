# Qwen3.5 per-layer DeltaNet virtual prefix

## Definition

For each linear-attention layer `l`, the trainable object is a table of M
continuous tokens at the input of that layer's GDN mixer:

```text
P_l: [M, hidden_size]
[p_l,0, ..., p_l,M-1, x_l,0, ..., x_l,T-1]
```

Here `X_l` is the real hidden sequence after the decoder layer's
`input_layernorm`. Qwen3.5-4B has 24 linear-attention layers and hidden size
2560. Two prepared variants are kept:

```text
M=256:  24 * 256  * 2560 =  15,728,640 trainable parameters
M=2048: 24 * 2048 * 2560 = 125,829,120 trainable parameters
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
project:     /mnt/storage/disk3/self_evolver
base:        /mnt/storage/disk1/verl_data/base_model/Qwen3.5-4B
model M=256: /mnt/storage/disk1/verl_data/base_model/Qwen3.5-4B-DeltaVirtualPrefix-256
model M=2048: /mnt/storage/disk1/verl_data/base_model/Qwen3.5-4B-DeltaVirtualPrefix-2048
```

The checkpoint keeps `model_type: qwen3_5`. Its only new weights are:

```text
model.language_model.layers.<linear-layer>.linear_attn.prefix_tokens
```

The base safetensors are hardlinked. The BF16 virtual-token shard is 30 MiB for
M=256 and 240 MiB for M=2048. The virtual tokens are zero-initialized, so a
freshly prepared model exactly matches the base model in the tested
Transformers forward.

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

The verl OPSD launcher for M=2048 uses six actor/rollout GPUs and two teacher
GPUs. The actor loads the prepared prefix model while the teacher loads the
original Qwen3.5-4B:

```bash
cd /mnt/storage/disk3/self_evolver
bash verl/examples/on_policy_distillation_trainer/run_qwen3_5_4b_awm_opsd_delta_prefix_m2048.sh
```

verl freezes parameters with a regex full match before FSDP2 wrapping, checks
24 tensors and 125,829,120 elements, builds the optimizer from only those
parameters, and sends only those tensors to native vLLM on rollout updates.
The launcher enables padding-free packed batches with sequence parallel size 1,
turns torch compile off, enables gradient checkpointing, and uses learning rate
`5e-6` with zero weight decay. For every packed segment, the Transformers path
reuses the differentiable recurrent state produced by that layer's prefix and
seeds the causal convolution with only its final three projected values. The
2048 virtual tokens do not enter `input_ids`, positions, or vLLM's token budget.

The older ms-swift M=256 launcher remains available:

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
zero. The legacy ms-swift launcher still uses ordinary padded data parallel and
DeepSpeed ZeRO-2; its packed/sequence-parallel path remains disabled. The verl
launcher above owns the supported packed implementation.

## Rebuild

The prepared checkpoint already exists. To build another prefix length or
output path:

```bash
cd /mnt/storage/disk3/self_evolver
.venv/bin/python prefix_tuning/virtual_prefix/prepare_delta_virtual_prefix_model.py \
  --source /mnt/storage/disk1/verl_data/base_model/Qwen3.5-4B \
  --output /mnt/storage/disk1/verl_data/base_model/Qwen3.5-4B-DeltaVirtualPrefix-new \
  --modeling-file prefix_tuning/virtual_prefix/modeling_qwen3_5_delta_virtual_prefix.py \
  --num-virtual-tokens 2048
```

The builder refuses to overwrite an existing output directory.

To export a verl trainable-only FSDP checkpoint as a deployable HF directory:

```bash
.venv/bin/python prefix_tuning/virtual_prefix/export_delta_virtual_prefix_checkpoint.py \
  --checkpoint checkpoints/<project>/<run>/global_step_<N>/actor \
  --prepared-model /mnt/storage/disk1/verl_data/base_model/Qwen3.5-4B-DeltaVirtualPrefix-2048 \
  --output /mnt/storage/disk1/verl_data/base_model/<export-name>
```

The exporter gathers only the prefix shards into
`delta_virtual_prefix.safetensors`. When output and prepared model are on the
same filesystem, the 4B base safetensors remain hardlinks.

## Files

- `modeling_qwen3_5_delta_virtual_prefix.py`: Transformers virtual-token path.
- `vllm_model.py`: native vLLM prefix-state construction and cache injection.
- `vllm_plugin.py`: process-safe vLLM ModelRegistry entry point.
- `plugin.py`: ms-swift model registration and prefix-only callback.
- `prepare_delta_virtual_prefix_model.py`: checkpoint builder.
- `export_delta_virtual_prefix_checkpoint.py`: verl checkpoint exporter.
- `run_opsd_delta_virtual_prefix.sh`: OPSD/GRPO launcher.
- `verify_verl_fsdp2.py`: real two-GPU FSDP2 train/sync/checkpoint smoke.
- `verify_vllm_smoke.py`: eager or compiled native-vLLM smoke test.

## Validation performed

- zero virtual prefix vs base full-model logits: max difference 0;
- nonzero virtual prefix changes full-model logits;
- left and right padding vs unpadded GDN: max difference 0;
- full GDN forward vs prefill plus decode: BF16 max difference below 0.002;
- all 24 M=2048 prefix tensors receive finite, nonzero gradients with gradient
  checkpointing;
- two-GPU FSDP2 selects exactly 125,829,120 optimizer parameters, changes the
  prefix only, synchronizes 24 tensors, and resumes model/optimizer/scheduler/RNG;
- prefix-only checkpoint export produces 24 BF16 `[2048, 2560]` tensors and a
  Transformers-loadable HF directory with hardlinked base shards;
- ms-swift loader and freeze callback load exactly 24 `[256, 2560]` tensors;
- native vLLM eager prefill/decode passes for zero and nonzero checkpoints;
- native vLLM torch.compile and CUDA-graph prefill/decode passes;
- a six-actor/two-teacher verl OPSD step completes rollout, top-k teacher
  distillation, actor update, checkpoint, and the next rollout weight sync.
