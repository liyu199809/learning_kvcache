"""Check anchored additive initialization against the BF16 Qwen3.5 backbone."""
import argparse
import json
from itertools import islice
from pathlib import Path
import random

import torch
from transformers import AutoModelForImageTextToText, AutoTokenizer
from transformers.models.qwen3_5 import modeling_qwen3_5 as qwen

from prepare import lines
from anchor_gate_metrics import AnchorGateMetrics
from sft_trajectory_tokens import marked_template, tokenize_trajectory
from train_envscaler_joint_prefix import Qwen3_5HybridDeltaResidualAttentionPrefixForConditionalGeneration
from train_prefix import MODEL
from prefix_tuning.virtual_prefix.joint_prefix_attention import anchored_prefix_attention, gdn_anchored_attention
from prefix_tuning.virtual_prefix.joint_prefix_attention import token_null_attention, gdn_token_null_attention


def check_anchor():
    query = torch.ones(1, 2, 1, 1)
    values = torch.ones(1, 1, 64, 1)
    bias = torch.zeros(2)
    for score, expected in ((1000.0, 1.0), (-1000.0, 0.0)):
        contribution, mass = anchored_prefix_attention(
            query, torch.full_like(values, score), values, bias, scale=1.0,
        )
        torch.testing.assert_close(mass, torch.full_like(mass, expected))
        torch.testing.assert_close(contribution, torch.full_like(contribution, expected))
    print("Anchor numerics passed at logits ±1000", flush=True)
    for score, expected in ((1000.0, 1.0), (-1000.0, 0.0)):
        contribution, mass = gdn_anchored_attention(
            query, values, values, torch.full_like(bias, score),
        )
        torch.testing.assert_close(mass, torch.full_like(mass, expected))
        torch.testing.assert_close(contribution, torch.full_like(contribution, expected))
    print("GDN anchor numerics passed at biases ±1000", flush=True)
    for null, expected in ((-1000., 1.), (1000., 0.)):
        projection = torch.full((1, 2, 1, 1), null)
        for output, mass, _ in (
            token_null_attention(query, values, values, bias, projection, scale=1.0),
            gdn_token_null_attention(query, values, values, bias, projection),
        ):
            torch.testing.assert_close(mass, torch.full_like(mass, expected))
            torch.testing.assert_close(output, torch.full_like(output, expected))
    print("Token-null numerics passed at projections ±1000", flush=True)


@torch.inference_mode()
def compare(initial, shard, shards, backend, data, metrics_path, calibrate, bias_path):
    if qwen.chunk_gated_delta_rule is None or not qwen.chunk_gated_delta_rule.__module__.startswith("fla."):
        raise RuntimeError("FLA GDN chunk kernel is unavailable")
    tokenizer = AutoTokenizer.from_pretrained(MODEL, local_files_only=True)
    template = marked_template(tokenizer.chat_template)
    rows = list(islice(lines(data), 400))
    samples = random.Random(20261124).sample(rows, 20)
    opts = dict(local_files_only=True, dtype=torch.bfloat16, attn_implementation=backend)
    anchored = Qwen3_5HybridDeltaResidualAttentionPrefixForConditionalGeneration.from_pretrained(
        initial, **opts,
    ).to("cuda").eval()
    assert anchored.config.text_config.residual_attention_prefix_mode in ("anchored_additive", "token_null_additive")
    assert all(layer.linear_attn._native_chunk_rule is qwen.chunk_gated_delta_rule
               for layer in anchored.model.language_model.layers if hasattr(layer, "linear_attn"))
    monitor = AnchorGateMetrics(anchored) if metrics_path else None
    if calibrate:
        from functools import partial
        from collections import defaultdict
        sums, counts = {}, defaultdict(int)
        def collect(name, raw_lse):
            sums[name] += raw_lse.sum(dim=(0, 2, 3), dtype=torch.float64)
            counts[name] += raw_lse.shape[0] * raw_lse.shape[2]
        for module_name, module in anchored.named_modules():
            if hasattr(module, "prefix_null_proj"):
                bias_name = "prefix_state_bias" if hasattr(module, "prefix_state_key") else "prefix_logit_bias"
                name = f"{module_name}.{bias_name}"
                sums[name] = torch.zeros(module.prefix_null_proj.shape[0], device="cuda", dtype=torch.float64)
                module.prefix_calibration = partial(collect, name)
        for row in samples[shard::shards]:
            tokens, _, _ = tokenize_trajectory(tokenizer, row, template)
            anchored.model(input_ids=torch.tensor([tokens], device="cuda"), use_cache=False)
            print(json.dumps({"id": row["id"], "calibration_tokens": len(tokens)}), flush=True)
        calibrate.write_text(json.dumps({name: {"sum": v.tolist(), "tokens": counts[name]} for name, v in sums.items()}))
        return
    if bias_path:
        biases = json.loads(bias_path.read_text())["heads"]
        for name, parameter in anchored.named_parameters():
            if name in biases:
                parameter.data = torch.tensor(biases[name]["c_h"], device="cuda", dtype=torch.float32)
    base = AutoModelForImageTextToText.from_pretrained(MODEL, **opts).to("cuda").eval()
    if monitor:
        monitor.enable()
    for row in samples[shard::shards]:
        tokens, _, _ = tokenize_trajectory(tokenizer, row, template)
        ids = torch.tensor([tokens], device="cuda")
        if monitor:
            monitor.length = len(tokens)
        base_hidden = base.model(input_ids=ids, use_cache=False).last_hidden_state
        prefix_hidden = anchored.model(input_ids=ids, use_cache=False).last_hidden_state
        errors = []
        for offset in range(0, len(tokens), 256):
            before = base.lm_head(base_hidden[:, offset:offset+256])
            after = anchored.lm_head(prefix_hidden[:, offset:offset+256])
            errors.append((before.float() - after.float()).abs().max())
        error = torch.stack(errors).max().item()
        print(json.dumps({"id": row["id"], "tokens": len(tokens), "max_logit_error": error,
                          "backend": backend, "gdn": qwen.chunk_gated_delta_rule.__module__}), flush=True)
        if error > 1e-3:
            raise RuntimeError(f"BF16 initial parity failed: {error:.6g} > 1e-3")
    if monitor:
        metrics_path.write_text(json.dumps({"values": monitor.values.tolist(), "layers": monitor.rows()}))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--initial", type=Path)
    parser.add_argument("--shard", type=int, default=0)
    parser.add_argument("--shards", type=int, default=1)
    parser.add_argument("--backend", choices=("sdpa", "flex_attention"), default="sdpa")
    parser.add_argument("--data", type=Path, default=Path("sft-decision6/t3-data/c-arm/train.jsonl"))
    parser.add_argument("--metrics", type=Path)
    parser.add_argument("--calibrate", type=Path)
    parser.add_argument("--bias", type=Path)
    args = parser.parse_args()
    check_anchor()
    if args.initial is not None:
        compare(args.initial, args.shard, args.shards, args.backend, args.data, args.metrics,
                args.calibrate, args.bias)
