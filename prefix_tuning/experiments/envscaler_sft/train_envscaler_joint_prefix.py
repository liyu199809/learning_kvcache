"""Fit the Qwen3.5 joint-softmax prefixes on masked EnvScaler trajectories."""
import argparse
from datetime import timedelta
import json
import math
import os
from pathlib import Path
import random
import statistics
import subprocess
import time

import torch
import torch._dynamo
import torch.distributed as dist
import torch.nn.functional as F
from safetensors.torch import save_file
from transformers import AutoTokenizer
from transformers.models.qwen3_5 import modeling_qwen3_5 as qwen

from prefix_tuning.virtual_prefix.modeling_qwen3_5_hybrid_delta_residual_attention_prefix import (
    Qwen3_5HybridDeltaResidualAttentionPrefixForConditionalGeneration,
)
from train_prefix import write_json
from train_prefix_distributed import append_metric, rank_batches, render_curve
from anchor_gate_metrics import AnchorGateMetrics


def trajectory_loss(model, row, pad_token):
    ids, target_positions = row[1], row[3]
    positions = torch.tensor(target_positions, device=model.device)
    inputs = torch.tensor([ids], device=model.device)
    labels = inputs[0, positions]
    # Right padding leaves the causal prefix and supervised positions unchanged.
    inputs = F.pad(inputs, (0, (-inputs.shape[-1]) % 1024), value=pad_token)
    logits = model(input_ids=inputs, logits_to_keep=positions - 1, use_cache=False).logits[0]
    return F.cross_entropy(logits.float(), labels, reduction="sum")


def save_prefixes(output, name, parameters):
    path = output / name
    temporary = path.with_suffix(".tmp")
    save_file({key: p.detach().cpu() for key, p in parameters.items()}, temporary,
              metadata={"format": "qwen3_5_joint_softmax", "gdn_query": "post_conv"})
    temporary.replace(path)


@torch.no_grad()
def validate(model, rows, rank, world, pad_token):
    model.eval()
    metrics = torch.zeros(2, dtype=torch.float64, device=model.device)
    for row in rows[rank::world]:
        metrics[0] += trajectory_loss(model, row, pad_token)
        metrics[1] += len(row[2])
    dist.all_reduce(metrics)
    model.train()
    return (metrics[0] / metrics[1]).item()


@torch.no_grad()
def compare_training_batch(model, rows, local_batch, pad_token, monitor):
    model.eval()
    sums = torch.zeros(3, dtype=torch.float64, device=model.device)
    monitor.values.zero_()
    for index in local_batch:
        row = rows[index]
        monitor.length = len(row[1])
        monitor.enable()
        sums[0] += trajectory_loss(model, row, pad_token)
        monitor.disable()
        for module in monitor.modules:
            module.prefix_disabled = True
        try:
            sums[1] += trajectory_loss(model, row, pad_token)
        finally:
            for module in monitor.modules:
                module.prefix_disabled = False
        sums[2] += len(row[2])
    dist.all_reduce(sums)
    dist.all_reduce(monitor.values)
    training, base, tokens = sums.tolist()
    layers = monitor.rows()
    ratios = {}
    for kind in ("gdn", "full_attention"):
        thirds = [third for layer in layers if layer["type"] == kind for third in layer["thirds"]
                  if third["count"]]
        ratios[kind] = sum(t["output_ratio"] * t["count"] for t in thirds) / sum(t["count"] for t in thirds)
    monitor.values.zero_()
    model.train()
    return {"training_nll": training / tokens, "base_nll": base / tokens, "target_tokens": int(tokens),
            "mean_output_ratio": ratios, "layers": layers}


def render_gate_curve(output):
    subprocess.run(["/usr/local/bin/python", str(Path(__file__).with_name("anchor_gate_metrics.py")),
                    str(output)], check=True,
                   env={**os.environ, "MPLCONFIGDIR": str(output / ".matplotlib")})


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run", type=Path, required=True)
    parser.add_argument("--attention-mode", choices=("joint", "anchored_additive", "token_null_additive"), default="joint")
    parser.add_argument("--gdn-mode", choices=("state_rows_joint_softmax", "anchored_additive", "token_null_additive"),
                        default="state_rows_joint_softmax")
    args = parser.parse_args()
    run = args.run.resolve()
    world, rank = int(os.environ["WORLD_SIZE"]), int(os.environ["RANK"])
    device = torch.device("cuda", int(os.environ["LOCAL_RANK"]))
    torch.cuda.set_device(device)
    dist.init_process_group("nccl", device_id=device, timeout=timedelta(minutes=30))
    torch.manual_seed(20261124)
    # FlexAttention specializes on each of the sixteen padded sequence lengths.
    torch._dynamo.config.recompile_limit = 64

    initial = run / "initial-model"
    output = run / "train"
    model_config = json.loads((initial / "config.json").read_text())
    text_config = model_config["text_config"]
    assert text_config["residual_attention_prefix_joint_softmax"]
    if text_config.get("residual_attention_prefix_mode", "joint") != args.attention_mode:
        raise ValueError(f"Checkpoint does not implement requested attention mode {args.attention_mode}")
    chunk_rule = qwen.chunk_gated_delta_rule
    if chunk_rule is None or not chunk_rule.__module__.startswith("fla."):
        raise RuntimeError("FLA GDN chunk kernel unavailable")
    assert model_config["independent_delta_kv_prefix"]["method"] == args.gdn_mode
    anchored_gdn = args.gdn_mode != "state_rows_joint_softmax"
    token_null = args.gdn_mode == "token_null_additive"
    if anchored_gdn:
        assert text_config["independent_delta_prefix_method"] == args.gdn_mode
        assert args.attention_mode == args.gdn_mode
    assert text_config["layer_types"].count("linear_attention") == 24
    assert text_config["layer_types"].count("full_attention") == 8
    model, loading = Qwen3_5HybridDeltaResidualAttentionPrefixForConditionalGeneration.from_pretrained(
        initial, local_files_only=True, dtype=torch.bfloat16,
        attn_implementation="sdpa", output_loading_info=True,
    )
    assert not loading["missing_keys"] and not loading["mismatched_keys"], loading
    model = model.to(device)
    model.requires_grad_(False)
    model.config.use_cache = False
    # The legacy joint attention uses FlexAttention. Anchored additive uses
    # native causal SDPA and computes only the 64 prefix logits separately.
    if args.attention_mode == "joint":
        model.config.text_config._attn_implementation = "flash_attention_2"
    model.gradient_checkpointing_enable(gradient_checkpointing_kwargs={"use_reentrant": False})
    qwen_layers = model.model.language_model.layers
    for layer in qwen_layers:
        if hasattr(layer, "linear_attn"):
            assert layer.linear_attn._native_chunk_rule is chunk_rule
            assert layer.linear_attn.anchored_prefix == anchored_gdn
            layer.linear_attn._use_compiled_prefix = True

    parameters = {}
    for name, parameter in model.named_parameters():
        suffix = name.rsplit(".", 1)[-1]
        if suffix.startswith("prefix_") and not (anchored_gdn and suffix == "prefix_state_temperature"):
            parameter.data = parameter.data.float()
            parameters[name] = parameter.requires_grad_(True)
    expected_count = 128 if token_null else 96 if anchored_gdn else 120
    assert len(parameters) == model_config["hybrid_prefix"]["trainable_tensor_count"] == expected_count
    trainable = list(parameters.values())
    assert sum(p.numel() for p in trainable) == model_config["hybrid_prefix"]["trainable_numel"]
    assert all(not p.detach().count_nonzero().item() for n, p in parameters.items()
               if n.endswith(("prefix_state_value", "prefix_value_tokens", "prefix_null_proj")))
    biases = [name for name in parameters if name.endswith(("prefix_logit_bias", "prefix_state_bias"))]
    if token_null:
        calibration = json.loads((run / "gate-calibration.json").read_text())
        assert calibration["samples"] == 20 and set(biases) == set(calibration["heads"])
        with torch.no_grad():
            for name in biases:
                p = parameters[name]
                p.copy_(torch.tensor(calibration["heads"][name]["c_h"], dtype=p.dtype, device=p.device))
        groups = [{"params": [p for name, p in parameters.items() if name not in biases], "lr": 1e-5},
                  {"params": [parameters[name] for name in biases], "lr": 1e-3}]
    else:
        groups = [{"params": trainable, "lr": 1e-5}]
    assert all((p.detach() < 0).all().item() for n, p in parameters.items()
               if n.endswith("prefix_logit_bias"))
    if rank == 0:
        print(f"Loaded {type(model).__name__}, {len(parameters)} trainable tensors, "
              f"{sum(p.numel() for p in trainable):,} trainable parameters, "
              f"attention={args.attention_mode}, GDN={args.gdn_mode} via {chunk_rule.__module__}", flush=True)

    while not (run / "encoded.pt").exists():
        time.sleep(2)
    # This cache is produced locally from the masked JSONL by the preparation script.
    # Native pickle decoding avoids the overhead of interpreting hundreds of millions of integers in Python.
    data = torch.load(run / "encoded.pt", map_location="cpu", weights_only=False)
    rows = data["train"]
    assert len(rows) == 33523
    pad_token = AutoTokenizer.from_pretrained(initial, local_files_only=True).pad_token_id
    assert pad_token is not None
    optimizer = torch.optim.AdamW(groups, weight_decay=0, fused=True)
    group_rates = [group["lr"] for group in optimizer.param_groups]
    batch_size = 8
    steps = math.ceil(len(rows) / batch_size)
    warmup_steps = math.ceil(steps * .1)
    if rank == 0:
        write_json(output / "config.json", {
            "model": str(initial), "architecture": type(model).__name__,
            "gdn_query": "post_conv", "gdn_mode": args.gdn_mode,
            "gdn_chunk_kernel": chunk_rule.__module__,
            "full_attention_mode": args.attention_mode,
            "gdn_prefix_tokens": text_config["independent_delta_prefix_num_virtual_tokens"],
            "full_attention_prefix_tokens": text_config["residual_attention_prefix_num_virtual_tokens"],
            "trainable": {name: list(p.shape) for name, p in parameters.items()},
            "world_size": world, "train_rows": len(rows), "epochs": 1, "batch_size": batch_size,
            "steps": steps, "learning_rate": 1e-5, "warmup_steps": warmup_steps,
            "optimizer_group_learning_rates": group_rates,
            "coverage": json.loads((run / "coverage.json").read_text()),
            "context": 16384, "seed": 20261124, "max_grad_norm": 1.0,
        })

    order = list(range(len(rows)))
    random.Random(20261125).shuffle(order)
    model.train()
    speed_times = []
    if anchored_gdn:
        warmup = order[:batch_size]
        for index in rank_batches(warmup, rows, world)[rank]:
            trajectory_loss(model, rows[index], pad_token).backward()
        optimizer.zero_grad(set_to_none=True)
        dist.barrier(device_ids=[device.index])
        torch.cuda.synchronize(device)
        if rank == 0:
            print("CUDA kernel warmup complete. Measuring 20 optimizer steps.", flush=True)
    gate_metrics = AnchorGateMetrics(model) if anchored_gdn else None
    started = time.monotonic()
    for step, start in enumerate(range(0, len(order), batch_size), 1):
        step_started = time.monotonic()
        batch = order[start:start + batch_size]
        optimizer.zero_grad(set_to_none=True)
        metrics = torch.zeros(2, device=device)
        collect = gate_metrics is not None and step % 100 == 0
        for index in rank_batches(batch, rows, world)[rank]:
            row = rows[index]
            if collect:
                gate_metrics.length = len(row[1])
                gate_metrics.enable()
            loss = trajectory_loss(model, row, pad_token)
            if collect:
                gate_metrics.disable()
            ((row[4] if len(row) > 4 else 1.0) * loss / len(row[2]) / len(batch)).backward()
            metrics[0] += loss.detach()
            metrics[1] += len(row[2])
        gradients = torch.cat([(p.grad if p.grad is not None else torch.zeros_like(p)).reshape(-1)
                               for p in trainable])
        dist.all_reduce(gradients)
        dist.all_reduce(metrics)
        offset = 0
        for p in trainable:
            p.grad = gradients[offset:offset + p.numel()].view_as(p)
            offset += p.numel()
        norm = torch.nn.utils.clip_grad_norm_(trainable, 1.0, error_if_nonfinite=True)
        fraction = min(step / warmup_steps, 1.0)
        lr = group_rates[0] * fraction
        for group, rate in zip(optimizer.param_groups, group_rates, strict=True):
            group["lr"] = rate * fraction
        optimizer.step()
        if anchored_gdn and step <= 20:
            torch.cuda.synchronize(device)
            elapsed = torch.tensor(time.monotonic() - step_started, device=device)
            dist.all_reduce(elapsed, op=dist.ReduceOp.MAX)
            speed_times.append(elapsed.item())
            if step == 20:
                passed = max(speed_times) <= 10
                if rank == 0:
                    write_json(output / "speed-gate.json", {
                        "seconds_per_step": speed_times, "mean_seconds": statistics.mean(speed_times),
                        "max_seconds": max(speed_times), "passed": passed,
                    })
                    print(f"20-step speed gate: max {max(speed_times):.2f}s, passed={passed}", flush=True)
                if not passed:
                    dist.destroy_process_group()
                    return
        if step == 1:
            for suffix in ("prefix_state_value", "prefix_value_tokens"):
                group = [p for name, p in parameters.items() if name.endswith(suffix)]
                assert all(p.grad.count_nonzero().item() for p in group), suffix
                assert all(p.detach().count_nonzero().item() for p in group), suffix
            assert all(p.grad is None for p in model.parameters() if not p.requires_grad)
        if rank == 0:
            metric = {"split": "train", "epoch": 1, "step": step, "rows": len(batch),
                      "target_tokens": int(metrics[1].item()), "nll": (metrics[0] / metrics[1]).item(),
                      "grad_norm": norm.item(), "learning_rate": lr,
                      "elapsed_seconds": round(time.monotonic() - started, 2)}
            if token_null:
                metric["bias_learning_rate"] = optimizer.param_groups[1]["lr"]
            append_metric(output, metric)
            if step == 1 or step % 10 == 0:
                print(json.dumps(metric), flush=True)
            if step == 1 or step % 100 == 0:
                save_prefixes(output, "prefix-in-progress.safetensors", parameters)
                render_curve(output)
        if collect:
            dist.all_reduce(gate_metrics.values)
            if rank == 0:
                with (output / "gate-metrics.jsonl").open("a") as stream:
                    stream.write(json.dumps({"step": step, "layers": gate_metrics.rows()}, allow_nan=False) + "\n")
            gate_metrics.values.zero_()
        if token_null and step in (200, 500):
            comparison = compare_training_batch(model, rows, rank_batches(batch, rows, world)[rank],
                                                pad_token, gate_metrics)
            comparison["step"] = step
            comparison["batch_indices"] = batch
            stop = step == 500 and (max(comparison["mean_output_ratio"].values()) < 1e-3
                                    or comparison["training_nll"] >= comparison["base_nll"])
            comparison["early_stop"] = stop
            if rank == 0:
                write_json(output / f"step-{step:04d}-comparison.json", comparison)
                print(json.dumps({k: v for k, v in comparison.items() if k != "layers"}), flush=True)
            if stop:
                if rank == 0:
                    write_json(output / "result.json", {"optimizer_steps": step, "early_stop": True,
                                                       "comparison": f"step-{step:04d}-comparison.json"})
                    render_curve(output)
                    render_gate_curve(output)
                dist.destroy_process_group()
                return
        if step in {math.ceil(len(rows) * i / (batch_size * 4)) for i in range(1, 5)}:
            score = validate(model, data["validation"], rank, world, pad_token)
            if rank == 0:
                append_metric(output, {"split": "validation", "epoch": 1, "step": step, "nll": score,
                                       "by_source": {"envscaler": {"nll": score}}})
                save_prefixes(output, f"step-{step:04d}-joint-prefix.safetensors", parameters)
                render_curve(output)
    if rank == 0:
        save_prefixes(output, "epoch-01-joint-prefix.safetensors", parameters)
        write_json(output / "result.json", {"optimizer_steps": steps, "validation_nll": score,
                                           "wall_seconds": round(time.monotonic() - started, 1)})
        if gate_metrics:
            render_gate_curve(output)
    dist.destroy_process_group()


if __name__ == "__main__":
    main()
