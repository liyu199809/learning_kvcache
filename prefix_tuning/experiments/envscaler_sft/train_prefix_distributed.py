"""Data-parallel SFT of a shared soft prefix with a frozen Qwen backbone."""
import argparse
from collections import defaultdict
from datetime import timedelta
import json
import math
import os
from pathlib import Path
import random
import subprocess
import time

import torch
import torch.distributed as dist
import torch.nn.functional as F
from safetensors.torch import save_file
from transformers import AutoModelForImageTextToText, AutoTokenizer

from train_prefix import COACH, MODEL, OUT, loss_sum, tokenize, write_json
from sft_trajectory_tokens import tokenize_split

SEED = 20261123
PREFIX_LENGTH = 32
SOURCES = ("approved_257", "q35_v1", "agent_world", "taco", "awm", "envscaler")
METADATA = {"base_model": MODEL, "format": "input_embedding_prefix", "objective": "next_expert_turn"}


@torch.no_grad()
def evaluate(model, rows, prefix, rank, world):
    local = defaultdict(lambda: [0.0, 0, 0, 0.0])
    for row in rows[rank::world]:
        values = local[row[0]]
        loss = loss_sum(model, row, prefix).item()
        values[0] += loss
        values[1] += len(row[2])
        values[2] += 1
        values[3] += loss / len(row[2])
    totals = torch.tensor([local[source] for source in SOURCES], dtype=torch.float64, device=model.device)
    dist.all_reduce(totals)
    scores = {}
    for source, (loss, tokens, count, sample_total) in zip((*SOURCES, "combined"), [*totals.tolist(), totals.sum(0).tolist()]):
        if tokens:
            scores[source] = {"nll": loss / tokens, "sample_nll": sample_total / count,
                              "target_tokens": int(tokens), "rows": int(count)}
    return scores


def append_metric(root, row):
    with (root / "loss.jsonl").open("a") as stream:
        stream.write(json.dumps(row, allow_nan=False) + "\n")


def save_prefix(root, name, prefix, objective):
    path = root / name
    temporary = path.with_suffix(".tmp")
    save_file({"soft_prefix": prefix.detach().cpu()}, temporary, metadata={**METADATA, "objective": objective})
    temporary.replace(path)


def render_curve(root):
    env = {**os.environ, "MPLCONFIGDIR": str(root / ".matplotlib")}
    env.pop("PYTHONPATH", None)
    subprocess.run(["/usr/local/bin/python", str(Path(__file__).with_name("plot_loss.py")), str(root)],
                   env=env, check=False)


def rank_batches(batch, rows, world):
    """Keep batch membership fixed while assigning long sequences to lighter ranks."""
    groups = [[] for _ in range(world)]
    loads = [0] * world
    for index in sorted(batch, key=lambda i: len(rows[i][1]), reverse=True):
        rank = loads.index(min(loads))
        groups[rank].append(index)
        loads[rank] += len(rows[index][1])
    return groups


def train(args, rank, world, device):
    output = args.output.resolve()
    batch_size = args.batch_size or world
    tokenizer = AutoTokenizer.from_pretrained(MODEL, local_files_only=True)
    if rank == 0:
        output.mkdir(parents=True, exist_ok=True)
        if (output / "loss.jsonl").exists():
            raise FileExistsError(f"Output already contains a run: {output}")
        examples, coverage = {}, {}
        for split in ("train", "validation"):
            if args.objective in {"student_p1", "student_p2"}:
                examples[split], coverage[split] = tokenize_split(
                    tokenizer, args.data_dir, "val" if split == "validation" else split,
                    args.context, PREFIX_LENGTH, include_c=args.objective == "student_p2",
                )
            else:
                examples[split], coverage[split] = tokenize(
                    tokenizer, split, args.context, PREFIX_LENGTH, args.data_dir, args.objective,
                )
        torch.save(examples, output / "encoded.pt")
        write_json(output / "config.json", {"world_size": world, "epochs": args.epochs,
                   "context": args.context, "batch_size": batch_size, "prefix_tokens": PREFIX_LENGTH,
                   "learning_rate": args.lr, "seed": args.seed, "coverage": coverage,
                   "warmup_fraction": args.warmup_fraction, "max_prefix_norm": args.max_prefix_norm,
                   "balance_batch_lengths": args.balance_batch_lengths,
                   "data_dir": str(args.data_dir.resolve()), "objective": args.objective})
    dist.barrier()
    data = torch.load(output / "encoded.pt", weights_only=True, map_location="cpu")
    train_rows, val_rows = data["train"], data["validation"]
    model = AutoModelForImageTextToText.from_pretrained(
        MODEL, local_files_only=True, dtype=torch.bfloat16, attn_implementation="sdpa",
    ).to(device)
    model.requires_grad_(False).eval()
    model.config.use_cache = False
    model.gradient_checkpointing_enable(gradient_checkpointing_kwargs={"use_reentrant": False})
    if rank == 0:
        print(f"Model loaded. Training on {len(train_rows)} rows and validating on {len(val_rows)} with {world} GPUs.", flush=True)
    tokens = tokenizer.encode(COACH, add_special_tokens=False)[:PREFIX_LENGTH]
    prefix = torch.nn.Parameter(model.get_input_embeddings().weight[tokens].detach().float().clone())
    if args.initial_prefix:
        from safetensors.torch import load_file

        prefix.data.copy_(load_file(args.initial_prefix)["soft_prefix"])
    dist.broadcast(prefix.detach(), src=0)
    initial = prefix.detach().clone()
    has_initial_direction = bool(torch.count_nonzero(initial).item())
    optimizer = torch.optim.AdamW([prefix], lr=args.lr, weight_decay=0)
    warmup_steps = math.ceil(args.epochs * math.ceil(len(train_rows) / batch_size) * args.warmup_fraction)
    sample_mean = args.objective in {"student_p1", "student_p2"}
    started = time.monotonic()
    baseline = evaluate(model, val_rows, None, rank, world)
    initial_validation = evaluate(model, val_rows, initial, rank, world)
    if rank == 0:
        write_json(output / "baseline.json", {"no_prefix": baseline, "initial_prefix": initial_validation})
        append_metric(output, {"split": "validation", "epoch": 0, "step": 0,
                              "nll": initial_validation["combined"]["nll"], "by_source": initial_validation})
        print(f"Baselines: no prefix {baseline['combined']['nll']:.4f}, initial prefix {initial_validation['combined']['nll']:.4f}", flush=True)
    step = 0
    best_nll = float("inf")
    best_epoch = None
    summaries = []
    for epoch in range(1, args.epochs + 1):
        order = list(range(len(train_rows)))
        random.Random(args.seed + epoch).shuffle(order)
        model.train()
        running_loss = 0.0
        running_tokens = 0
        for start in range(0, len(order), batch_size):
            step += 1
            batch = order[start:start + batch_size]
            target_tokens = sum(len(train_rows[index][2]) for index in batch)
            local_loss = torch.zeros((), device=device, dtype=torch.float32)
            batch_objective = torch.zeros_like(local_loss)
            prefix.grad = None
            local_batch = (rank_batches(batch, train_rows, world)[rank]
                           if args.balance_batch_lengths else batch[rank::world])
            for index in local_batch:
                example = train_rows[index]
                loss = loss_sum(model, example, prefix)
                weight = example[4] if len(example) > 4 else 1.0
                objective = weight * loss / (len(example[2]) if sample_mean else target_tokens)
                (objective / (len(batch) if sample_mean else 1)).backward()
                local_loss += loss.detach()
                batch_objective += objective.detach()
            if prefix.grad is None:
                prefix.grad = torch.zeros_like(prefix)
            # Gradients already use the global denominator. Sum across ranks.
            dist.all_reduce(prefix.grad)
            dist.all_reduce(local_loss.detach())
            if sample_mean:
                dist.all_reduce(batch_objective.detach())
            norm = torch.nn.utils.clip_grad_norm_([prefix], 1.0, error_if_nonfinite=True)
            lr = args.lr * step / warmup_steps if step <= warmup_steps else args.lr
            optimizer.param_groups[0]["lr"] = lr
            optimizer.step()
            if args.max_prefix_norm is not None:
                with torch.no_grad():
                    scale = prefix.norm(dim=1, keepdim=True).clamp_min(args.max_prefix_norm)
                    prefix.mul_(args.max_prefix_norm / scale)
            batch_loss = local_loss.detach().item()
            running_loss += batch_loss
            running_tokens += target_tokens
            quarter = None
            if args.save_quarters:
                quarter = next((n for n in range(1, 5)
                                if start // batch_size + 1 == math.ceil(len(order) * n / (4 * batch_size))), None)
            if quarter:
                model.eval()
                quarter_validation = evaluate(model, val_rows, prefix, rank, world)
                model.train()
                if rank == 0:
                    save_prefix(output, f"quarter-{quarter}-soft-prefix.safetensors", prefix, args.objective)
                    write_json(output / f"quarter-{quarter}-validation.json", quarter_validation)
                    current = prefix.detach()
                    geometry = {
                        "mean_norm": current.norm(dim=1).mean().item(),
                        "max_norm": current.norm(dim=1).max().item(),
                        "mean_l2_from_initial": (current - initial).norm(dim=1).mean().item(),
                    }
                    if has_initial_direction:
                        geometry["mean_cosine"] = F.cosine_similarity(current, initial).mean().item()
                    write_json(output / f"quarter-{quarter}-geometry.json", geometry)
            if rank == 0:
                row = {"split": "train", "epoch": epoch, "step": step, "rows": len(batch),
                       "target_tokens": target_tokens, "nll": batch_loss / target_tokens,
                       "objective_loss": (batch_objective.item() / len(batch) if sample_mean
                                          else batch_loss / target_tokens),
                       "grad_norm": norm.item(), "learning_rate": lr,
                       "elapsed_seconds": round(time.monotonic() - started, 2)}
                append_metric(output, row)
                if args.save_half_epoch and start // batch_size + 1 == math.ceil(len(order) / (2 * batch_size)):
                    save_prefix(output, f"epoch-{epoch:02}-half-soft-prefix.safetensors", prefix, args.objective)
                if step == 1 or step % 10 == 0:
                    print(json.dumps(row), flush=True)
                    save_prefix(output, "prefix-in-progress.safetensors", prefix, args.objective)
                if step == 1 or step % 25 == 0:
                    render_curve(output)
        model.eval()
        validation = evaluate(model, val_rows, prefix, rank, world)
        if rank == 0:
            nll = validation["combined"]["nll"]
            row = {"split": "validation", "epoch": epoch, "step": step, "nll": nll, "by_source": validation,
                   "epoch_train_nll": running_loss / running_tokens}
            append_metric(output, row)
            print(json.dumps(row), flush=True)
            summaries.append(row)
            save_prefix(output, f"epoch-{epoch:02}-soft-prefix.safetensors", prefix, args.objective)
            if nll < best_nll:
                best_nll, best_epoch = nll, epoch
                save_prefix(output, "best-soft-prefix.safetensors", prefix, args.objective)
            render_curve(output)
    if rank == 0:
        delta = (prefix.detach() - initial).norm().item()
        assert delta > 0 and all(parameter.grad is None for parameter in model.parameters())
        save_prefix(output, "soft-prefix.safetensors", prefix, args.objective)
        write_json(output / "result.json", {"world_size": world, "train_rows": len(train_rows),
                   "validation_rows": len(val_rows), "optimizer_steps": step, "best_epoch": best_epoch,
                   "trainable_parameters": prefix.numel(), "prefix_change_l2": delta,
                   "no_prefix": baseline, "initial_prefix": initial_validation,
                   "epochs": summaries, "wall_seconds": round(time.monotonic() - started, 1)})
        (output / "encoded.pt").unlink()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--epochs", type=int, default=3)
    parser.add_argument("--save-half-epoch", action="store_true")
    parser.add_argument("--save-quarters", action="store_true")
    parser.add_argument("--seed", type=int, default=SEED)
    parser.add_argument("--initial-prefix", type=Path)
    parser.add_argument("--context", type=int, default=32768)
    parser.add_argument("--lr", type=float, default=0.003)
    parser.add_argument("--batch-size", type=int)
    parser.add_argument("--balance-batch-lengths", action="store_true")
    parser.add_argument("--warmup-fraction", type=float, default=0.0)
    parser.add_argument("--max-prefix-norm", type=float)
    parser.add_argument("--data-dir", type=Path, default=OUT)
    parser.add_argument("--objective", choices=("next_expert_turn", "direct_code", "student_p1", "student_p2"),
                        default="next_expert_turn")
    args = parser.parse_args()
    if args.batch_size is not None and args.batch_size < 1:
        parser.error("--batch-size must be positive")
    if not 0 <= args.warmup_fraction <= 1:
        parser.error("--warmup-fraction must be between 0 and 1")
    if args.max_prefix_norm is not None and args.max_prefix_norm <= 0:
        parser.error("--max-prefix-norm must be positive")
    local_rank = int(os.environ["LOCAL_RANK"])
    torch.cuda.set_device(local_rank)
    torch.manual_seed(args.seed)
    device = torch.device("cuda", local_rank)
    dist.init_process_group("nccl", timeout=timedelta(minutes=20), device_id=device)
    try:
        train(args, dist.get_rank(), dist.get_world_size(), device)
    finally:
        dist.destroy_process_group()


if __name__ == "__main__":
    main()
