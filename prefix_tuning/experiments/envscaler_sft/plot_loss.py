"""Render logged SFT loss and per-source validation curves."""
from collections import deque
import json
import math
from pathlib import Path
import sys

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt


def main(root):
    with (root / "loss.jsonl").open() as stream:
        rows = [json.loads(line) for line in stream]
    training = [r for r in rows if r["split"] == "train"]
    validation = [r for r in rows if r["split"] == "validation"]
    config_path = root / "config.json"
    if config_path.is_file():
        config = json.loads(config_path.read_text())
        if config.get("epochs") == 1 and config.get("batch_size") and "train" in config.get("coverage", {}):
            train_rows = sum(source["included"] for source in config["coverage"]["train"].values())
            by_step = {row["step"]: row["by_source"] for row in validation}
            for quarter in range(1, 5):
                checkpoint = root / f"quarter-{quarter}-validation.json"
                if checkpoint.is_file():
                    step = math.ceil(train_rows * quarter / (4 * config["batch_size"]))
                    if step not in by_step:
                        by_step[step] = json.loads(checkpoint.read_text())
            validation = [{"step": step, "by_source": by_step[step]} for step in sorted(by_step)]
    fig, (train_ax, val_ax) = plt.subplots(2, 1, figsize=(10, 7), sharex=True, layout="constrained")
    if training:
        rolling = deque(maxlen=15)
        smoothed = []
        for row in training:
            rolling.append((row["nll"] * row["target_tokens"], row["target_tokens"]))
            smoothed.append(sum(loss for loss, _ in rolling) / sum(tokens for _, tokens in rolling))
        steps = [r["step"] for r in training]
        train_ax.plot(steps, [r["nll"] for r in training], alpha=.25, label="Batch NLL")
        train_ax.plot(steps, smoothed, label="15-step token-weighted NLL")
        train_ax.legend()
    train_ax.set_ylabel("Training NLL")
    if validation:
        if "by_source" in validation[0]:
            for source in validation[0]["by_source"]:
                val_ax.plot([r["step"] for r in validation], [r["by_source"][source]["nll"] for r in validation],
                            marker="o", label=source)
        else:
            val_ax.plot([r["step"] for r in validation], [r["nll"] for r in validation],
                        marker="o", label="EnvScaler")
        val_ax.legend()
    val_ax.set_ylabel("Validation NLL")
    val_ax.set_xlabel("Optimizer step")
    for ax in (train_ax, val_ax):
        ax.grid(alpha=.25)
    fig.savefig(root / "loss-curve.tmp.png", dpi=160)
    (root / "loss-curve.tmp.png").replace(root / "loss-curve.png")
    plt.close(fig)


if __name__ == "__main__":
    main(Path(sys.argv[1]))
