"""Collect attention gate and prefix-to-native output ratios by sequence third."""
from functools import partial
import json
from pathlib import Path
import sys

import torch


class AnchorGateMetrics:
    def __init__(self, model):
        self.modules = [layer.linear_attn if hasattr(layer, "linear_attn") else layer.self_attn
                        for layer in model.model.language_model.layers]
        self.values = torch.zeros(len(self.modules), 3, 5, device=model.device, dtype=torch.float64)
        self.length = 0

    def enable(self):
        for index, module in enumerate(self.modules):
            module.prefix_monitor = partial(self._observe, index)

    def disable(self):
        for module in self.modules:
            module.prefix_monitor = None

    @torch.no_grad()
    def _observe(self, index, mass, contribution, native):
        # Recomputed blocks in the backward pass run after monitoring is disabled.
        ratio = contribution[:, :, :self.length].float().norm(dim=-1) / (
            native[:, :self.length].transpose(1, 2).float().norm(dim=-1).clamp_min(1e-12))
        for third in range(3):
            start, end = self.length * third // 3, self.length * (third + 1) // 3
            segment = mass[:, :, start:end, 0].float()
            if not segment.numel():
                continue
            self.values[index, third, 0] += segment.sum(dtype=torch.float64)
            self.values[index, third, 1] += ratio[:, :, start:end].sum(dtype=torch.float64)
            self.values[index, third, 2] += ratio[:, :, start:end].numel()
            self.values[index, third, 3] += segment.std(dim=-1, unbiased=False).sum(dtype=torch.float64)
            self.values[index, third, 4] += segment.shape[0] * segment.shape[1]

    def rows(self):
        return [
            {"layer": index + 1, "type": "gdn" if hasattr(module, "prefix_state_key") else "full_attention",
             "thirds": [{"gate": gate / count, "output_ratio": ratio / count, "count": int(count),
                         "mean_within_sequence_gate_std": std / heads}
                        if count else {"gate": None, "output_ratio": None, "count": 0,
                                       "mean_within_sequence_gate_std": None}
                        for gate, ratio, count, std, heads in layers]}
            for index, (module, layers) in enumerate(zip(self.modules, self.values.tolist()))
        ]


def plot_metrics(output):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    output = Path(output)
    with (output / "gate-metrics.jsonl").open() as stream:
        log = [json.loads(line) for line in stream]
    figure, axes = plt.subplots(2, 2, figsize=(12, 7), sharex=True, layout="constrained")
    for column, layer_type in enumerate(("gdn", "full_attention")):
        for row, metric in enumerate(("gate", "output_ratio")):
            axis = axes[row, column]
            for third, label in enumerate(("Beginning", "Middle", "End")):
                points = [[layer["thirds"][third] for layer in record["layers"]
                           if layer["type"] == layer_type] for record in log]
                means = [sum(p[metric] * p["count"] for p in group if p["count"]) /
                         sum(p["count"] for p in group) for group in points]
                axis.plot([record["step"] for record in log], means, marker=".", label=label)
            axis.set_title(layer_type)
            axis.set_ylabel(metric)
            axis.grid(alpha=.25)
            axis.legend()
    for axis in axes[-1]:
        axis.set_xlabel("Optimizer step")
    temporary = output / "gate-curve.tmp.png"
    figure.savefig(temporary, dpi=160)
    temporary.replace(output / "gate-curve.png")
    plt.close(figure)


if __name__ == "__main__":
    plot_metrics(sys.argv[1])
