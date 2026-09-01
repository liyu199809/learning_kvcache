from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

import torch
from safetensors import safe_open
from safetensors.torch import save_file

from prefix_tuning.virtual_prefix.merge_hybrid_prefix_partitions import merge_partitions
from prefix_tuning.virtual_prefix.split_hybrid_prefix_model import split_model


class SplitMergeHybridPrefixTest(unittest.TestCase):
    def setUp(self):
        self.temp_dir = tempfile.TemporaryDirectory()
        self.root = Path(self.temp_dir.name)
        self.source = self.root / "source"
        self.source.mkdir()

        delta = {}
        for layer in range(24):
            base = f"model.language_model.layers.{layer}.linear_attn"
            delta[f"{base}.prefix_k"] = torch.arange(16, dtype=torch.bfloat16).reshape(8, 2) + layer
            delta[f"{base}.prefix_v"] = torch.arange(24, dtype=torch.bfloat16).reshape(8, 3) + layer
            delta[f"{base}.prefix_beta_logits"] = torch.arange(8, dtype=torch.bfloat16).reshape(8, 1)
            delta[f"{base}.prefix_a"] = torch.arange(8, dtype=torch.bfloat16).reshape(8, 1)
        attention = {}
        for layer in range(8):
            base = f"model.language_model.layers.{layer}.self_attn"
            attention[f"{base}.prefix_key_tokens"] = (
                torch.arange(20, dtype=torch.bfloat16).reshape(4, 5) + layer
            )
            attention[f"{base}.prefix_value_tokens"] = (
                torch.arange(20, dtype=torch.bfloat16).reshape(4, 5) - layer
            )
        save_file(delta, self.source / "independent_delta_kv_prefix.safetensors")
        save_file(attention, self.source / "residual_attention_prefix.safetensors")
        save_file({"base.weight": torch.ones(2, 2, dtype=torch.bfloat16)}, self.source / "base.safetensors")

        delta_numel = sum(t.numel() for t in delta.values())
        attention_numel = sum(t.numel() for t in attention.values())
        config = {
            "model_type": "qwen3_5",
            "architectures": ["Qwen3_5HybridDeltaResidualAttentionPrefixForConditionalGeneration"],
            "text_config": {
                "independent_delta_prefix_num_virtual_tokens": 8,
                "residual_attention_prefix_num_virtual_tokens": 4,
            },
            "independent_delta_kv_prefix": {
                "num_virtual_tokens": 8,
                "trainable_numel": delta_numel,
            },
            "residual_attention_prefix": {
                "num_virtual_tokens": 4,
                "prefix_positions": [-4, -1],
                "trainable_numel": attention_numel,
            },
            "hybrid_prefix": {
                "delta_prefix_num_virtual_tokens": 8,
                "attention_prefix_num_virtual_tokens": 4,
                "trainable_numel": delta_numel + attention_numel,
            },
        }
        (self.source / "config.json").write_text(json.dumps(config))
        (self.source / "independent_delta_kv_prefix_config.json").write_text(
            json.dumps(config["independent_delta_kv_prefix"])
        )
        (self.source / "residual_attention_prefix_config.json").write_text(
            json.dumps(config["residual_attention_prefix"])
        )
        weight_map = {
            "base.weight": "base.safetensors",
            **{name: "independent_delta_kv_prefix.safetensors" for name in delta},
            **{name: "residual_attention_prefix.safetensors" for name in attention},
        }
        total_bytes = 8 + sum(t.numel() * t.element_size() for t in (*delta.values(), *attention.values()))
        (self.source / "model.safetensors.index.json").write_text(
            json.dumps({"metadata": {"total_size": total_bytes}, "weight_map": weight_map})
        )

    def tearDown(self):
        self.temp_dir.cleanup()

    def test_four_way_split_and_exact_merge(self):
        split_root = self.root / "split"
        result = split_model(self.source, split_root, 4, "hardlink")

        self.assertEqual(result["num_splits"], 4)
        for index in range(4):
            config = json.loads((split_root / f"cluster{index}" / "config.json").read_text())
            self.assertEqual(config["text_config"]["independent_delta_prefix_num_virtual_tokens"], 2)
            self.assertEqual(config["text_config"]["residual_attention_prefix_num_virtual_tokens"], 1)
            self.assertEqual(
                (self.source / "base.safetensors").stat().st_ino,
                (split_root / f"cluster{index}" / "base.safetensors").stat().st_ino,
            )

        output = self.root / "merged"
        merged = merge_partitions(
            [split_root / f"cluster{index}" for index in range(4)],
            output,
            torch.bfloat16,
            "hardlink",
            True,
        )
        self.assertTrue(merged["source_exact_verified"])
        for filename in (
            "independent_delta_kv_prefix.safetensors",
            "residual_attention_prefix.safetensors",
        ):
            with safe_open(self.source / filename, framework="pt", device="cpu") as expected_handle:
                with safe_open(output / filename, framework="pt", device="cpu") as actual_handle:
                    self.assertEqual(set(expected_handle.keys()), set(actual_handle.keys()))
                    for name in expected_handle.keys():
                        torch.testing.assert_close(actual_handle.get_tensor(name), expected_handle.get_tensor(name))


if __name__ == "__main__":
    unittest.main()
