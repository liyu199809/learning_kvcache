from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

import torch
from safetensors import safe_open
from safetensors.torch import save_file

from prefix_tuning.virtual_prefix.merge_prefix_checkpoint import merge_prefix_checkpoint


class MergePrefixCheckpointTest(unittest.TestCase):
    def setUp(self):
        self.temp_dir = tempfile.TemporaryDirectory()
        self.root = Path(self.temp_dir.name)

    def tearDown(self):
        self.temp_dir.cleanup()

    def _fixture(self, trainable_type: str, *, legacy_type: bool = False):
        prepared = self.root / f"prepared-{trainable_type}"
        checkpoint = self.root / f"checkpoint-{trainable_type}"
        output = self.root / f"output-{trainable_type}"
        prepared.mkdir()
        checkpoint.mkdir()
        (prepared / "tokenizer.json").write_text("{}")
        save_file({"base.weight": torch.ones(2, 2)}, prepared / "base.safetensors")

        if trainable_type == "delta_virtual_prefix":
            weights_file = "delta_virtual_prefix.safetensors"
            config_key = "delta_virtual_prefix"
            virtual_tokens_key = "delta_prefix_num_virtual_tokens"
            tensors = {
                "model.language_model.layers.0.linear_attn.prefix_tokens": torch.arange(
                    12, dtype=torch.float32
                ).reshape(3, 4)
            }
        else:
            weights_file = "independent_delta_kv_prefix.safetensors"
            config_key = "independent_delta_kv_prefix"
            virtual_tokens_key = "independent_delta_prefix_num_virtual_tokens"
            base = "model.language_model.layers.0.linear_attn"
            tensors = {
                f"{base}.prefix_k": torch.arange(6, dtype=torch.float32).reshape(3, 2),
                f"{base}.prefix_v": torch.arange(12, dtype=torch.float32).reshape(3, 4),
                f"{base}.prefix_beta_logits": torch.arange(3, dtype=torch.float32).reshape(3, 1),
                f"{base}.prefix_a": torch.arange(3, dtype=torch.float32).reshape(3, 1),
            }

        prepared_tensors = {name: torch.zeros_like(tensor, dtype=torch.bfloat16) for name, tensor in tensors.items()}
        save_file(prepared_tensors, prepared / weights_file)
        config = {
            "model_type": "qwen3_5",
            "text_config": {virtual_tokens_key: 3},
            config_key: {"num_virtual_tokens": 3},
        }
        (prepared / "config.json").write_text(json.dumps(config))
        prefix_bytes = sum(tensor.numel() * tensor.element_size() for tensor in prepared_tensors.values())
        index = {
            "metadata": {"total_size": 16 + prefix_bytes},
            "weight_map": {
                "base.weight": "base.safetensors",
                **{name: weights_file for name in tensors},
            },
        }
        (prepared / "model.safetensors.index.json").write_text(json.dumps(index))

        parameters = {
            name: {"shape": list(tensor.shape), "numel": tensor.numel(), "dtype": str(tensor.dtype)}
            for name, tensor in tensors.items()
        }
        metadata = {
            "checkpoint_type": "trainable_only",
            "trainable_type": None if legacy_type else trainable_type,
            "world_size": 1,
            "base_model_path": str(prepared),
            "num_virtual_tokens": 3,
            "parameter_names": list(tensors),
            "parameters": parameters,
        }
        (checkpoint / "trainable_only_meta.json").write_text(json.dumps(metadata))
        torch.save(tensors, checkpoint / "model_world_size_1_rank_0.pt")
        return prepared, checkpoint, output, tensors, weights_file

    def test_merges_legacy_delta_checkpoint_and_uses_metadata_model_path(self):
        prepared, checkpoint, output, tensors, weights_file = self._fixture(
            "delta_virtual_prefix", legacy_type=True
        )

        result = merge_prefix_checkpoint(checkpoint, output)

        self.assertEqual(result["trainable_type"], "delta_virtual_prefix")
        self.assertTrue(output.is_dir())
        self.assertTrue((output / "prefix_merge_manifest.json").is_file())
        self.assertEqual((prepared / "base.safetensors").stat().st_ino, (output / "base.safetensors").stat().st_ino)
        with safe_open(output / weights_file, framework="pt", device="cpu") as handle:
            actual = handle.get_tensor(next(iter(tensors)))
        torch.testing.assert_close(actual, next(iter(tensors.values())).to(torch.bfloat16))

    def test_merges_independent_checkpoint(self):
        _, checkpoint, output, tensors, weights_file = self._fixture(
            "independent_delta_kv_prefix"
        )

        result = merge_prefix_checkpoint(checkpoint, output, output_dtype=torch.float32)

        self.assertEqual(result["prefix_tensors"], 4)
        with safe_open(output / weights_file, framework="pt", device="cpu") as handle:
            self.assertEqual(set(handle.keys()), set(tensors))
            for name, expected in tensors.items():
                torch.testing.assert_close(handle.get_tensor(name), expected)
        index = json.loads((output / "model.safetensors.index.json").read_text())
        expected_total = 16 + sum(tensor.numel() * 4 for tensor in tensors.values())
        self.assertEqual(index["metadata"]["total_size"], expected_total)

    def test_rejects_prepared_model_layout_mismatch_without_partial_output(self):
        prepared, checkpoint, output, _, weights_file = self._fixture("delta_virtual_prefix")
        bad_name = "model.language_model.layers.1.linear_attn.prefix_tokens"
        save_file({bad_name: torch.zeros(3, 4, dtype=torch.bfloat16)}, prepared / weights_file)

        with self.assertRaisesRegex(ValueError, "invalid prefix mappings|layout does not match"):
            merge_prefix_checkpoint(checkpoint, output, prepared)

        self.assertFalse(output.exists())

    def test_type_specific_entry_point_rejects_other_prefix_type(self):
        _, checkpoint, output, _, _ = self._fixture("independent_delta_kv_prefix")

        with self.assertRaisesRegex(ValueError, "Expected 'delta_virtual_prefix'"):
            merge_prefix_checkpoint(
                checkpoint,
                output,
                expected_trainable_type="delta_virtual_prefix",
            )


if __name__ == "__main__":
    unittest.main()
