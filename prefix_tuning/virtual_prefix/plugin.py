"""Register Qwen3.5 Delta virtual-prefix with ms-swift and vLLM."""

from __future__ import annotations

from transformers.dynamic_module_utils import get_class_from_dynamic_module
from swift.callbacks import TrainerCallback, callbacks_map
from swift.model import Model, ModelGroup, ModelLoader, ModelMeta, register_model
from swift.model.model_arch import ModelArch
from swift.model.models.qwen import Qwen3_5Loader
from swift.model.patcher import patch_get_input_embeddings

from prefix_tuning.virtual_prefix.vllm_plugin import register as register_vllm_model


ARCHITECTURE = "Qwen3_5DeltaVirtualPrefixForConditionalGeneration"
MODEL_TYPE = "qwen3_5_delta_virtual_prefix"
CALLBACK_NAME = "delta_virtual_prefix_only"


register_vllm_model()


class Qwen3_5DeltaVirtualPrefixLoader(Qwen3_5Loader):
    def get_model(self, model_dir, config, processor, model_kwargs):
        model_cls = get_class_from_dynamic_module(
            "modeling_qwen3_5_delta_virtual_prefix."
            "Qwen3_5DeltaVirtualPrefixForConditionalGeneration",
            model_dir,
        )
        self.auto_model_cls = model_cls
        # Do not apply ms-swift's packed/sequence-parallel Qwen3.5 GDN patch:
        # its alternate forward does not insert layer-specific virtual tokens.
        model = ModelLoader.get_model(self, model_dir, config, processor, model_kwargs)
        vision_tower = getattr(model, "visual", None)
        if vision_tower is None:
            vision_tower = model.model.visual
        patch_get_input_embeddings(vision_tower, "patch_embed")
        return model


register_model(
    ModelMeta(
        MODEL_TYPE,
        [
            ModelGroup(
                [
                    Model(
                        "/mnt/storage/disk1/verl_data/base_model/"
                        "Qwen3.5-4B-DeltaVirtualPrefix-256"
                    )
                ]
            )
        ],
        Qwen3_5DeltaVirtualPrefixLoader,
        model_arch=ModelArch.qwen2_vl,
        architectures=[ARCHITECTURE],
        template="qwen3_5",
        requires=["transformers>=5.0.0", "qwen_vl_utils>=0.0.14"],
        tags=["vision", "video"],
        additional_saved_files=[
            "modeling_qwen3_5_delta_virtual_prefix.py",
            "delta_virtual_prefix_config.json",
        ],
    )
)


class DeltaVirtualPrefixOnlyCallback(TrainerCallback):
    def __init__(self, args, trainer):
        super().__init__(args, trainer)
        if float(args.weight_decay) != 0.0:
            raise ValueError("Delta virtual-prefix tuning requires --weight_decay 0.0")

        prefix_parameters = []
        for name, parameter in trainer.model.named_parameters():
            parameter.requires_grad_(False)
            if name.endswith(".linear_attn.prefix_tokens"):
                parameter.requires_grad_(True)
                prefix_parameters.append((name, parameter))

        expected_layers = sum(
            layer_type == "linear_attention"
            for layer_type in trainer.model.config.text_config.layer_types
        )
        expected_shape = (
            int(trainer.model.config.text_config.delta_prefix_num_virtual_tokens),
            int(trainer.model.config.text_config.hidden_size),
        )
        bad_shapes = [
            (name, tuple(parameter.shape))
            for name, parameter in prefix_parameters
            if tuple(parameter.shape) != expected_shape
        ]
        if len(prefix_parameters) != expected_layers or bad_shapes:
            raise RuntimeError(
                f"Expected {expected_layers} virtual-prefix tensors of shape "
                f"{expected_shape}; found {len(prefix_parameters)}, bad_shapes={bad_shapes}"
            )
        effective_parameters = sum(parameter.numel() for _, parameter in prefix_parameters)
        print(
            f"[qwen3.5-delta-virtual-prefix] tensors={len(prefix_parameters)} "
            f"shape={expected_shape} trainable_parameters={effective_parameters:,}"
        )


callbacks_map[CALLBACK_NAME] = DeltaVirtualPrefixOnlyCallback
