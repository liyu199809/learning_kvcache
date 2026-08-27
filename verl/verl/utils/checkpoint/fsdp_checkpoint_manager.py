# Copyright 2024 Bytedance Ltd. and/or its affiliates
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import json
import logging
import os
import shutil
import warnings
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Iterable, Optional

import torch
import torch.distributed
from accelerate import init_empty_weights
from omegaconf import DictConfig
from torch.distributed.fsdp import FullyShardedDataParallel as FSDP
from torch.distributed.fsdp import ShardedOptimStateDictConfig, ShardedStateDictConfig, StateDictType
from transformers import GenerationConfig, PreTrainedTokenizer, ProcessorMixin
from transformers.dynamic_module_utils import custom_object_save

from verl.utils.device import is_cuda_available
from verl.utils.fs import copy_to_local, is_non_local, local_mkdir_safe
from verl.utils.fsdp_utils import fsdp_version, get_fsdp_full_state_dict, get_fsdp_state_ctx
from verl.utils.logger import log_with_rank
from verl.utils.transformers_compat import drop_tied_target_keys, get_auto_model_for_vision2seq

from .checkpoint_manager import BaseCheckpointManager

# Setup logging
logger = logging.getLogger(__file__)
logger.setLevel(os.getenv("VERL_LOGGING_LEVEL", "INFO"))


@dataclass
class FSDPConfig:
    """Configuration for FSDP checkpointing.

    Args:
        FSDP_version (int): Version of FSDP being used.
        world_size (int): Number of processes in the distributed training setup.
    """

    FSDP_version: int
    world_size: int


class FSDPCheckpointManager(BaseCheckpointManager):
    """
    Manage FSDP checkpointing in SPMD training.

    - Saves/loads per-rank sharded model & optimizer states
    - Persists full lr_scheduler and RNG state
    - Stores HF tokenizer/processor and model/config for unified restore

    Args:
        model (FSDP): Wrapped model instance.
        optimizer (Optimizer): Training optimizer.
        lr_scheduler (LRScheduler): Learning-rate scheduler.
        processing_class (PreTrainedTokenizer or ProcessorMixin, optional):
            Pre-/post-processing artifact handler.
        checkpoint_contents DictConfig: Configuration for checkpoint contents.
            - 'load': Components to load; must contain 'model'. Defaults to ['model', 'optimizer', 'extra'].
            - 'save': Components to save; must contain 'model'. Defaults to ['model', 'optimizer', 'extra'].
        trust_remote_code: Whether to trust_remote_code when loading the model configuration
    """

    def __init__(
        self,
        model: FSDP,
        optimizer: Optional[torch.optim.Optimizer] = None,
        lr_scheduler: Optional[torch.optim.lr_scheduler.LRScheduler] = None,
        processing_class: PreTrainedTokenizer | ProcessorMixin = None,
        checkpoint_config: DictConfig = None,
        trust_remote_code: bool = False,
        trainable_param_names: Optional[Iterable[str]] = None,
        base_model_path: Optional[str] = None,
        **kwargs,
    ):
        if processing_class is None and "tokenizer" in kwargs:
            warnings.warn(
                "`tokenizer` is deprecated. use `processing_class` instead.", DeprecationWarning, stacklevel=2
            )
            processing_class = kwargs.pop("tokenizer")

        super().__init__(
            model,
            optimizer,
            lr_scheduler=lr_scheduler,
            processing_class=processing_class,
            checkpoint_config=checkpoint_config,
        )
        self.trust_remote_code = trust_remote_code
        self.trainable_param_order = tuple(trainable_param_names or ())
        self.trainable_param_names = set(self.trainable_param_order)
        self.base_model_path = base_model_path

    @staticmethod
    def _trainable_only_meta_path(local_path: str) -> str:
        return os.path.join(local_path, "trainable_only_meta.json")

    def _filter_trainable_state_dict(self, state_dict: dict) -> dict:
        if not self.trainable_param_names:
            raise ValueError("save_trainable_only=True requires explicit trainable parameter names")
        filtered = type(state_dict)(
            (name, value) for name, value in state_dict.items() if name in self.trainable_param_names
        )
        missing = self.trainable_param_names.difference(filtered)
        if missing:
            raise ValueError(
                "Checkpoint state dict is missing selected trainable parameters: " + ", ".join(sorted(missing))
            )
        return filtered

    def _build_trainable_only_meta(self, state_dict: dict) -> dict:
        state_order = tuple(state_dict)
        if state_order != self.trainable_param_order:
            raise ValueError(
                "Trainable-only checkpoint parameter order differs from optimizer/model order: "
                f"checkpoint={state_order}, configured={self.trainable_param_order}"
            )
        parameters = {
            name: {
                "shape": list(value.shape),
                "numel": int(value.numel()),
                "dtype": str(value.dtype),
            }
            for name, value in state_dict.items()
        }
        unwrap = getattr(self.model, "_fsdp_wrapped_module", self.model)
        config = getattr(unwrap, "config", None)
        text_config = getattr(config, "text_config", config)
        independent_delta_tokens = getattr(
            text_config, "independent_delta_prefix_num_virtual_tokens", None
        )
        residual_attention_tokens = getattr(
            text_config, "residual_attention_prefix_num_virtual_tokens", None
        )
        if independent_delta_tokens is not None and residual_attention_tokens is not None:
            trainable_type = "hybrid_delta_residual_attention_prefix"
            num_virtual_tokens = independent_delta_tokens
        elif independent_delta_tokens is not None:
            trainable_type = "independent_delta_kv_prefix"
            num_virtual_tokens = independent_delta_tokens
        elif getattr(text_config, "delta_prefix_num_virtual_tokens", None) is not None:
            trainable_type = "delta_virtual_prefix"
            num_virtual_tokens = text_config.delta_prefix_num_virtual_tokens
        else:
            trainable_type = "selected_parameters"
            num_virtual_tokens = None
        metadata = {
            "checkpoint_type": "trainable_only",
            "trainable_type": trainable_type,
            "world_size": self.world_size,
            "base_model_path": self.base_model_path,
            "num_virtual_tokens": num_virtual_tokens,
            "parameter_names": list(state_order),
            "parameter_count": len(parameters),
            "total_numel": sum(item["numel"] for item in parameters.values()),
            "parameters": parameters,
        }
        if residual_attention_tokens is not None:
            metadata["attention_num_virtual_tokens"] = residual_attention_tokens
        return metadata

    def _load_trainable_only_meta(self, local_path: str) -> Optional[dict]:
        path = self._trainable_only_meta_path(local_path)
        if not os.path.isfile(path):
            return None
        with open(path, encoding="utf-8") as handle:
            metadata = json.load(handle)
        if metadata.get("checkpoint_type") != "trainable_only":
            raise ValueError(f"Unsupported checkpoint metadata in {path}: {metadata.get('checkpoint_type')!r}")
        if metadata.get("world_size") != self.world_size:
            raise ValueError(
                f"Trainable-only checkpoint world size {metadata.get('world_size')} does not match {self.world_size}"
            )
        metadata_order = tuple(metadata.get("parameter_names", metadata.get("parameters", {})))
        metadata_names = set(metadata_order)
        if metadata_names != self.trainable_param_names:
            raise ValueError(
                "Trainable-only checkpoint parameter names do not match the current model: "
                f"missing={sorted(self.trainable_param_names - metadata_names)}, "
                f"unexpected={sorted(metadata_names - self.trainable_param_names)}"
            )
        if metadata_order != self.trainable_param_order:
            raise ValueError(
                "Trainable-only checkpoint parameter order does not match the current optimizer/model order: "
                f"checkpoint={metadata_order}, current={self.trainable_param_order}"
            )
        # Normalize checkpoints written before parameter_names became explicit;
        # JSON object insertion order already recorded the optimizer/model order.
        metadata["parameter_names"] = list(metadata_order)
        return metadata

    def _get_lora_train_meta(self, unwrap_model):
        peft_config = getattr(unwrap_model, "peft_config", None)
        if not peft_config:
            return None
        if isinstance(peft_config, dict):
            peft_config = peft_config.get("default") or next(iter(peft_config.values()), None)
        if peft_config is None:
            return None

        lora_rank = int(getattr(peft_config, "r", 0) or 0)
        if lora_rank <= 0:
            return None

        lora_alpha = int(getattr(peft_config, "lora_alpha", lora_rank) or 0)
        task_type = getattr(peft_config, "task_type", None) or "CAUSAL_LM"
        if hasattr(task_type, "value"):
            task_type = task_type.value

        return {"r": lora_rank, "lora_alpha": lora_alpha, "task_type": str(task_type)}

    def _save_lora_train_meta(self, local_path: str, unwrap_model):
        lora_train_meta = self._get_lora_train_meta(unwrap_model)
        if lora_train_meta is None:
            return None

        lora_meta_path = os.path.join(local_path, "lora_train_meta.json")
        with open(lora_meta_path, "w", encoding="utf-8") as f:
            json.dump(lora_train_meta, f, ensure_ascii=False, indent=4)
        log_with_rank(
            f"Saved LoRA rank/alpha metadata to {os.path.abspath(lora_meta_path)}",
            rank=self.rank,
            logger=logger,
            log_only_rank_0=True,
        )
        return lora_meta_path

    def _has_lora(self) -> bool:
        unwrap = getattr(self.model, "_fsdp_wrapped_module", self.model)
        return hasattr(unwrap, "peft_config")

    def load_checkpoint(self, local_path: str, hdfs_path: str = None, del_local_after_load=False):
        """
        Load an FSDP checkpoint for this rank.

        Downloads and loads:
          - model and optimizer shards
          - extra state dict (scheduler + RNG)

        Args:
            local_path: Directory with per-rank checkpoint files.
            hdfs_path: Unused (for API compatibility).
            del_local_after_load: Remove local files after loading.
        """
        if local_path is None:
            return

        trainable_only_meta = self._load_trainable_only_meta(local_path)

        # check if the checkpoint_load_contents is valid
        if self.should_load_model:
            assert self.model is not None, "model must be provided when checkpoint_contents.load includes ['model']"
        if self.should_load_optimizer:
            assert self.optimizer is not None, (
                "optimizer must be provided when checkpoint_contents.load includes ['optimizer']"
            )

        # every rank download its own checkpoint
        state_dict_cfg = (
            ShardedStateDictConfig(offload_to_cpu=True if is_cuda_available else False)
            if self.should_load_model
            else None
        )
        optim_cfg = (
            ShardedOptimStateDictConfig(offload_to_cpu=True if is_cuda_available else False)
            if self.should_load_optimizer
            else None
        )
        with get_fsdp_state_ctx(self.model, StateDictType.SHARDED_STATE_DICT, state_dict_cfg, optim_cfg):
            if self.should_load_model:
                remote_model_path = os.path.join(local_path, f"model_world_size_{self.world_size}_rank_{self.rank}.pt")
                local_model_path = copy_to_local(remote_model_path)
                model_state_dict = torch.load(local_model_path, weights_only=False)
                if trainable_only_meta is not None:
                    expected = trainable_only_meta["parameters"]
                    actual_names = set(model_state_dict)
                    if actual_names != self.trainable_param_names:
                        raise ValueError(
                            "Trainable-only checkpoint shard keys do not match configured trainable parameters: "
                            f"missing={sorted(self.trainable_param_names - actual_names)}, "
                            f"unexpected={sorted(actual_names - self.trainable_param_names)}"
                        )
                    if tuple(model_state_dict) != tuple(trainable_only_meta["parameter_names"]):
                        raise ValueError(
                            "Trainable-only checkpoint shard parameter order does not match metadata"
                        )
                    for name, value in model_state_dict.items():
                        if list(value.shape) != expected[name]["shape"]:
                            raise ValueError(
                                f"Trainable-only checkpoint shape mismatch for {name}: "
                                f"got {list(value.shape)}, expected {expected[name]['shape']}"
                            )
                    result = self.model.load_state_dict(model_state_dict, strict=False)
                    if result is not None and result.unexpected_keys:
                        raise ValueError(
                            f"Failed to load trainable-only checkpoint: unexpected keys {result.unexpected_keys}"
                        )
                    missing_trainable = self.trainable_param_names.intersection(
                        getattr(result, "missing_keys", ())
                    )
                    if missing_trainable:
                        raise ValueError(
                            "Failed to restore trainable parameters: " + ", ".join(sorted(missing_trainable))
                        )
                    log_with_rank(
                        f"Loaded trainable-only checkpoint ({len(model_state_dict)} keys) from {remote_model_path}",
                        rank=self.rank,
                        logger=logger,
                    )
                elif self.is_lora_only_state_dict(model_state_dict):
                    result = self.model.load_state_dict(model_state_dict, strict=False)
                    if result is not None and result.unexpected_keys:
                        raise ValueError(
                            f"Failed to load LoRA-only checkpoint: unexpected keys {result.unexpected_keys}. "
                            f"Ensure the model has the correct LoRA adapters configured."
                        )
                    log_with_rank(
                        f"Loaded LoRA-only checkpoint ({len(model_state_dict)} keys) from {remote_model_path}",
                        rank=self.rank,
                        logger=logger,
                    )
                else:
                    self.model.load_state_dict(model_state_dict)
                    log_with_rank(f"Loaded model from {remote_model_path}", rank=self.rank, logger=logger)

            if self.should_load_optimizer:
                remote_optim_path = os.path.join(local_path, f"optim_world_size_{self.world_size}_rank_{self.rank}.pt")
                local_optim_path = copy_to_local(remote_optim_path)
                optimizer_state_dict = torch.load(local_optim_path, weights_only=False)
                self.optimizer.load_state_dict(optimizer_state_dict)
                log_with_rank(f"Loaded optimizer from {remote_optim_path}", rank=self.rank, logger=logger)

        if self.should_load_extra:
            remote_extra_state_path = os.path.join(
                local_path, f"extra_state_world_size_{self.world_size}_rank_{self.rank}.pt"
            )
            local_extra_state_path = copy_to_local(remote_extra_state_path)
            extra_state_dict = torch.load(local_extra_state_path, weights_only=False)
            # recover random state
            if "rng" in extra_state_dict:
                # 'rng' may not exist for backward compatibility
                self.load_rng_state(extra_state_dict["rng"])
                log_with_rank(f"Loaded rng from {remote_extra_state_path}", rank=self.rank, logger=logger)

            lr_scheduler_state_dict = extra_state_dict["lr_scheduler"]
            if lr_scheduler_state_dict is not None and self.lr_scheduler is not None:
                self.lr_scheduler.load_state_dict(lr_scheduler_state_dict)
                log_with_rank(f"Loaded lr_scheduler from {remote_extra_state_path}", rank=self.rank, logger=logger)

        if self.rank == 0 and del_local_after_load:
            try:
                os.remove(local_model_path) if is_non_local(local_model_path) else None
                os.remove(local_optim_path) if is_non_local(local_optim_path) else None
                os.remove(local_extra_state_path) if is_non_local(local_extra_state_path) else None
            except Exception as e:
                log_with_rank(
                    f"remove local resume ckpt file after loading failed, exception {e} will be ignored",
                    rank=self.rank,
                    logger=logger,
                )

        # wait for everyone to load checkpoints
        torch.distributed.barrier()

    def save_checkpoint(self, local_path: str, hdfs_path: str = None, global_step: int = 0, max_ckpt_to_keep=None):
        """
        Save an FSDP checkpoint for this rank.

        Writes:
          - model & optimizer shard files
          - extra state dict (scheduler + RNG)
          - HF tokenizer/processor and model/config on rank 0
          - optional full HF model under 'huggingface/' if requested

        Rotates old checkpoints, keeping at most `max_ckpt_to_keep`.

        Args:
            local_path: Target directory for checkpoint files.
            hdfs_path: Unused (for API compatibility).
            global_step: Current training step (used for bookkeeping).
            max_ckpt_to_keep: Number of recent checkpoints to retain.
        """
        if local_path is None:
            return

        # record the previous global step
        self.previous_global_step = global_step

        if self.rank == 0:
            self.ensure_checkpoint_capacity(max_ckpt_to_keep)

        local_path = local_mkdir_safe(local_path)
        torch.distributed.barrier()

        if self.should_save_trainable_only and self.should_save_lora_only:
            raise ValueError("save_trainable_only and save_lora_only are mutually exclusive")
        trainable_only_meta = None

        # check if the checkpoint_save_contents is valid
        if self.should_save_model:
            assert self.model is not None, "model must be provided when checkpoint_contents.save includes ['model']"
        if self.should_save_optimizer:
            assert self.optimizer is not None, (
                "optimizer must be provided when checkpoint_contents.save includes ['optimizer']"
            )

        # every rank will save its own model and optim shard
        state_dict_cfg = ShardedStateDictConfig(offload_to_cpu=True if is_cuda_available else False)
        optim_cfg = ShardedOptimStateDictConfig(offload_to_cpu=True if is_cuda_available else False)
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            with get_fsdp_state_ctx(self.model, StateDictType.SHARDED_STATE_DICT, state_dict_cfg, optim_cfg):
                model_path = os.path.join(local_path, f"model_world_size_{self.world_size}_rank_{self.rank}.pt")
                optim_path = os.path.join(local_path, f"optim_world_size_{self.world_size}_rank_{self.rank}.pt")
                extra_path = os.path.join(local_path, f"extra_state_world_size_{self.world_size}_rank_{self.rank}.pt")

                if self.should_save_model:
                    model_state_dict = self.model.state_dict()
                    if self.should_save_trainable_only:
                        model_state_dict = self._filter_trainable_state_dict(model_state_dict)
                        trainable_only_meta = self._build_trainable_only_meta(model_state_dict)
                        log_with_rank(
                            f"Trainable-only save: {len(model_state_dict)} tensors "
                            f"({trainable_only_meta['total_numel']} elements)",
                            rank=self.rank,
                            logger=logger,
                            log_only_rank_0=True,
                        )
                    elif self.should_save_lora_only and self._has_lora():
                        n_total = len(model_state_dict)
                        model_state_dict = {
                            k: v for k, v in model_state_dict.items() if "lora_" in k or ".adapter_" in k
                        }
                        if not model_state_dict:
                            raise ValueError(
                                f"save_lora_only is True and the model has a peft_config, "
                                f"but no LoRA/adapter parameters were found in the state dict. "
                                f"Total params checked: {n_total}."
                            )
                        lora_bytes = 0
                        for v in model_state_dict.values():
                            if hasattr(v, "numel") and hasattr(v, "element_size"):
                                lora_bytes += v.numel() * v.element_size()
                            elif hasattr(v, "local_shards"):
                                for shard in v.local_shards():
                                    lora_bytes += shard.tensor.numel() * shard.tensor.element_size()
                        lora_mib = lora_bytes / 1024**2
                        log_with_rank(
                            f"LoRA-only save: {len(model_state_dict)}/{n_total} params ({lora_mib:.1f} MiB)",
                            rank=self.rank,
                            logger=logger,
                            log_only_rank_0=True,
                        )
                    torch.save(model_state_dict, model_path)
                    log_with_rank(f"Saved model to {os.path.abspath(model_path)}", rank=self.rank, logger=logger)

                if self.should_save_optimizer:
                    optimizer_state_dict = self.optimizer.state_dict()
                    torch.save(optimizer_state_dict, optim_path)
                    log_with_rank(f"Saved optim to {os.path.abspath(optim_path)}", rank=self.rank, logger=logger)

                if self.should_save_extra:
                    lr_scheduler_state_dict = self.lr_scheduler.state_dict() if self.lr_scheduler is not None else None
                    extra_state_dict = {
                        "lr_scheduler": lr_scheduler_state_dict,
                        "rng": self.get_rng_state(),
                    }
                    torch.save(extra_state_dict, extra_path)
                    log_with_rank(f"Saved extra_state to {os.path.abspath(extra_path)}", rank=self.rank, logger=logger)

        if self.rank == 0:
            # Save HF tokenizer/processor and model config on rank 0 to huggingface/ directory, no matter whether
            # huggingface model is requested to be saved or not.

            if fsdp_version(self.model) == 1:
                unwrap_model = self.model._fsdp_wrapped_module
            else:
                unwrap_model = self.model

            hf_config_tokenizer_path = os.path.join(local_path, "huggingface")
            local_mkdir_safe(hf_config_tokenizer_path)
            model_config = unwrap_model.config
            generation_config = None
            if unwrap_model.can_generate() and hasattr(model_config, "name_or_path") and model_config.name_or_path:
                try:
                    # Some model's name_or_path is empty if not initialized from pretrained,
                    # in this cases, we don't save generation config.
                    generation_config = GenerationConfig.from_pretrained(model_config.name_or_path)
                    generation_config.save_pretrained(hf_config_tokenizer_path)
                except Exception:
                    # if the generation config isn't available, we don't save it
                    pass

            if hasattr(model_config, "auto_map") and None in model_config.auto_map:
                model_config.auto_map = {k: v for k, v in model_config.auto_map.items() if k is not None}

            model_config.save_pretrained(hf_config_tokenizer_path)
            if self.processing_class is not None:
                self.processing_class.save_pretrained(hf_config_tokenizer_path)
            log_with_rank(
                f"Saved model config and tokenizer class to {os.path.abspath(hf_config_tokenizer_path)}",
                rank=self.rank,
                logger=logger,
                log_only_rank_0=True,
            )

            # If we have a custom model, we copy the file defining it in the folder and set the attributes so it can be
            # loaded from the Hub.
            if hasattr(model_config, "auto_map"):
                copied_remote_code = False
                if self.should_save_trainable_only and self.base_model_path:
                    source_dir = Path(self.base_model_path)
                    if source_dir.is_dir():
                        for source_file in source_dir.glob("*.py"):
                            shutil.copy2(source_file, Path(hf_config_tokenizer_path) / source_file.name)
                            copied_remote_code = True
                if not copied_remote_code:
                    custom_object_save(unwrap_model, hf_config_tokenizer_path, config=model_config)

            # Also save runtime FSDP config
            fsdp_config_path = os.path.join(local_path, "fsdp_config.json")
            fsdp_config = FSDPConfig(
                FSDP_version=fsdp_version(self.model),
                world_size=self.world_size,
            )
            with open(fsdp_config_path, "w") as f:
                json.dump(asdict(fsdp_config), f, indent=4)
            self._save_lora_train_meta(local_path, unwrap_model)
            if trainable_only_meta is not None:
                meta_path = self._trainable_only_meta_path(local_path)
                with open(meta_path, "w", encoding="utf-8") as handle:
                    json.dump(trainable_only_meta, handle, ensure_ascii=False, indent=2)
                log_with_rank(
                    f"Saved trainable-only metadata to {os.path.abspath(meta_path)}",
                    rank=self.rank,
                    logger=logger,
                    log_only_rank_0=True,
                )

        # wait for everyone to dump to local
        torch.distributed.barrier()

        if self.should_save_hf_model:
            # Only rank 0 will save hf model and,
            # offload to cpu to save LLMs which may be too large to fit in one GPU
            state_dict = get_fsdp_full_state_dict(self.model, offload_to_cpu=True, rank0_only=True)

            if self.rank == 0:
                hf_local_path = os.path.join(local_path, "huggingface")
                os.makedirs(hf_local_path, exist_ok=True)

                if "ForTokenClassification" in model_config.architectures[0]:
                    from transformers import AutoModelForTokenClassification

                    auto_model_cls = AutoModelForTokenClassification
                elif "ForCausalLM" in model_config.architectures[0]:
                    from transformers import AutoModelForCausalLM

                    auto_model_cls = AutoModelForCausalLM
                elif "ForConditionalGeneration" in model_config.architectures[0]:
                    auto_model_cls = get_auto_model_for_vision2seq()
                else:
                    raise NotImplementedError(f"Unknown architecture {model_config['architectures']}")

                with init_empty_weights():
                    save_model = auto_model_cls.from_config(
                        model_config, torch_dtype=torch.bfloat16, trust_remote_code=self.trust_remote_code
                    )

                save_model.to_empty(device="cpu")

                if save_model.can_generate():
                    if generation_config is not None:
                        save_model.generation_config = generation_config
                    else:
                        print(
                            f"Warning: {self.__class__.__name__}.save_checkpoint: Generation config file not found "
                            f"in, using a generation config created from the model config when saving hf_model."
                        )

                drop_tied_target_keys(state_dict, save_model, model_config)

                save_model.save_pretrained(hf_local_path, state_dict=state_dict)
                log_with_rank(
                    f"Saved hf_model to {os.path.abspath(hf_local_path)}",
                    rank=self.rank,
                    logger=logger,
                    log_only_rank_0=True,
                )
                del state_dict
                del save_model

            # wait for rank0 to dump hf_model to local
            torch.distributed.barrier()

        if self.rank == 0:
            self.register_checkpoint(local_path, max_ckpt_to_keep)
