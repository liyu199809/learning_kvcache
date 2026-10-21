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
import logging
import os

import transfer_queue as tq

from verl.trainer.ppo.v1.trainer_base import PPOTrainer, register_trainer
from verl.utils.debug import marked_timer
from verl.workers.rollout.llm_server import FullyAsyncLLMServerClient

logger = logging.getLogger(__name__)
logger.setLevel(os.getenv("VERL_LOGGING_LEVEL", "INFO"))


@register_trainer("colocate_async")
class PPOTrainerColocateAsync(PPOTrainer):
    """Asynchronous PPO trainer
    1. Trainer and rollout are colocated.
    2. Partial rollout is enabled.
    """

    def get_llm_client(self):
        """Get the LLM server client for rollout generation."""
        return self.llm_server_manager.get_client(client_cls=FullyAsyncLLMServerClient)

    def on_init_end(self):
        # update weights after loading checkpoint
        self.checkpoint_manager.update_weights(self.global_steps)

    def on_train_begin(self):
        if self.config.skip.rollout_tq.enable:
            return
        self._rollout_target_size = self.config.trainer.v1.colocate_async.get("rollout_target_size")
        if self._rollout_target_size is not None:
            if not isinstance(self._rollout_target_size, int) or self._rollout_target_size < self.config.data.train_batch_size:
                raise ValueError("rollout_target_size must be an integer >= data.train_batch_size")
            remaining_updates = max(0, self.total_training_steps - self.global_steps + 1)
            self._remaining_feed_prompts = max(
                0, remaining_updates * self.config.data.train_batch_size - self._queued_prompt_groups()
            )
            logger.info(
                f"Bounded rollout pool: target={self._rollout_target_size}, "
                f"train_batch={self.config.data.train_batch_size}, admission_budget={self._remaining_feed_prompts}"
            )
            return
        # Warmup brings prompts forward; it must not add extra prompts beyond
        # the requested updates. Account for checkpointed groups and stop feeding
        # new batches near the end, so the last updates drain the existing tails.
        items = (tq.kv_list("train") or {}).get("train", {})
        existing_groups = sum(tag.get("is_prompt", False) for tag in items.values())
        remaining_updates = max(0, self.total_training_steps - self.global_steps + 1)
        self._remaining_feed_batches = max(
            0, remaining_updates - existing_groups // self.config.data.train_batch_size
        )
        num_warmup_batches = min(
            self.config.trainer.v1.colocate_async.num_warmup_batches, self._remaining_feed_batches
        )
        for _ in range(num_warmup_batches):
            self._add_batch_to_generate()
        logger.info(f"Added {num_warmup_batches} warmup batches to the agent loop manager")

    def _add_batch_to_generate(self):
        if getattr(self, "_rollout_target_size", None) is not None:
            self._last_queued_before = self._queued_prompt_groups()
            self._last_feed_count = min(
                max(0, self._rollout_target_size - self._last_queued_before), self._remaining_feed_prompts
            )
            if self._last_feed_count:
                submitted = self._add_prompts_to_generate(self._last_feed_count)
                self._remaining_feed_prompts -= submitted
                self._last_feed_count = submitted
            return
        remaining = getattr(self, "_remaining_feed_batches", None)
        if remaining is not None and remaining <= 0:
            return
        super()._add_batch_to_generate()
        if remaining is not None:
            self._remaining_feed_batches -= 1

    def _queued_prompt_groups(self):
        items = (tq.kv_list("train") or {}).get("train", {})
        return sum(tag.get("is_prompt", False) for tag in items.values())

    def step(self, metrics, timing_raw):
        batch = super().step(metrics, timing_raw)
        if getattr(self, "_rollout_target_size", None) is not None:
            queued = self._queued_prompt_groups()
            metrics.update({
                "training/partial/rollout_target": self._rollout_target_size,
                "training/partial/submitted_prompts": self._last_feed_count,
                "training/partial/consumed_prompts": self._last_queued_before + self._last_feed_count - queued,
                "training/partial/queued_prompts": queued,
                "training/partial/remaining_admission_budget": self._remaining_feed_prompts,
            })
        return batch

    def on_step_end(self):
        with marked_timer("update_weights", self.timing_raw, color="red"):
            # wake up all replicas to update weights
            self.checkpoint_manager.update_weights(self.global_steps)
            # resume generation
            self.checkpoint_manager.resume_generation_replicas()

    def on_sample_end(self):
        # abort all unfinished requests and pause generation
        self.checkpoint_manager.abort_replicas()
        # sleep all replicas to discard weights and kv cache
        self.checkpoint_manager.sleep_replicas()
