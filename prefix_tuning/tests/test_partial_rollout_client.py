"""Exercise the abort/resume contract used by hybrid-prefix OPSD."""
import asyncio
from unittest.mock import AsyncMock

import pytest
from omegaconf import OmegaConf

from verl.workers.rollout.llm_server import FullyAsyncLLMServerClient, LLMServerClient
from verl.workers.rollout.replica import TokenOutput


def test_resume_keeps_tokens_logprobs_and_remaining_budget(monkeypatch):
    replies = iter([
        TokenOutput(token_ids=[10, 11], log_probs=[-.2, -.3], stop_reason="aborted",
                    extra_fields={"global_steps": 1}),
        TokenOutput(token_ids=[12], log_probs=[-.4], stop_reason="stop",
                    extra_fields={"global_steps": 2}),
    ])
    requests = []

    async def generate(self, request_id, **kwargs):
        requests.append((request_id, list(kwargs["prompt_ids"]), dict(kwargs["sampling_params"])))
        return next(replies)

    monkeypatch.setattr(LLMServerClient, "generate", generate)
    monkeypatch.setattr(asyncio, "sleep", AsyncMock())
    client = object.__new__(FullyAsyncLLMServerClient)
    client.config = OmegaConf.create({"trainer": {"use_v1": True}})
    out = asyncio.run(client.generate("episode-1", prompt_ids=[1, 2],
                                    sampling_params={"max_tokens": 5}))
    assert out.token_ids == [10, 11, 12]
    assert out.log_probs == [-.2, -.3, -.4]
    assert requests == [("episode-1", [1, 2], {"max_tokens": 5}),
                        ("episode-1", [1, 2, 10, 11], {"max_tokens": 3})]
    assert out.extra_fields["min_global_steps"] == 1
    assert out.extra_fields["max_global_steps"] == 2
    assert out.stop_reason == "stop"


def test_resume_stops_at_original_token_limit(monkeypatch):
    calls = []

    async def generate(self, request_id, **kwargs):
        calls.append(kwargs["sampling_params"]["max_new_tokens"])
        return TokenOutput(token_ids=[10, 11], log_probs=[-.2, -.3],
                           stop_reason="aborted", extra_fields={"global_steps": 1})

    monkeypatch.setattr(LLMServerClient, "generate", generate)
    client = object.__new__(FullyAsyncLLMServerClient)
    client.config = OmegaConf.create({"trainer": {"use_v1": True}})
    out = asyncio.run(client.generate("episode-2", prompt_ids=[1],
                                    sampling_params={"max_new_tokens": 2}))
    assert calls == [2]
    assert out.token_ids == [10, 11]
    assert out.stop_reason == "length"


def test_wait_strategy_allows_a_tail_until_version_limit():
    from verl.trainer.ppo.v1.replay_buffer import ReplayBuffer

    buffer = ReplayBuffer("colocate_async", OmegaConf.create({}), 8, "wait", OmegaConf.create({}))
    buffer.running_keys["train"].add("long")
    buffer.prompt_global_steps["train"]["long"] = 1
    buffer.finished_keys["train"].update({"ready-a", "ready-b"})
    buffer.prompt_global_steps["train"].update({"ready-a": 1, "ready-b": 2})
    assert buffer._has_enough_samples(2, "train", 2)
    assert not buffer._has_enough_samples(8, "train", 2)
    buffer.running_keys["train"].clear()
    buffer.finished_keys["train"].add("long")
    assert buffer._has_enough_samples(8, "train", 2)
    assert "long" in buffer._sampleable_terminal_keys(8, "train")


def test_server_holds_retries_across_staged_wake_up():
    from types import SimpleNamespace
    from verl.workers.rollout.replica import RolloutMode
    from verl.workers.rollout.vllm_rollout.vllm_async_server import vLLMHttpServer

    impl = getattr(getattr(vLLMHttpServer, "__ray_metadata__", None), "modified_class", vLLMHttpServer)

    async def scenario():
        server = object.__new__(impl)
        server._generation_ready = asyncio.Event()
        server._generation_ready.set()
        server.node_rank = 0
        server.replica_rank = 0
        server.rollout_mode = RolloutMode.HYBRID
        server._disaggregation_role = "null"
        server.global_steps = 1
        server.config = OmegaConf.create({"max_model_len": 32, "full_determinism": False})
        server.model_config = SimpleNamespace(processor=None, lora_rank=0, lora={})
        state = {"weights": False, "kv_cache": False, "entered": False}

        async def wake_up(tags):
            for tag in tags:
                state[tag] = True
            # Current vLLM also resumes its scheduler here, before staged sync ends.

        async def generate(**kwargs):
            assert state["weights"] and state["kv_cache"]
            state["entered"] = True
            yield SimpleNamespace(outputs=[])

        server.engine = SimpleNamespace(
            output_processor=SimpleNamespace(request_states={}),
            pause_generation=AsyncMock(), resume_generation=AsyncMock(),
            wake_up=wake_up, reset_prefix_cache=AsyncMock(), generate=generate,
        )
        await server.abort_all_requests()
        retry = asyncio.create_task(server.generate([1], {"max_tokens": 1}, "tail"))
        await asyncio.sleep(0)
        await server.wake_up(tags=["weights"])
        await asyncio.sleep(0)
        assert not state["entered"]
        assert not retry.done()
        await server.wake_up(tags=["kv_cache"])
        server.global_steps = 2
        await server.resume_generation()
        out = await asyncio.wait_for(retry, timeout=1)
        assert state["entered"]
        assert out.extra_fields["global_steps"] == 2

    asyncio.run(scenario())


def test_server_pause_failure_stops_the_update():
    from types import SimpleNamespace
    import pytest
    from verl.workers.rollout.vllm_rollout.vllm_async_server import vLLMHttpServer

    impl = getattr(getattr(vLLMHttpServer, "__ray_metadata__", None), "modified_class", vLLMHttpServer)

    async def scenario():
        server = object.__new__(impl)
        server._generation_ready = asyncio.Event()
        server._generation_ready.set()
        server.engine = SimpleNamespace(
            output_processor=SimpleNamespace(request_states={}),
            pause_generation=AsyncMock(side_effect=RuntimeError("pause failed")),
        )
        with pytest.raises(RuntimeError, match="pause failed"):
            await server.abort_all_requests()
        assert not server._generation_ready.is_set()

    asyncio.run(scenario())


@pytest.mark.parametrize("max_model_len,expected_budgets", [(32, [5, 3]), (5, [3, 1])])
def test_resume_uses_default_request_budget_without_mutating_caller(monkeypatch, max_model_len, expected_budgets):
    replies = iter([
        TokenOutput(token_ids=[10, 11], log_probs=[-.2, -.3], stop_reason="aborted",
                    extra_fields={"global_steps": 1}),
        TokenOutput(token_ids=[12], log_probs=[-.4], stop_reason="stop",
                    extra_fields={"global_steps": 2}),
    ])
    budgets = []

    async def generate(self, request_id, **kwargs):
        budgets.append(kwargs["sampling_params"]["max_tokens"])
        return next(replies)

    monkeypatch.setattr(LLMServerClient, "generate", generate)
    monkeypatch.setattr(asyncio, "sleep", AsyncMock())
    client = object.__new__(FullyAsyncLLMServerClient)
    client.config = OmegaConf.create({"trainer": {"use_v1": True}, "actor_rollout_ref": {
        "rollout": {"response_length": 5, "prompt_length": 10, "max_model_len": max_model_len}}})
    sampling_params = {"temperature": 1.0}
    out = asyncio.run(client.generate("default-budget", prompt_ids=[1, 2],
                                    sampling_params=sampling_params))
    assert budgets == expected_budgets
    assert out.token_ids == [10, 11, 12]
    assert sampling_params == {"temperature": 1.0}


@pytest.mark.parametrize("start_step,existing_groups,expected_new_groups", [(1, 0, 6), (2, 2, 2)])
def test_colocate_warmup_is_drained_within_requested_updates(monkeypatch, start_step, existing_groups, expected_new_groups):
    from verl.trainer.ppo.v1 import trainer_colocate_async as module
    from verl.trainer.ppo.v1.trainer_base import PPOTrainer

    trainer = object.__new__(module.PPOTrainerColocateAsync)
    trainer.config = OmegaConf.create({"skip": {"rollout_tq": {"enable": False}},
        "data": {"train_batch_size": 2},
        "trainer": {"v1": {"colocate_async": {"num_warmup_batches": 1}}}})
    trainer.total_training_steps = 3
    trainer.global_steps = start_step
    submitted = []
    items = {str(i): {"is_prompt": True, "status": "running"} for i in range(existing_groups)}
    monkeypatch.setattr(module.tq, "kv_list", lambda partition: {partition: items})
    monkeypatch.setattr(PPOTrainer, "_add_batch_to_generate", lambda self: submitted.extend([None, None]))
    trainer.on_train_begin()
    for _ in range(3 - start_step + 1):
        trainer._add_batch_to_generate()
    assert len(submitted) == expected_new_groups
    assert len(submitted) + existing_groups == (3 - start_step + 1) * 2
    assert trainer._remaining_feed_batches == 0


@pytest.mark.parametrize("target,existing,steps,expected_feeds", [
    (32, 0, 4, [32, 24, 24, 16]),
    (36, 0, 4, [36, 24, 24, 12]),
    (32, 0, 1, [24]),
    (32, 8, 2, [24, 16]),
    (32, 5, 2, [27, 16]),
    (32, 48, 3, [0, 8, 16]),
])
def test_bounded_pool_trains_24_and_drains_without_accumulation(monkeypatch, target, existing, steps, expected_feeds):
    from types import MethodType, SimpleNamespace
    from verl.trainer.ppo.v1 import trainer_colocate_async as module

    trainer = object.__new__(module.PPOTrainerColocateAsync)
    trainer.config = OmegaConf.create({"skip": {"rollout_tq": {"enable": False}},
        "data": {"train_batch_size": 24},
        "trainer": {"v1": {"colocate_async": {"num_warmup_batches": 1, "rollout_target_size": target}}}})
    trainer.total_training_steps = steps
    trainer.global_steps = 1
    trainer.parameter_sync_step = 1
    items = {f"restored-{i}": {"is_prompt": True} for i in range(existing)}
    # Trajectory keys do not count against the pool of prompt groups.
    items["trajectory"] = {"is_prompt": False}
    submitted = []
    counter = iter(range(1000))
    monkeypatch.setattr(module.tq, "kv_list", lambda partition: {partition: dict(items)})

    def add(self, count):
        submitted.extend(range(count))
        for _ in range(count):
            items[f"new-{next(counter)}"] = {"is_prompt": True}
        return count

    def train_once(self, metrics, timing, batch_size):
        assert batch_size == 24
        # Consume recently completed samples first to leave initial long tails
        # in the pool until the final drain, exercising out-of-order completion.
        keys = [k for k, tag in items.items() if tag.get("is_prompt")][-batch_size:]
        assert len(keys) == 24
        for key in keys:
            del items[key]
        return SimpleNamespace(keys=keys, tags=[{} for _ in keys], partition_id="train")

    trainer._add_prompts_to_generate = MethodType(add, trainer)
    trainer._step_once = MethodType(train_once, trainer)
    trainer.on_train_begin()
    assert not submitted  # Pool admission is measured inside step/feed.
    for step, expected in enumerate(expected_feeds, 1):
        trainer.global_steps = step
        metrics = {}
        batch = trainer.step(metrics, {})
        assert len(batch.keys) == 24
        assert metrics["training/partial/submitted_prompts"] == expected
        assert metrics["training/partial/consumed_prompts"] == 24
        assert metrics["training/partial/queued_prompts"] <= max(target - 24, existing - 24)
    assert trainer._queued_prompt_groups() == 0
    assert trainer._remaining_feed_prompts == 0
    assert len(submitted) + existing == steps * 24


def test_bounded_pool_can_coalesce_32_then_24_without_changing_train_batch():
    import torch
    from tensordict import TensorDict
    from verl.trainer.ppo.v1.trainer_colocate_async import PPOTrainerColocateAsync

    trainer = object.__new__(PPOTrainerColocateAsync)
    trainer.config = OmegaConf.create({"data": {"train_batch_size": 24, "gen_batch_size": 1}})
    trainer.global_steps = 1
    counter = iter(range(56))
    trainer._fetch_one_gen_batch = lambda: TensorDict({"index": torch.tensor([next(counter)])}, batch_size=[1])
    first = trainer._next_train_batch(32)
    refill = trainer._next_train_batch(24)
    assert len(first) == 32 and len(refill) == 24
    assert first["index"].tolist() == list(range(32))
    assert refill["index"].tolist() == list(range(32, 56))
    assert trainer.config.data.train_batch_size == 24


def test_bounded_pool_fetch_size_does_not_change_epoch_or_optimizer_steps(monkeypatch):
    from verl.trainer.ppo.v1 import trainer_base as base
    from verl.trainer.ppo.v1.trainer_colocate_async import PPOTrainerColocateAsync

    trainer = object.__new__(PPOTrainerColocateAsync)
    trainer.config = OmegaConf.create({
        "data": {"train_batch_size": 24, "gen_batch_size": 8, "train_files": [], "val_files": [],
                 "val_batch_size": 2, "dataloader_num_workers": 0},
        "trainer": {"total_epochs": 1, "total_training_steps": None,
                    "v1": {"colocate_async": {"rollout_target_size": 32},
                           "sampler": {"max_off_policy_strategy": "wait"}}}})
    trainer.trainer_mode = "colocate_async"
    trainer.parameter_sync_step = 1
    trainer.tokenizer = trainer.processor = None
    monkeypatch.setattr(base, "create_rl_dataset", lambda *args, is_train, **kwargs: [{}] * (96 if is_train else 2))
    monkeypatch.setattr(base, "create_rl_sampler", lambda *args: None)
    trainer._init_dataloader()
    assert trainer.train_dataloader.batch_size == 1
    assert trainer.config.data.train_batch_size == 24
    assert trainer.steps_per_epoch == 4 and trainer.total_training_steps == 4


def test_bounded_pool_rejects_target_smaller_than_training_batch():
    from verl.trainer.ppo.v1.trainer_colocate_async import PPOTrainerColocateAsync

    trainer = object.__new__(PPOTrainerColocateAsync)
    trainer.config = OmegaConf.create({"skip": {"rollout_tq": {"enable": False}},
        "data": {"train_batch_size": 24},
        "trainer": {"v1": {"colocate_async": {"rollout_target_size": 23}}}})
    with pytest.raises(ValueError, match="data.train_batch_size"):
        trainer.on_train_begin()
