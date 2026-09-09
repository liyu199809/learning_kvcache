#!/usr/bin/env python3
"""CPU audit of all TACO prompts and the real teacher-scoring call path.

Run with the project venv and PYTHONPATH=verl from the repository root.
Prints JSON; does not generate answers or start GPU workers.
"""
import argparse
import asyncio
import hashlib
import json
from pathlib import Path
from types import SimpleNamespace

import pyarrow.parquet as pq
import torch
from transformers import AutoTokenizer

from verl.trainer.distillation.privileged_context import (
    build_privileged_sequence,
    build_thinking_prompt_override,
)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", default="/mnt/storage/disk1/verl_data/base_model/Qwen3.5-4B")
    parser.add_argument("--data-root", default="traj_data/opsd_taco_condensed_v2")
    args = parser.parse_args()
    root = Path(args.data_root)
    tokenizer = AutoTokenizer.from_pretrained(args.model, trust_remote_code=True)
    override = build_thinking_prompt_override(tokenizer, {"enable_thinking": False}, True)
    prefix = tokenizer.encode("\n\n", add_special_tokens=False)
    marker = tokenizer.encode("<|im_end|>\n<|im_start|>assistant\n", add_special_tokens=False)
    response = tokenizer.encode("```python\nprint(1)\n```", add_special_tokens=False)
    train = pq.read_table(root / "sources/taco/train.parquet").to_pylist()
    assert train
    first = None
    for row in train:
        kwargs = dict(tokenize=True, add_generation_prompt=True, return_dict=False)
        student = tokenizer.apply_chat_template(row["prompt"], enable_thinking=False, **kwargs)
        on_prompt = tokenizer.apply_chat_template(row["prompt"], enable_thinking=True, **kwargs)
        assert student[: -len(override[0])] == on_prompt[: -len(override[1])]
        assert "[Expert advice]" not in tokenizer.decode(student)
        solution = tokenizer.encode(row["reward_model"]["ground_truth"].strip(), add_special_tokens=False)
        expected = build_privileged_sequence(on_prompt, response, solution, prefix, [], marker)
        snapshot = student[:]
        actual = build_privileged_sequence(student, response, solution, prefix, [], marker, override)
        assert actual == expected
        assert actual[-len(response):] == response
        assert student == snapshot
        # The entire advice block must sit inside the last user turn.
        teacher_prompt = tokenizer.decode(actual[:-len(response)])
        user_end = teacher_prompt.rfind("<|im_end|>\n<|im_start|>assistant\n")
        assert teacher_prompt.rfind("[Expert advice]") < user_end
        if first is None:
            first = row, student, expected

    # Exercise the production method, including response logit realignment.
    from verl.experimental.agent_loop.agent_loop import AgentLoopWorker

    calls = []

    class Recorder:
        async def compute_teacher_logprobs_single(self, sequence_ids, **kwargs):
            calls.append(sequence_ids)
            n = len(sequence_ids)
            ids = torch.arange(n * 2).reshape(n, 2)
            return ids, ids.float()

    worker = object.__new__(AgentLoopWorker)
    worker.distillation_enabled = True
    worker.self_distillation = True
    worker.teacher_key = "data_source"
    worker.privileged_solution_key = "reward_model.ground_truth"
    worker.privileged_mode = "append"
    worker.tokenizer = tokenizer
    worker._privileged_prefix_ids = prefix
    worker._privileged_suffix_ids = []
    worker._privileged_insert_before_ids = marker
    worker._privileged_prompt_suffix_override = override
    worker.teacher_server_manager = Recorder()
    row, student, expected = first
    output = SimpleNamespace(extra_fields={}, multi_modal_data=None, mm_processor_kwargs={})
    asyncio.run(worker._compute_teacher_logprobs(output, student, response, False, row))
    assert calls == [expected]
    aligned = output.extra_fields["teacher_ids"]
    for j in range(len(response)):
        teacher_position = len(expected) - len(response) - 1 + j
        assert aligned[len(student) - 1 + j, 0].item() == teacher_position * 2
    # Validation must not use privileged context or invoke the teacher.
    asyncio.run(worker._compute_teacher_logprobs(output, student, response, True, {}))
    assert len(calls) == 1
    print(json.dumps({
        "status": "passed",
        "train_rows_checked": len(train),
        "student_opener": tokenizer.decode(override[0]),
        "teacher_opener": tokenizer.decode(override[1]),
        "student_opener_ids": override[0],
        "teacher_opener_ids": override[1],
        "system_and_user_tokens_unchanged": True,
        "student_response_tokens_unchanged": True,
        "teacher_response_alignment": "passed",
        "validation_does_not_call_teacher": True,
        "files_sha256": {
            str(p): hashlib.sha256(p.read_bytes()).hexdigest()
            for p in (root / "sources/taco/train.parquet", root / "val.parquet")
        },
    }, indent=2))


if __name__ == "__main__":
    main()
