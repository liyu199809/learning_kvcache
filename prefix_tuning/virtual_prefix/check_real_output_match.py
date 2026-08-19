#!/usr/bin/env python3
"""Real-output equivalence check: base Qwen3.5-4B vs DeltaVirtualPrefix-256.

Loads one of the two checkpoints with native vLLM, runs greedy chat
generation plus prompt-logprob prefill on a fixed prompt set, and dumps
the results to JSON. Compare the two JSONs afterwards.

Usage: MODEL=<path> OUT=<file> CUDA_VISIBLE_DEVICES=<gpu> python check_real_output_match.py
"""

import json
import os

from vllm import LLM, SamplingParams

BASE = "/mnt/storage/disk1/verl_data/base_model/Qwen3.5-4B"
PREFIX = "/mnt/storage/disk1/verl_data/base_model/Qwen3.5-4B-DeltaVirtualPrefix-256"

PROMPTS = [
    "请用一句话解释什么是线性注意力。",
    "Write a Python function that checks whether a string is a palindrome.",
    "简述牛顿第二定律，并给出一个日常生活中的例子。",
    "Solve: if 3x + 7 = 22, what is x? Show your steps briefly.",
]

GEN_PARAMS = SamplingParams(max_tokens=64, temperature=0.0, logprobs=1)
PREFILL_PARAMS = SamplingParams(max_tokens=1, temperature=0.0, prompt_logprobs=0, logprobs=1)


def main() -> None:
    model = os.environ["MODEL"]
    out_path = os.environ["OUT"]
    llm = LLM(
        model=model,
        model_impl="vllm",
        trust_remote_code=True,
        max_model_len=4096,
        gpu_memory_utilization=0.5,
        enforce_eager=os.getenv("ENFORCE_EAGER", "1") != "0",
    )

    result = {"model": model, "generation": [], "prefill_first_logprobs": []}

    msgs = [[{"role": "user", "content": p}] for p in PROMPTS]
    outputs = llm.chat(msgs, GEN_PARAMS)
    for prompt, out in zip(PROMPTS, outputs):
        result["generation"].append(
            {
                "prompt": prompt,
                "prompt_token_ids": list(out.prompt_token_ids),
                "output_token_ids": list(out.outputs[0].token_ids),
                "output_text": out.outputs[0].text,
                "top_logprobs": [
                    {int(tid): lp_.logprob for tid, lp_ in (lp or {}).items()}
                    for lp in out.outputs[0].logprobs[:5]
                ],
            }
        )

    # Prefill path: first sampled-token logprob carries the full-prompt forward.
    outputs = llm.chat(msgs, PREFILL_PARAMS)
    for prompt, out in zip(PROMPTS, outputs):
        plp = out.prompt_logprobs
        result["prefill_first_logprobs"].append(
            {
                "prompt": prompt,
                "n_prompt_tokens": len(out.prompt_token_ids),
                "last_prompt_logprob": (
                    next(iter(plp[-1].values())).logprob if plp and plp[-1] else None
                ),
                "first_token": out.outputs[0].token_ids[0],
                "first_token_logprob": next(iter(out.outputs[0].logprobs[0].values())).logprob,
            }
        )

    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(result, f, ensure_ascii=False, indent=2)
    print(f"wrote {out_path}")


if __name__ == "__main__":
    main()
