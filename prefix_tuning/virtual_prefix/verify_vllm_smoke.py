#!/usr/bin/env python3
"""One-request native vLLM smoke test for a virtual-prefix checkpoint."""

import os

from vllm import LLM, SamplingParams


def main() -> None:
    path = os.getenv(
        "MODEL",
        "/mnt/storage/disk1/verl_data/base_model/Qwen3.5-4B-DeltaVirtualPrefix-256",
    )
    enforce_eager = os.getenv("ENFORCE_EAGER", "1") != "0"
    llm = LLM(
        model=path,
        model_impl="vllm",
        trust_remote_code=True,
        max_model_len=256,
        gpu_memory_utilization=0.25,
        enforce_eager=enforce_eager,
    )
    output = llm.generate(
        ["请用一句话解释virtual prefix。"],
        SamplingParams(max_tokens=2, temperature=0),
    )[0]
    print(
        {
            "prompt_token_ids": output.prompt_token_ids,
            "output_token_ids": output.outputs[0].token_ids,
            "text": output.outputs[0].text,
        }
    )
    assert len(output.outputs[0].token_ids) == 2
    print("vllm_virtual_prefill_decode_ok", True)


if __name__ == "__main__":
    main()
