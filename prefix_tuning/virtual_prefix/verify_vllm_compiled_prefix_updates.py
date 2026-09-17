"""Test prefix-only weight reload after compiled CUDA graphs have been captured."""
import argparse
import functools
import json
from pathlib import Path
import time

import torch
from safetensors.torch import load_file
from vllm import LLM, SamplingParams

from prefix_tuning.virtual_prefix.verify_vllm_prefix_runtime import inspect_prefix


def reload_prefix(model, model_path):
    state = {}
    for filename in ('independent_delta_kv_prefix.safetensors',
                     'residual_attention_prefix.safetensors'):
        state.update(load_file(str(Path(model_path) / filename), device='cpu'))
    loaded = model.load_weights(iter(state.items()))
    mapped = dict(model.hf_to_vllm_mapper.apply(state.items()))
    assert set(mapped) <= loaded
    params = dict(model.named_parameters())
    for name, tensor in mapped.items():
        assert torch.equal(params[name].detach().cpu(), tensor), name
    return {'loaded_tensors': len(state), 'inventory': inspect_prefix(model)}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--model', required=True)
    parser.add_argument('--replacement', required=True)
    parser.add_argument('--reference', required=True)
    parser.add_argument('--output', required=True)
    args = parser.parse_args()
    target = Path(args.output)
    if target.exists():
        raise FileExistsError(target)
    reference = json.loads(Path(args.reference).read_text())
    prompts = [{'prompt_token_ids': row['prompt_token_ids']}
               for row in reference['requests']]
    llm = LLM(model=args.model, model_impl='vllm', trust_remote_code=True,
              dtype='bfloat16', max_model_len=4096, gpu_memory_utilization=.18,
              max_num_seqs=4, max_num_batched_tokens=256,
              enable_chunked_prefill=True, enable_prefix_caching=False,
              enforce_eager=False, limit_mm_per_prompt={'image': 0, 'video': 0}, seed=42)
    sampling = SamplingParams(max_tokens=48, temperature=0, logprobs=20)

    def generate():
        start = time.perf_counter()
        outputs = llm.generate(prompts, sampling, use_tqdm=False)
        return {'seconds': time.perf_counter() - start, 'requests': [
            {'token_ids': list(out.outputs[0].token_ids), 'text': out.outputs[0].text,
             'first_logprobs': {str(k): v.logprob
                                for k, v in out.outputs[0].logprobs[0].items()}}
            for out in outputs]}

    before = generate()
    warm_repeat = generate()
    reload_info = llm.apply_model(functools.partial(
        reload_prefix, model_path=str(Path(args.replacement).resolve())))
    llm.reset_prefix_cache()
    after = generate()
    after_repeat = generate()
    comparisons = []
    for actual, expected in zip(after['requests'], reference['requests']):
        shared = set(actual['first_logprobs']) & set(expected['first_logprobs'])
        comparisons.append({
            'tokens_equal_reference': actual['token_ids'] == expected['token_ids'],
            'shared_first_logprobs': len(shared),
            'max_first_logprob_error': max(abs(actual['first_logprobs'][k]
                                             - expected['first_logprobs'][k]) for k in shared),
        })
    report = {'model': args.model, 'replacement': args.replacement,
              'reference': args.reference, 'before': before, 'warm_repeat': warm_repeat,
              'reload': reload_info, 'after': after, 'after_repeat': after_repeat,
              'comparisons': comparisons, 'final_inventory': llm.apply_model(inspect_prefix)}
    target.write_text(json.dumps(report, ensure_ascii=False, indent=2) + '\n')
    assert all(torch.isfinite(torch.tensor(v)) for r in after['requests']
               for v in r['first_logprobs'].values())
    assert all(c['max_first_logprob_error'] < .01 for c in comparisons), comparisons
    assert all(c['tokens_equal_reference'] for c in comparisons), comparisons
    assert [r['token_ids'] for r in after['requests']] == [r['token_ids'] for r in after_repeat['requests']]
    print('COMPILED_PREFIX_RELOAD_OK', json.dumps(comparisons), flush=True)


if __name__ == '__main__':
    main()
