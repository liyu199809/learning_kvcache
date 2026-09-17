"""Exercise real prefix loading, chunked prefill, decode and request reuse."""
import argparse
import json
from pathlib import Path

import torch
from vllm import LLM, SamplingParams


def inspect_prefix(model):
    suffixes = ('.prefix_k', '.prefix_v', '.prefix_beta_logits', '.prefix_a',
                '.prefix_key_tokens', '.prefix_value_tokens')
    params = {n: p for n, p in model.named_parameters() if n.endswith(suffixes)}
    return {
        'class': type(model).__name__,
        'tensors': len(params),
        'numel': sum(p.numel() for p in params.values()),
        'finite': all(bool(torch.isfinite(p).all()) for p in params.values()),
        'delta_cache_misses': [m._prefix_cache_misses for m in model.modules()
                               if hasattr(m, '_prefix_cache_misses')],
        'attention_cache_misses': [m._prefix_projection_cache_misses for m in model.modules()
                                   if hasattr(m, '_prefix_projection_cache_misses')],
    }


def compare_delta_states(model):
    """Check prefix-specific cache math independently of full-model BF16 rounding."""
    from prefix_tuning.virtual_prefix.modeling_qwen3_5_independent_delta_kv_prefix import (
        Qwen3_5IndependentDeltaKVPrefixGatedDeltaNet as HFDelta,
    )
    modules = [m for m in model.modules() if hasattr(m, '_cached_prefix_recurrent_state')]
    results = []
    if not modules:
        return results
    for native in (modules[0], modules[len(modules)//2], modules[-1]):
        with torch.inference_mode():
            reference = HFDelta(native.config, native.layer_idx).to(
                device=native.prefix_k.device, dtype=native.prefix_k.dtype)
            for name in ('prefix_k', 'prefix_v', 'prefix_beta_logits', 'prefix_a', 'A_log', 'dt_bias'):
                getattr(reference, name).copy_(getattr(native, name))
            reference.conv1d.weight.copy_(native.conv1d.weight)
            tail, _, state = reference._compute_prefix_states(native.prefix_k.dtype)
            actual_conv = native._cached_prefix_conv_state
            actual_state = native._cached_prefix_recurrent_state
            expected_state = state[0].transpose(-1, -2).float()
            conv_error = float((tail[0].float() - actual_conv.float()).abs().max())
            state_error = float((expected_state - actual_state.float()).abs().max())
            torch.testing.assert_close(tail[0].float(), actual_conv.float(), rtol=0, atol=0)
            torch.testing.assert_close(expected_state, actual_state.float(), rtol=0.03, atol=1e-6)
            results.append({'layer': native.layer_idx, 'conv_max_abs': conv_error,
                            'recurrent_max_abs': state_error})
            del reference
    return results


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--model', required=True)
    parser.add_argument('--output', required=True)
    parser.add_argument('--compiled', action='store_true')
    parser.add_argument('--max-model-len', type=int, default=4096)
    parser.add_argument('--check-states', action='store_true')
    args = parser.parse_args()
    output = Path(args.output)
    if output.exists():
        raise FileExistsError(output)
    llm = LLM(model=args.model, model_impl='vllm', trust_remote_code=True,
              dtype='bfloat16', max_model_len=args.max_model_len,
              gpu_memory_utilization=0.18, max_num_seqs=4,
              max_num_batched_tokens=256, enable_chunked_prefill=True,
              enable_prefix_caching=False, enforce_eager=not args.compiled,
              limit_mm_per_prompt={'image': 0, 'video': 0}, seed=42)
    tokenizer = llm.get_tokenizer()
    questions = ['Return only the result of 17 + 25.',
                 'Write a Python function add(a, b) that returns their sum.',
                 '请简短解释什么是缓存。',
                 'Context: ' + 'This is a prefix cache isolation check. ' * 45
                 + '\nReturn only the word DONE.']
    prompts = [tokenizer.apply_chat_template(
        [{'role': 'user', 'content': q}], tokenize=False,
        add_generation_prompt=True, enable_thinking=False) for q in questions]
    sampling = SamplingParams(max_tokens=48, temperature=0, logprobs=20)
    first = llm.generate(prompts, sampling, use_tqdm=False)
    repeat = llm.generate(prompts, sampling, use_tqdm=False)
    rows = []
    for a, b in zip(first, repeat):
        result = a.outputs[0]
        assert result.token_ids, 'Empty generation'
        assert all(torch.isfinite(torch.tensor(v.logprob))
                   for step in result.logprobs for v in step.values())
        rows.append({'prompt_token_ids': a.prompt_token_ids,
                     'token_ids': list(result.token_ids), 'text': result.text,
                     'repeat_token_ids': list(b.outputs[0].token_ids),
                     'repeat_text': b.outputs[0].text,
                     'repeat_first_logprobs': {str(k): v.logprob for k, v in b.outputs[0].logprobs[0].items()},
                     'finish_reason': result.finish_reason,
                     'first_logprobs': {str(k): v.logprob for k, v in result.logprobs[0].items()}})
    inventory = llm.apply_model(inspect_prefix)
    state_checks = llm.apply_model(compare_delta_states) if args.check_states else None
    config = json.loads((Path(args.model) / 'config.json').read_text())
    if config.get('hybrid_prefix'):
        for item in inventory:
            assert item['tensors'] == 112, item
            assert item['numel'] == config['hybrid_prefix']['trainable_numel'], item
            assert len(item['delta_cache_misses']) == 24, item
            assert len(item['attention_cache_misses']) == 8, item
            assert all(n >= 1 for n in item['delta_cache_misses']), item
            assert item['finite'], item
    report = {'model': args.model, 'compiled': args.compiled,
              'max_model_len': args.max_model_len,
              'state_checks': state_checks,
              'inventory': inventory, 'requests': rows,
              'request_reuse_equal': all(r['token_ids'] == r['repeat_token_ids'] for r in rows)}
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(report, ensure_ascii=False, indent=2) + '\n')
    print('PREFIX_RUNTIME_OK', json.dumps(inventory), flush=True)
    for row in rows:
        print('GENERATED', repr(row['text']), flush=True)
        if row['token_ids'] != row['repeat_token_ids']:
            print('REPEAT_DIFFERED', repr(row['repeat_text']), flush=True)
    assert report['request_reuse_equal'], 'Request reuse changed greedy generation; inspect saved report'


if __name__ == '__main__':
    main()
