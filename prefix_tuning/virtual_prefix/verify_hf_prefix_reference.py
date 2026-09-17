"""Compare real vLLM generation with the Transformers prefix implementation."""
import argparse
import json
from pathlib import Path

import torch
from transformers import AutoModelForImageTextToText


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--model', required=True)
    p.add_argument('--vllm-report', required=True)
    p.add_argument('--output', required=True)
    args = p.parse_args()
    report = json.loads(Path(args.vllm_report).read_text())
    model = AutoModelForImageTextToText.from_pretrained(
        args.model, dtype=torch.bfloat16, trust_remote_code=True,
        attn_implementation='eager').cuda().eval()
    rows = []
    with torch.inference_mode():
        for row in report['requests']:
            ids = torch.tensor([row['prompt_token_ids'] + row['token_ids'][:-1]], device='cuda')
            n = len(row['token_ids'])
            logits = model(input_ids=ids, use_cache=False, logits_to_keep=n).logits[0].float()
            assert logits.shape[0] == n
            lp = logits[0].log_softmax(-1)
            errors = [abs(float(lp[int(k)]) - v) for k, v in row['first_logprobs'].items()]
            expected = torch.tensor(row['token_ids'], device='cuda')
            predicted = logits.argmax(-1)
            gaps = logits.max(-1).values - logits.gather(-1, expected[:, None]).squeeze(-1)
            result = {'tokens': n, 'greedy_mismatches': int((expected != predicted).sum()),
                      'max_expected_token_logit_gap': float(gaps.max()),
                      'first_logprob_max_abs': max(errors),
                      'first_logprob_mean_abs': sum(errors) / len(errors),
                      'finite': bool(torch.isfinite(logits).all())}
            rows.append(result)
            print('HF_REFERENCE', result, flush=True)
    Path(args.output).write_text(json.dumps({'model': args.model, 'requests': rows}, indent=2) + '\n')
    assert all(r['finite'] and r['first_logprob_max_abs'] < 0.15
               and r['max_expected_token_logit_gap'] < 0.15 for r in rows), rows
    print('HF_VLLM_REFERENCE_OK', flush=True)


if __name__ == '__main__':
    main()
