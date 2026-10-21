"""Compare the SGLang adapter's first-token log probabilities with HF."""
import json
import os
from pathlib import Path
import time
import urllib.request

import torch
import torch.nn.functional as F
from transformers import AutoConfig, AutoTokenizer
from prefix_tuning.virtual_prefix.modeling_qwen3_5_hybrid_delta_residual_attention_prefix import (
    Qwen3_5HybridDeltaResidualAttentionPrefixForConditionalGeneration,
)

ROOT = Path('/home/liyu.468/learning_kvcache')
OUT = ROOT / 'diagnostics/b300_taco_compare_20260929'
MODEL = ROOT / 'outputs/taco_partial_pool32_retrain_20260928_1830/prefixes/taco'
tokenizer = AutoTokenizer.from_pretrained(MODEL, trust_remote_code=True)
config = AutoConfig.from_pretrained(MODEL, trust_remote_code=True)
config._attn_implementation = 'eager'
model = Qwen3_5HybridDeltaResidualAttentionPrefixForConditionalGeneration.from_pretrained(
    MODEL, config=config, dtype=torch.bfloat16, attn_implementation='eager',
).eval().to('cuda')
def torch_conv(x, weight, bias=None, activation=None, **kwargs):
    if kwargs.get('seq_idx') is not None:
        raise ValueError('The reference check uses unpacked sequences only')
    result = F.conv1d(x, weight.unsqueeze(1), bias=bias,
                      padding=weight.shape[-1] - 1, groups=x.shape[1])[..., :x.shape[-1]]
    return F.silu(result) if activation in ('silu', 'swish') else result

for module in model.modules():
    if hasattr(module, 'causal_conv1d_fn') and module.causal_conv1d_fn is None:
        module.causal_conv1d_fn = torch_conv
results = []
opener = urllib.request.build_opener(urllib.request.ProxyHandler({}))
deadline = time.monotonic() + 900
while True:
    try:
        with opener.open('http://127.0.0.1:18340/v1/models', timeout=3) as response:
            json.load(response)
        break
    except OSError:
        if time.monotonic() >= deadline:
            raise TimeoutError('SGLang service did not become ready')
        time.sleep(3)
for prompt in ('Reply with OK.', 'Write a Python function to add two integers.'):
    ids = tokenizer.apply_chat_template([{'role': 'user', 'content': prompt}],
                                        add_generation_prompt=True,
                                        enable_thinking=False, return_tensors='pt').to('cuda')
    if not isinstance(ids, torch.Tensor):
        ids = ids['input_ids']
    with torch.no_grad():
        logits = model(input_ids=ids, use_cache=False).logits[0, -1].float()
    hf_logp = logits.log_softmax(-1)
    payload = {'input_ids': ids[0].tolist(), 'sampling_params': {'temperature': 0, 'max_new_tokens': 4},
               'return_logprob': True, 'top_logprobs_num': 10, 'logprob_start_len': -1}
    request = urllib.request.Request('http://127.0.0.1:18340/generate',
        data=json.dumps(payload).encode(), headers={'Content-Type': 'application/json'})
    with opener.open(request, timeout=120) as response:
        sgl = json.load(response)
    top = sgl['meta_info']['output_top_logprobs'][0]
    errors = [abs(float(hf_logp[item[1]]) - item[0]) for item in top]
    probability_errors = [abs(float(hf_logp[item[1]].exp()) - float(torch.tensor(item[0]).exp())) for item in top]
    generated = [item[1] for item in sgl['meta_info']['output_token_logprobs']]
    decode = []
    for index in range(1, len(generated)):
        previous = torch.tensor([generated[:index]], device='cuda')
        with torch.no_grad():
            continuation = model(input_ids=torch.cat((ids, previous), -1), use_cache=False).logits[0, -1].float()
        top_i = sgl['meta_info']['output_top_logprobs'][index]
        lp_i = continuation.log_softmax(-1)
        decode.append({'hf_argmax': continuation.argmax().item(), 'sglang_argmax': generated[index],
                       'max_top10_logprob_delta': max(abs(float(lp_i[item[1]]) - item[0]) for item in top_i),
                       'max_top10_probability_delta': max(abs(float(lp_i[item[1]].exp()) - float(torch.tensor(item[0]).exp())) for item in top_i)})
    results.append({'prompt': prompt, 'hf_argmax': logits.argmax().item(),
                    'sglang_argmax': sgl['meta_info']['output_token_logprobs'][0][1],
                    'max_top10_logprob_delta': max(errors),
                    'max_top10_probability_delta': max(probability_errors), 'top10': top, 'decode': decode})
    (OUT / 'hf_parity.json').write_text(json.dumps(results, indent=2))
    print(results[-1], flush=True)
checked = results + [item for result in results for item in result['decode']]
# Different BF16 GEMM/fused-normalization paths can move logits by one ULP
# (0.25 near a logit of 32). Check both log probabilities and absolute token
# probabilities, and require exact agreement of every greedy generated token.
if any(r['hf_argmax'] != r['sglang_argmax'] or r['max_top10_logprob_delta'] > 0.30
       or r['max_top10_probability_delta'] > 0.03 for r in checked):
    raise RuntimeError('SGLang/HF first-token parity failed')
