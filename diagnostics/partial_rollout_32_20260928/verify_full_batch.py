"""Check actual admission, training batch, drain, and prefix checkpoint."""
import json
import math
from pathlib import Path
import re

import torch
from safetensors.torch import load_file
from prefix_tuning.virtual_prefix.export_hybrid_delta_residual_attention_prefix_checkpoint import export

root = Path('/home/liyu.468/learning_kvcache')
exp = root / 'outputs/partial_rollout_32_h200_20260928'
run = 'envscaler_24train_32rollout'
content = re.sub(r'\x1b\[[0-9;]*m', '', (exp / 'launcher.log').read_text())
assert 'Final validation metrics:' in content, 'Run has not completed'
metrics = []
for line in content.splitlines():
    if re.search(r'\bstep:\s*\d+\s*-', line):
        metrics.append({key: float(value) for key, value in re.findall(
            r'([\w/.-]+):\s*(?:np\.\w+\()?([-+]?(?:\d+(?:\.\d*)?|\.\d+)(?:[eE][-+]?\d+)?|nan|inf)', line)})
assert [row['step'] for row in metrics] == [1, 2, 3]
for row, submitted, queued, budget in zip(metrics, [32, 24, 16], [8, 8, 0], [40, 16, 0], strict=True):
    assert row['training/partial/rollout_target'] == 32
    assert row['training/partial/submitted_prompts'] == submitted
    assert row['training/partial/consumed_prompts'] == 24
    assert row['training/partial/queued_prompts'] == queued
    assert row['training/partial/remaining_admission_budget'] == budget
    assert math.isfinite(row['actor/distillation/loss'])
    assert math.isfinite(row['actor/grad_norm']) and row['actor/grad_norm'] > 0
    records = [json.loads(line) for line in (exp / 'rollouts' / run / f"{int(row['step'])}.jsonl").read_text().splitlines()]
    assert len(records) == 24, (row['step'], len(records))
checkpoint = exp / 'checkpoints' / run / 'global_step_3'
state = torch.load(checkpoint / 'data.pt', weights_only=False)
if '_num_yielded' in state:
    yielded = state['_num_yielded']
else:
    from torchdata.stateful_dataloader.stateful_dataloader import _StatefulMultiProcessingDataLoaderIter as Iter
    yielded = state[Iter._SNAPSHOT][Iter._SNAPSHOT_STEP] + state[Iter._STEPS_SINCE_SNAPSHOT]
assert yielded == 72, yielded
prepared = root / 'outputs/prefix_composition_3domain_20260917/prepared/hybrid-G512-A64'
output = exp / 'prefixes' / run
manifest = (json.loads((output / 'prefix_merge_manifest.json').read_text()) if output.exists()
            else export(checkpoint / 'actor', prepared, output, torch.float32, 'hardlink'))
assert manifest['world_size'] == 3 and manifest['trainable_numel'] == 78905344
changes = {}
for filename in ['independent_delta_kv_prefix.safetensors', 'residual_attention_prefix.safetensors']:
    initial, trained = load_file(str(prepared / filename)), load_file(str(output / filename))
    assert set(initial) == set(trained)
    deltas = []
    for name, tensor in trained.items():
        assert torch.isfinite(tensor).all(), name
        delta = float((tensor.float() - initial[name].float()).abs().max())
        assert delta > 0, name
        deltas.append(delta)
    changes[filename] = {'changed_tensors': len(deltas), 'max_delta': max(deltas)}
new_average = sum(row['timing_s/step'] for row in metrics[1:]) / 2
historical_average = 260.2
result = {'status': 'passed', 'train_batch': 24, 'ppo_mini_batch': 24, 'rollout_target': 32,
          'response_limit': 16384, 'actor_gpus': 3, 'teacher_gpus': 1,
          'admissions': [32, 24, 16], 'consumed': [24, 24, 24], 'queued_after': [8, 8, 0],
          'dataloader_consumed_prompts': 72, 'metrics': metrics, 'prefix_changes': changes,
          'comparison': {'historical_mean_step_seconds': historical_average,
                         'current_last_two_mean_step_seconds': new_average,
                         'throughput_ratio': historical_average / new_average,
                         'step_time_reduction_percent': 100 * (1 - new_average / historical_average),
                         'scope': 'Historical different host; only two measured steps, including final drain'}}
(exp / 'acceptance.json').write_text(json.dumps(result, indent=2) + '\n')
print(json.dumps({key: value for key, value in result.items() if key != 'metrics'}, indent=2))
