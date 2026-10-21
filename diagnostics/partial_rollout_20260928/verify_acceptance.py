"""Verify functional acceptance using the actual H200 runs and checkpoints."""
import json
import math
from pathlib import Path

import torch

exp = Path('/home/liyu.468/learning_kvcache/outputs/partial_rollout_h200_20260928')
metrics = json.loads((exp / 'smoke_metrics.json').read_text())
prefixes = json.loads((exp / 'prefix_verification.json').read_text())
runs = {}
for name, expected_steps in [('taco_partial_smoke_v3', [1, 2, 3]),
                             ('envscaler_partial_3actor1teacher_smoke', [1, 2, 3]),
                             ('envscaler_resume_step4', [4])]:
    data = metrics[name]
    assert data['completed'], name
    assert [int(row['step']) for row in data['metrics']] == expected_steps, name
    for row in data['metrics']:
        assert math.isfinite(row['actor/distillation/loss']), name
        assert math.isfinite(row['actor/grad_norm']) and row['actor/grad_norm'] > 0, name
    changes = prefixes[name]['changes']
    assert sum(item['tensor_count'] for item in changes.values()) == 112
    assert all(item['changed_tensor_count'] == item['tensor_count'] for item in changes.values())
    runs[name] = {
        'steps': expected_steps,
        'max_version_span': max(row['training/off_policy/trajectory_spans/max'] for row in data['metrics']),
        'max_turns': max(row['training/num_turns/max'] for row in data['metrics']),
        'prefix_tensors_changed': 112,
        'checkpoint_world_size': prefixes[name]['manifest']['world_size'],
    }
assert runs['taco_partial_smoke_v3']['max_version_span'] >= 2
assert runs['envscaler_partial_3actor1teacher_smoke']['max_turns'] > 2
assert runs['envscaler_resume_step4']['checkpoint_world_size'] == 3
checkpoint_root = exp / 'checkpoints/envscaler_partial_3actor1teacher_smoke'
before = torch.load(checkpoint_root / 'global_step_3/data.pt', weights_only=False)
after = torch.load(checkpoint_root / 'global_step_4/data.pt', weights_only=False)
samples_before = before['_sampler_iter_state']['samples_yielded']
samples_after = after['_sampler_iter_state']['samples_yielded']
assert samples_after - samples_before == 6, (samples_before, samples_after)
assert after['_num_yielded'] - before['_num_yielded'] == 1
result = {'status': 'passed', 'runs': runs,
          'resume_admitted_prompts': samples_after - samples_before,
          'resume_training_batch_size': 6,
          'queue_checkpoint_supported': False,
          'scope': 'Functional smoke; no matched throughput comparison'}
(exp / 'acceptance.json').write_text(json.dumps(result, indent=2) + '\n')
print(json.dumps(result, indent=2))
