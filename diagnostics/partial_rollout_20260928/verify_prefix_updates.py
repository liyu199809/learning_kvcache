"""CPU-only export and prefix-change verification for real smoke checkpoints."""
import json
from pathlib import Path

import torch
from safetensors.torch import load_file
from prefix_tuning.virtual_prefix.export_hybrid_delta_residual_attention_prefix_checkpoint import export

root = Path('/home/liyu.468/learning_kvcache')
exp = root / 'outputs/partial_rollout_h200_20260928'
prepared = root / 'outputs/prefix_composition_3domain_20260917/prepared/hybrid-G512-A64'
summary = {}
for run, step, export_name in [('taco_partial_smoke_v3', 3, 'taco_partial_smoke_v3'),
                               ('envscaler_partial_3actor1teacher_smoke', 3, 'envscaler_partial_3actor1teacher_smoke'),
                               ('envscaler_partial_3actor1teacher_smoke', 4, 'envscaler_resume_step4')]:
    checkpoint = exp / 'checkpoints' / run / f'global_step_{step}' / 'actor'
    output = exp / 'prefixes' / export_name
    manifest = (json.loads((output / 'prefix_merge_manifest.json').read_text())
                if output.exists() else export(checkpoint, prepared, output, torch.float32, 'hardlink'))
    changed = {}
    for filename in ['independent_delta_kv_prefix.safetensors', 'residual_attention_prefix.safetensors']:
        initial, trained = load_file(str(prepared / filename)), load_file(str(output / filename))
        assert set(initial) == set(trained)
        deltas = []
        for name, tensor in trained.items():
            assert torch.isfinite(tensor).all(), name
            deltas.append(float((tensor.float() - initial[name].float()).abs().max()))
        changed[filename] = {'tensor_count': len(deltas), 'changed_tensor_count': sum(d > 0 for d in deltas), 'max_abs_change': max(deltas)}
        assert changed[filename]['changed_tensor_count'] == len(deltas)
    summary[export_name] = {'manifest': manifest, 'changes': changed}
(exp / 'prefix_verification.json').write_text(json.dumps(summary, indent=2) + '\n')
print(json.dumps(summary, indent=2))
