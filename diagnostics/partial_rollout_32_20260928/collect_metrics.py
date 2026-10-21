import json
from pathlib import Path
import re

exp = Path('/home/liyu.468/learning_kvcache/outputs/partial_rollout_32_h200_20260928')
content = re.sub(r'\x1b\[[0-9;]*m', '', (exp / 'launcher.log').read_text())
metrics = []
for line in content.splitlines():
    if re.search(r'\bstep:\s*\d+\s*-', line):
        metrics.append({key: float(value) for key, value in re.findall(
            r'([\w/.-]+):\s*(?:np\.\w+\()?([-+]?(?:\d+(?:\.\d*)?|\.\d+)(?:[eE][-+]?\d+)?|nan|inf)', line)})
result = {'completed': 'Final validation metrics:' in content, 'metrics': metrics}
(exp / 'metrics.json').write_text(json.dumps(result, indent=2) + '\n')
print('completed:', result['completed'])
for row in metrics:
    print({key: value for key, value in row.items() if key in [
        'step', 'timing_s/gen', 'timing_s/update_actor', 'timing_s/old_log_prob', 'timing_s/feed',
        'timing_s/step', 'actor/distillation/loss', 'actor/grad_norm',
        'training/off_policy/trajectory_spans/max', 'response_length/mean'] or key.startswith('training/partial/')})
