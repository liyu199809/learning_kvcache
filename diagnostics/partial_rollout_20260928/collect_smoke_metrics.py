import json
import re
from pathlib import Path

root = Path('/home/liyu.468/learning_kvcache/outputs/partial_rollout_h200_20260928')
all_metrics = {}
for run, filename in [('taco_partial_smoke_v3', 'smoke_v3_launcher.log'),
                      ('envscaler_partial_3actor1teacher_smoke', 'envscaler_smoke_launcher.log'),
                      ('envscaler_resume_step4', 'envscaler_resume_launcher.log')]:
    path = root / filename
    if not path.exists():
        continue
    content = re.sub(r'\x1b\[[0-9;]*m', '', path.read_text())
    metrics = []
    for line in content.splitlines():
        if re.search(r'\bstep:\s*\d+\s*-', line):
            row = {key: float(value) for key, value in re.findall(
                r'([\w/.-]+):\s*(?:np\.\w+\()?([-+]?(?:\d+(?:\.\d*)?|\.\d+)(?:[eE][-+]?\d+)?|nan|inf)', line)}
            metrics.append(row)
    all_metrics[run] = {'log': str(path), 'completed': 'Final validation metrics:' in content,
                        'metrics': metrics}
(root / 'smoke_metrics.json').write_text(json.dumps(all_metrics, indent=2) + '\n')
for run, data in all_metrics.items():
    print(run, 'completed:', data['completed'])
    for row in data['metrics']:
        print({key: value for key, value in row.items() if key == 'step' or
               any(s in key for s in ['grad_norm', 'distill', 'trajectory_spans', 'num_turns',
                                     'timing_s/step', 'timing_s/gen', 'timing_s/update_actor',
                                     'timing_s/update_weights', 'env_error'])})
