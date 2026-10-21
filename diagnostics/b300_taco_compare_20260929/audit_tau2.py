"""Require complete tau2 outputs and verify the historical evaluation protocol."""
import collections
import json
import subprocess
from pathlib import Path
ROOT = Path('/home/liyu.468/learning_kvcache')
OUT = ROOT/'outputs/b300_taco_compare_20260929'
HIST = ROOT/'outputs/prefix_composition_3domain_20260922_4gpu/eval/serial_dp4_20260923_161445/taco'
checks = {}
for domain, count in [('airline', 50), ('retail', 114)]:
    suite = f'tau2_{domain}'
    old = json.loads((HIST/suite/'results.json').read_text())
    new = json.loads((OUT/'new'/suite/'results.json').read_text())
    simulations = new['simulations']
    assert len(new['tasks']) == len(simulations) == count, f'Incomplete {suite}'
    assert collections.Counter(x['task_id'] for x in simulations) == collections.Counter(x['id'] for x in new['tasks'])
    assert all(x['reward_info'] is not None for x in simulations)
    oi, ni = old['info'], new['info']
    row = {k: oi[k] == ni[k] for k in ('num_trials', 'max_steps', 'max_errors', 'seed')}
    # B300's ownership check causes tau2's unconfigured git call to report unknown.
    actual_commit = subprocess.check_output(['git', '-c', f'safe.directory={ROOT}',
        '-C', str(ROOT), 'rev-parse', 'HEAD'], text=True).strip()
    row['source_commit'] = actual_commit == oi['git_commit'] and ni['git_commit'] in (actual_commit, 'unknown')
    row['task_definitions'] = sorted(old['tasks'], key=lambda x:x['id']) == sorted(new['tasks'], key=lambda x:x['id'])
    row['user_config'] = oi['user_info'] == ni['user_info']
    def agent(info):
        a = dict(info['agent_info']); a.pop('llm', None)
        a['llm_args'] = {k:v for k,v in a['llm_args'].items() if k not in ('api_base', 'api_key')}
        return a
    row['agent_sampling'] = agent(oi) == agent(ni)
    row['simulation_seeds'] = {x['task_id']:x['seed'] for x in old['simulations']} == {x['task_id']:x['seed'] for x in simulations}
    row['environment_config'] = oi['environment_info'] == ni['environment_info']
    checks[suite] = row
(OUT/'tau2_protocol_audit.json').write_text(json.dumps({'checks':checks,
    'git_metadata_note':'New tau2 records unknown due to the B300 Git ownership check; source HEAD is verified explicitly.'}, indent=2))
print(json.dumps(checks, indent=2))
assert all(all(row.values()) for row in checks.values()), 'tau2 protocol differences require review'
