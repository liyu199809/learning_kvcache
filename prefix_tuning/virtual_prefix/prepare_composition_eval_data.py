"""Fix task-disjoint AWM evaluation and clean first-turn domain evaluations."""
import copy
import hashlib
import json
from pathlib import Path
import random

import pyarrow as pa
import pyarrow.parquet as pq


def main():
    repo = Path(__file__).resolve().parents[2]
    source = repo / 'traj_data/opsd_taco_condensed_v2/sources'
    root = repo / 'outputs/prefix_composition_3domain_20260917/data/eval'
    root.mkdir(parents=True, exist_ok=False)
    table = pq.read_table(source / 'awm/train.parquet')
    train = table.to_pylist()
    task_file = repo / 'awm_data/gen_tasks.jsonl'
    pool = {r['scenario']: r['tasks'] for r in map(json.loads, task_file.read_text().splitlines())}
    occupied = {(r['scenario'], r['task_idx']) for r in train}
    templates = {r['scenario']: r for r in train}
    assert all(r['extra_info']['problem'] == pool[r['scenario']][r['task_idx']] for r in train)
    rng = random.Random(42)
    candidates = {s: [i for i in range(len(pool[s])) if (s, i) not in occupied] for s in templates}
    scenarios = sorted(s for s in candidates if candidates[s])
    rng.shuffle(scenarios)
    rows = []
    for scenario in scenarios[:150]:
        i = rng.choice(candidates[scenario])
        r = copy.deepcopy(templates[scenario])
        problem = pool[scenario][i]
        r['task_idx'] = i
        r['env_config']['task_idx'] = i
        r['prompt'] = [r['prompt'][0], {'role': 'user', 'content': problem}]
        r['teacher_prompt'] = ''
        r['reward_model']['ground_truth'] = ''
        # Clear inherited training provenance/scores rather than mislabel held-out tasks.
        extra = {k: None for k in r['extra_info']}
        extra.update(problem=problem, task_id=f'{scenario}:{i}', scenario=scenario,
                     task_idx=i, sample_type='evaluation', source_dataset='awm')
        r['extra_info'] = extra
        rows.append(r)
    assert len(rows) == 150
    assert not occupied.intersection((r['scenario'], r['task_idx']) for r in rows)
    pq.write_table(pa.Table.from_pylist(rows, schema=table.schema), root / 'awm_first_turn.parquet')
    manifest = {'seed': 42, 'awm_gen_tasks_sha256': hashlib.sha256(task_file.read_bytes()).hexdigest(),
                'awm_scope': 'held-out tasks in 150 training-seen scenarios, one task per scenario',
                'awm_train_overlap': 0, 'awm_task_ids': [r['extra_info']['task_id'] for r in rows]}
    for domain in ['envscaler', 'taco']:
        val = pq.read_table(source / domain / 'val.parquet')
        clean = val.to_pylist()
        for r in clean:
            r['prompt'] = [r['prompt'][0], {'role': 'user', 'content': r['extra_info']['problem']}]
            r['teacher_prompt'] = ''
            r['reward_model']['ground_truth'] = ''
        pq.write_table(pa.Table.from_pylist(clean, schema=val.schema), root / f'{domain}_first_turn.parquet')
        manifest[f'{domain}_task_ids'] = [r['extra_info']['task_id'] for r in clean]
    (root / 'manifest.json').write_text(json.dumps(manifest, indent=2) + '\n')
    print('FIXED_EVAL_READY', {d: len(manifest[f'{d}_task_ids']) for d in ['awm', 'envscaler', 'taco']})


if __name__ == '__main__':
    main()
