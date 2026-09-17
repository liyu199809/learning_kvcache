"""Audit actual rendered token lengths without altering the source datasets."""
import json
from pathlib import Path

import pyarrow.parquet as pq
from transformers import AutoTokenizer


def main():
    repo = Path(__file__).resolve().parents[2]
    source = repo / 'traj_data/opsd_taco_condensed_v2'
    root = repo / 'outputs/prefix_composition_3domain_20260917/data'
    root.mkdir(parents=True, exist_ok=True)
    tokenizer = AutoTokenizer.from_pretrained(repo.parent / 'models/Qwen3.5-4B')
    report = {}
    for domain in ['awm', 'envscaler', 'taco']:
        rows = pq.read_table(source / 'sources' / domain / 'train.parquet').to_pylist()
        sizes, teacher_sizes = [], []
        for row in rows:
            tools = json.loads(row['tools']) if isinstance(row['tools'], str) else row['tools']
            text = tokenizer.apply_chat_template(row['prompt'], tools=tools or None,
                                                 tokenize=False, add_generation_prompt=True,
                                                 enable_thinking=domain == 'taco')
            ids = tokenizer.encode(text, add_special_tokens=False)
            sizes.append(len(ids))
            teacher_sizes.append(len(ids) + len(tokenizer.encode('\n\n' + row['teacher_prompt'],
                                                                add_special_tokens=False)))
        cap = 4096 if domain == 'taco' else 28672
        report[domain] = {
            'rows': len(rows), 'prompt_max': max(sizes), 'prompt_median': sorted(sizes)[len(sizes)//2],
            'teacher_input_max_upper_bound': max(teacher_sizes),
            'overlong_ids': [r['extra_info']['task_id'] for r, n in zip(rows, sizes) if n > cap],
            'per_task': [{'task_id': r['extra_info']['task_id'], 'prompt_tokens': n, 'teacher_tokens_upper_bound': t}
                         for r, n, t in zip(rows, sizes, teacher_sizes)],
        }
        print('LENGTH_AUDIT', domain, {k: v for k, v in report[domain].items() if k != 'per_task'}, flush=True)
        (root / 'rendered_length_audit.json').write_text(json.dumps(report, indent=2))


if __name__ == '__main__':
    main()
