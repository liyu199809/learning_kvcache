"""Run the unchanged official BFCL scorer on an explicitly selected smoke subset.

This is a pipeline check, not a comparable BFCL leaderboard score.
"""
import argparse
import json
from pathlib import Path
import runpy
import sys


def select_rows(source, ids):
    rows = [json.loads(line) for line in source.read_text().splitlines() if line.strip()]
    selected = [row for row in rows if row['id'] in ids]
    found = {row['id'] for row in selected}
    if found != ids or len(selected) != len(ids):
        raise ValueError(f'{source}: missing or duplicate selected IDs: {ids - found}')
    return selected


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--ids', required=True, type=Path)
    parser.add_argument('--subset-dir', required=True, type=Path)
    args, official_args = parser.parse_known_args()
    from bfcl_eval.constants import eval_config
    requested = json.loads(args.ids.read_text())
    prompt_dir = args.subset_dir / 'data'
    answer_dir = prompt_dir / 'possible_answer'
    answer_dir.mkdir(parents=True, exist_ok=True)
    for category, task_ids in requested.items():
        ids = set(task_ids)
        if not ids or len(ids) != len(task_ids):
            raise ValueError(f'{category}: selection must be nonempty and unique')
        for source_dir, destination in ((eval_config.PROMPT_PATH, prompt_dir),
                                        (eval_config.POSSIBLE_ANSWER_PATH, answer_dir)):
            source = source_dir / f'BFCL_v3_{category}.json'
            if not source.exists() and source_dir == eval_config.POSSIBLE_ANSWER_PATH \
                    and ('irrelevance' in category or category == 'live_relevance'):
                continue
            rows = select_rows(source, ids)
            (destination / source.name).write_text(''.join(json.dumps(row) + '\n' for row in rows))
    (args.subset_dir / 'selection.json').write_text(json.dumps(requested, indent=2))
    # Only input locations change. Official parsers, state transitions and scoring stay intact.
    eval_config.PROMPT_PATH = prompt_dir.resolve()
    eval_config.POSSIBLE_ANSWER_PATH = answer_dir.resolve()
    print('BFCL SUBSET SMOKE: results are not full leaderboard scores.', flush=True)
    sys.argv = ['bfcl_eval', *official_args]
    runpy.run_module('bfcl_eval', run_name='__main__')


if __name__ == '__main__':
    main()
