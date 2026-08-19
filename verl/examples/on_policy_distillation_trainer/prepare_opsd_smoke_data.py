#!/usr/bin/env python3
"""Build a tiny, deterministic Parquet dataset for the OPSD smoke test.

The student sees only ``prompt``.  The frozen self-teacher additionally sees
``reward_model.ground_truth`` and the raw problem in ``extra_info.problem`` via
``distillation.privileged_mode=chat_turn``.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import pyarrow as pa
import pyarrow.parquet as pq

EXAMPLES = [
    {
        "problem": "A box contains 12 red balls and 8 blue balls. Five red balls are removed. How many balls remain?",
        "solution": "There are 12 + 8 = 20 balls initially. Removing 5 red balls leaves 20 - 5 = 15 balls. #### 15",
    },
    {
        "problem": "A train travels 180 kilometers in 3 hours at a constant speed. How far does it travel in 5 hours?",
        "solution": (
            "The speed is 180 / 3 = 60 kilometers per hour. In 5 hours it travels 60 * 5 = 300 kilometers. #### 300"
        ),
    },
    {
        "problem": (
            "Mina has 24 stickers. She gives one quarter of them away and then buys 9 more. "
            "How many stickers does she have?"
        ),
        "solution": (
            "One quarter of 24 is 24 / 4 = 6. After giving them away Mina has 24 - 6 = 18, then 18 + 9 = 27. #### 27"
        ),
    },
    {
        "problem": "The sum of three consecutive integers is 72. What is the largest integer?",
        "solution": "Let the integers be n-1, n, and n+1. Their sum is 3n = 72, so n = 24. The largest is 25. #### 25",
    },
    {
        "problem": "A rectangle is 9 meters long and 6 meters wide. What is its perimeter in meters?",
        "solution": "A rectangle's perimeter is 2 times length plus width: 2 * (9 + 6) = 2 * 15 = 30. #### 30",
    },
    {
        "problem": (
            "A baker packs 84 cookies equally into 7 boxes. Then 2 cookies are eaten from each box. "
            "How many cookies remain altogether?"
        ),
        "solution": (
            "Each box starts with 84 / 7 = 12 cookies. After 2 are eaten, each has 10, so 7 * 10 = 70 remain. #### 70"
        ),
    },
]


def make_row(example: dict[str, str], index: int, split: str) -> dict:
    problem = example["problem"]
    return {
        "data_source": "openai/gsm8k",
        "prompt": [{"role": "user", "content": problem}],
        "ability": "math",
        "reward_model": {"style": "rule", "ground_truth": example["solution"]},
        "extra_info": {"problem": problem, "index": index, "split": split},
    }


def write_parquet(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    table = pa.Table.from_pylist(rows)
    pq.write_table(table, path)
    print(f"wrote {len(rows)} rows to {path}")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("traj_data/opsd_smoke"),
        help="Directory receiving train.parquet and val.parquet.",
    )
    args = parser.parse_args()

    train_rows = [make_row(example, i, "train") for i, example in enumerate(EXAMPLES[:4])]
    val_rows = [make_row(example, i + 4, "val") for i, example in enumerate(EXAMPLES[4:])]
    write_parquet(args.output_dir / "train.parquet", train_rows)
    write_parquet(args.output_dir / "val.parquet", val_rows)


if __name__ == "__main__":
    main()
