#!/usr/bin/env python3
from __future__ import annotations

import json
import sys
from pathlib import Path


root = Path(sys.argv[1])
output = root / "coding_merged/livecodebench"
output.mkdir(parents=True, exist_ok=True)


def read(path: Path) -> list[dict]:
    return [json.loads(line) for line in path.read_text().splitlines() if line.strip()]


v5 = sum((read(path) for path in sorted(root.glob(
    "livecodebench/v5_shard_*/samples.jsonl"
))), [])
v6_new = sum((read(path) for path in sorted(root.glob(
    "livecodebench/v6_new_shard_*/samples.jsonl"
))), [])
if len(v5) != 880 or len(v6_new) != 175:
    raise SystemExit(f"Unexpected sample counts: v5={len(v5)}, v6_new={len(v6_new)}")
v5_ids = {row["task_id"] for row in v5}
v6_ids = {row["task_id"] for row in v6_new}
if len(v5_ids) != 880 or len(v6_ids) != 175 or v5_ids & v6_ids:
    raise SystemExit("LCB samples contain duplicates or overlapping release IDs")

bad_errors = [
    (row["task_id"], row.get("error"))
    for row in v5 + v6_new
    if row.get("error") not in (None, "", "empty_final_answer")
]
if bad_errors:
    raise SystemExit(f"LCB samples contain infrastructure errors: {bad_errors[:20]}")
empty_answers = sum(row.get("error") == "empty_final_answer" for row in v5 + v6_new)
print("empty_final_answer", empty_answers)

for name, rows in (("v5", v5), ("v6", v5 + v6_new)):
    path = output / f"{name}.samples.jsonl"
    path.write_text("".join(json.dumps(row, ensure_ascii=False) + "\n" for row in rows))
    print(name, len(rows), path)
