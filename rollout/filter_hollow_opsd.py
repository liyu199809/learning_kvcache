#!/usr/bin/env python
"""Filter vacuous "hollow advice" rows out of an OPSD parquet dataset.

Rows whose teacher_prompt is the content-free POSITIVE_FEEDBACK fallback
("This task is within your ability ...") carry no privileged signal: they
come from trajectories where every tool call errored (broken env) yet the
verifier passed vacuously. Distilling them adds noise, and their RL label
is wrong (the student actually failed).

Usage:
    python -m rollout.filter_hollow_opsd \
        --data-dir traj_data/awm_opsd_full_task1 [--dry-run]
"""
from __future__ import annotations

import argparse
import json
import shutil
from pathlib import Path

import pandas as pd

HOLLOW_MARKER = "This task is within your ability and can be solved directly."


def is_hollow(teacher_prompt: str) -> bool:
    return teacher_prompt.startswith("[Expert advice]") and HOLLOW_MARKER in teacher_prompt


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--data-dir", required=True)
    ap.add_argument("--dry-run", action="store_true")
    args = ap.parse_args()

    data_dir = Path(args.data_dir)
    manifest_path = data_dir / "split_manifest.json"
    manifest = json.loads(manifest_path.read_text()) if manifest_path.exists() else {}

    suffix = "" if args.dry_run else ".bak_hollow"
    for split in ("train", "val"):
        path = data_dir / f"{split}.parquet"
        if not path.exists():
            continue
        df = pd.read_parquet(path)
        mask = df["teacher_prompt"].map(is_hollow)
        print(f"{split}: {len(df)} rows -> {int(mask.sum())} hollow, "
              f"{int((~mask).sum())} kept")
        if args.dry_run or mask.sum() == 0:
            continue

        if not path.with_suffix(path.suffix + suffix).exists():
            shutil.copy2(path, path.with_suffix(path.suffix + suffix))
        df[~mask].to_parquet(path, index=False)

        if split == "train" and manifest:
            manifest["train_rows"] = int((~mask).sum())
            manifest["hollow_rows_removed"] = int(mask.sum())
            manifest["train_contains_all_selected_rows"] = True

    if not args.dry_run and manifest:
        if not manifest_path.with_suffix(".bak_hollow").exists():
            shutil.copy2(manifest_path, manifest_path.with_suffix(".bak_hollow"))
        manifest_path.write_text(json.dumps(manifest, indent=1))
        print(f"manifest updated: {manifest_path}")


if __name__ == "__main__":
    main()
