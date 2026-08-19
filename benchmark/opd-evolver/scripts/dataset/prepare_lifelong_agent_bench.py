#!/usr/bin/env python3
"""Prepare LifelongAgentBench data for opd-evolver.

将 HF 数据集 csyq/LifelongAgentBench 的 parquet 转成 bench_lifelong_agent.py 期望的
``data/lifelong/processed/{db,os,kg}/{train,test}.jsonl`` 布局。

parquet 里 entity_dict / table_info / answer_info / initialization_command_item 等
字段是“字符串化的 dict/list”，这里用与 bench_lifelong_agent._maybe_parse_obj 相同的
逻辑（先 json.loads 再 ast.literal_eval）解析回 Python 对象。

注意: HF 发布只有 train split。这里同时写出 train.jsonl 与 test.jsonl（内容相同），
以便 README 中 ``--split test`` 直接可用; 如需严格划分请自行切分。
"""
from __future__ import annotations

import ast
import json
from pathlib import Path

import pyarrow.parquet as pq

ROOT = Path(__file__).resolve().parents[2]
SRC = ROOT / "data" / "lifelong" / "processed" / "parquet"
DST = ROOT / "data" / "lifelong" / "processed"

# parquet 文件名(子目录) -> 任务类型
MAPPING = {"db_bench": "db", "os_interaction": "os", "knowledge_graph": "kg"}


def parse(value):
    """复刻 bench_lifelong_agent._maybe_parse_obj: json.loads 失败则 ast.literal_eval。"""
    if not isinstance(value, str):
        return value
    text = value.strip()
    if not text:
        return value
    try:
        return json.loads(text)
    except Exception:
        pass
    try:
        return ast.literal_eval(text)
    except Exception:
        return value


def convert(name: str) -> None:
    pq_path = SRC / f"{name}.parquet"
    if not pq_path.is_file():
        print(f"[skip] {name}: parquet not found at {pq_path}")
        return
    rows = pq.read_table(pq_path).to_pylist()
    out_dir = DST / MAPPING[name]
    out_dir.mkdir(parents=True, exist_ok=True)
    records: list[dict] = []
    for r in rows:
        rec = {k: parse(v) for k, v in r.items()}
        # bench 优先用 task_id, 没有则回退到 {task_type}_{idx}; 这里补一个稳定 id
        rec.setdefault("task_id", f"{MAPPING[name]}_{rec.get('sample_index', 0)}")
        records.append(rec)
    for split in ("train", "test"):
        out_file = out_dir / f"{split}.jsonl"
        with out_file.open("w", encoding="utf-8") as f:
            for rec in records:
                f.write(json.dumps(rec, ensure_ascii=False) + "\n")
    print(f"[ok] {name} -> {MAPPING[name]}: {len(records)} rows (train+test written)")


def main() -> None:
    if not SRC.is_dir():
        raise SystemExit(
            f"parquet source dir not found: {SRC}\n"
            "Download csyq/LifelongAgentBench parquets into data/lifelong/processed/parquet/ first."
        )
    for name in MAPPING:
        convert(name)
    print(f"\nDone. Output under: {DST}/{{db,os,kg}}/{{train,test}}.jsonl")


if __name__ == "__main__":
    main()
