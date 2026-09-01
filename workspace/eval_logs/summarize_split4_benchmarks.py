#!/usr/bin/env python3
"""Summarize split-4 cluster/merge benchmark outputs into one Markdown report."""

from __future__ import annotations

import csv
import json
import sys
from datetime import datetime
from pathlib import Path


MODELS = [
    {
        "slug": "cluster0-step35",
        "label": "Cluster 0 (step 35)",
        "model_path": "/mnt/storage/disk1/verl_data/base_model/Qwen3.5-4B-HybridDeltaResidualAttentionPrefix-split4-trained-eval/cluster0-step35",
        "checkpoint": "checkpoints/self_evolver_opsd/qwen3_5_4b_awm_opsd_hybrid_prefix_split4_cluster0/global_step_35/actor",
        "prefix": "G512/A64",
    },
    {
        "slug": "cluster1-step44",
        "label": "Cluster 1 (step 44)",
        "model_path": "/mnt/storage/disk1/verl_data/base_model/Qwen3.5-4B-HybridDeltaResidualAttentionPrefix-split4-trained-eval/cluster1-step44",
        "checkpoint": "checkpoints/self_evolver_opsd/qwen3_5_4b_awm_opsd_hybrid_prefix_split4_cluster1/global_step_44/actor",
        "prefix": "G512/A64",
    },
    {
        "slug": "cluster2-step66",
        "label": "Cluster 2 (step 66)",
        "model_path": "/mnt/storage/disk1/verl_data/base_model/Qwen3.5-4B-HybridDeltaResidualAttentionPrefix-split4-trained-eval/cluster2-step66",
        "checkpoint": "checkpoints/self_evolver_opsd/qwen3_5_4b_awm_opsd_hybrid_prefix_split4_cluster2/global_step_66/actor",
        "prefix": "G512/A64",
    },
    {
        "slug": "cluster3-step57",
        "label": "Cluster 3 (step 57)",
        "model_path": "/mnt/storage/disk1/verl_data/base_model/Qwen3.5-4B-HybridDeltaResidualAttentionPrefix-split4-trained-eval/cluster3-step57",
        "checkpoint": "checkpoints/self_evolver_opsd/qwen3_5_4b_awm_opsd_hybrid_prefix_split4_cluster3/global_step_57/actor",
        "prefix": "G512/A64",
    },
    {
        "slug": "merged-split4",
        "label": "Merged split-4 model",
        "model_path": "/mnt/storage/disk1/verl_data/base_model/Qwen3.5-4B-HybridDeltaResidualAttentionPrefix-G2048-A256-step203-split4-trained-merged",
        "checkpoint": "four latest cluster checkpoints concatenated in cluster0/1/2/3 order",
        "prefix": "G2048/A256",
    },
]


def percentage(value: str) -> float:
    return float(value.rstrip("%"))


def read_one_csv(path: Path) -> dict[str, str] | None:
    if not path.is_file():
        return None
    with path.open(newline="") as handle:
        rows = list(csv.DictReader(handle))
    return rows[0] if len(rows) == 1 else None


def bfcl_metrics(model_root: Path) -> dict[str, float] | None:
    score_root = model_root / "bfcl_v3" / "all" / "score"
    overall = read_one_csv(score_root / "data_overall.csv")
    if overall is None:
        return None
    score_files = list((score_root / "Qwen_Qwen3-4B-FC").glob("*_score.json"))
    result_files = list(
        (model_root / "bfcl_v3" / "all" / "result" / "Qwen_Qwen3-4B-FC").glob(
            "*_result.json"
        )
    )
    if len(score_files) != 17 or len(result_files) != 17:
        return None
    return {
        "non_live": percentage(overall["Non-Live AST Acc"]),
        "live": percentage(overall["Live Acc"]),
        "multi_turn": percentage(overall["Multi Turn Acc"]),
        "hallucination": percentage(overall["Irrelevance Detection"]),
        "overall": percentage(overall["Overall Acc"]),
    }


def tau_metrics(path: Path, expected: int) -> tuple[int, int] | None:
    if not path.is_file():
        return None
    payload = json.loads(path.read_text())
    simulations = payload.get("simulations", [])
    task_ids = [item.get("task_id") for item in simulations]
    if len(simulations) != expected or len(set(task_ids)) != expected:
        return None
    passed = sum(
        float(item.get("reward_info", {}).get("reward", 0.0)) == 1.0
        for item in simulations
    )
    return passed, expected


def lifelong_metrics(path: Path, expected: int = 500) -> tuple[int, int, int] | None:
    if not path.is_file():
        return None
    with path.open(newline="") as handle:
        rows = list(csv.DictReader(handle))
    if len(rows) != expected or len({row["task_id"] for row in rows}) != expected:
        return None
    passed = sum(row["success"].lower() == "true" for row in rows)
    errors = sum(bool(row.get("error", "").strip()) for row in rows)
    return passed, expected, errors


def fmt_pct(count_total: tuple[int, int] | None) -> str:
    if count_total is None:
        return "—"
    count, total = count_total
    return f"{count / total * 100:.2f} ({count}/{total})"


def main() -> int:
    root = Path(sys.argv[1]) if len(sys.argv) > 1 else Path(
        "workspace/eval_logs/split4_cluster_merge_20260828"
    )
    report = Path(sys.argv[2]) if len(sys.argv) > 2 else Path(
        "workspace/eval_logs/benchmark_results_split4_cluster_merge_20260828.md"
    )
    rows = []
    for model in MODELS:
        model_root = root / model["slug"]
        bfcl = bfcl_metrics(model_root)
        airline = tau_metrics(model_root / "tau2" / "airline" / "results.json", 50)
        retail = tau_metrics(model_root / "tau2" / "retail" / "results.json", 114)
        db = lifelong_metrics(model_root / "lifelong_db" / "test" / "summary.csv")
        os_result = lifelong_metrics(
            model_root / "lifelong_os" / "test" / "summary.csv"
        )
        tau_overall = None
        if airline and retail:
            tau_overall = (airline[0] + retail[0], airline[1] + retail[1])
        lifelong_macro = None
        if db and os_result:
            lifelong_macro = ((db[0] / db[1]) + (os_result[0] / os_result[1])) * 50
        complete = all((bfcl, airline, retail, db, os_result))
        rows.append(
            {
                **model,
                "bfcl": bfcl,
                "airline": airline,
                "retail": retail,
                "tau_overall": tau_overall,
                "db": db,
                "os": os_result,
                "lifelong_macro": lifelong_macro,
                "complete": complete,
            }
        )

    complete_count = sum(row["complete"] for row in rows)
    now = datetime.now().astimezone().strftime("%Y-%m-%d %H:%M:%S %Z")
    lines = [
        "# Split-4 Cluster / Merge 模型工具使用评测",
        "",
        f"最后更新：{now}",
        "",
        f"> 当前进度：{complete_count}/5 个模型完成全部评测。表中的 `—` 表示该项尚未形成完整、可校验的正式产物。",
        "",
        "## 1. 实验对象",
        "",
        "| 模型 | Prefix 结构 | 训练 checkpoint / 合并来源 | 推理模型 |",
        "|---|---:|---|---|",
    ]
    for row in rows:
        lines.append(
            f"| {row['label']} | {row['prefix']} | `{row['checkpoint']}` | `{row['model_path']}` |"
        )
    lines += [
        "",
        "四个 cluster checkpoint 均聚合为 BF16 prefix-only HF 评测目录；dense base 权重通过同文件系统硬链接复用。Merged 模型按 cluster0 → cluster1 → cluster2 → cluster3 的顺序拼接四段 prefix。",
        "",
        "## 2. 统一评测口径",
        "",
        "| Benchmark | 数据与设置 |",
        "|---|---|",
        "| BFCL v3 | 官方 `all`，17 类、4441 条；temperature=0；并发 64；官方 generate → evaluate → aggregate |",
        "| tau2-bench | Airline `base` 50 条 + Retail `base` 114 条；1 trial；max steps=50；agent/user max tokens=4096；thinking off；并发 32；Retail NL judge=`deepseek-v4-pro-ga-260813` |",
        "| LifelongDB | `test` 500 条；max steps=6；max tokens=2048；MySQL 8.0；并发 16 |",
        "| LifelongOS | `test` 500 条；max steps=8；max tokens=2048；shell timeout=20s；并发 16 |",
        "",
        "推理统一使用 vLLM 0.18.0、BF16、8×A800 data parallel；每次只加载一个模型，跑完该模型的全部 benchmark 后再切换模型。tau2 是单次 self-play trial，user simulator 使用与被测 agent 相同的模型，因此结果包含模拟轨迹的单次运行波动。",
        "",
        "## 3. 主结果",
        "",
        "### 3.1 BFCL v3",
        "",
        "| 模型 | Non-Live AST | Live | Multi-Turn | Hall. / Irrelevance | Overall |",
        "|---|---:|---:|---:|---:|---:|",
    ]
    for row in rows:
        b = row["bfcl"]
        values = ["—"] * 5 if b is None else [f"{b[k]:.2f}" for k in ("non_live", "live", "multi_turn", "hallucination", "overall")]
        lines.append(f"| {row['label']} | " + " | ".join(values) + " |")
    lines += [
        "",
        "单位：%。`Hall.` 是官方 Overall CSV 的 Irrelevance Detection。",
        "",
        "### 3.2 tau2-bench",
        "",
        "| 模型 | Airline Pass@1 | Retail Pass@1 | Overall Pass@1 |",
        "|---|---:|---:|---:|",
    ]
    for row in rows:
        lines.append(
            f"| {row['label']} | {fmt_pct(row['airline'])} | {fmt_pct(row['retail'])} | {fmt_pct(row['tau_overall'])} |"
        )
    lines += [
        "",
        "### 3.3 LifelongAgentBench",
        "",
        "| 模型 | LifelongDB Pass@1 | LifelongOS Pass@1 | Macro Avg | Runtime errors |",
        "|---|---:|---:|---:|---:|",
    ]
    for row in rows:
        db_pair = None if row["db"] is None else row["db"][:2]
        os_pair = None if row["os"] is None else row["os"][:2]
        macro = "—" if row["lifelong_macro"] is None else f"{row['lifelong_macro']:.2f}"
        errors = "—" if row["db"] is None or row["os"] is None else str(row["db"][2] + row["os"][2])
        lines.append(
            f"| {row['label']} | {fmt_pct(db_pair)} | {fmt_pct(os_pair)} | {macro} | {errors} |"
        )
    lines += [
        "",
        "## 4. 完整性检查与原始产物",
        "",
        "| 模型 | BFCL 17/17 | tau2 Airline 50/50 | tau2 Retail 114/114 | LifelongDB 500/500 | LifelongOS 500/500 | 总状态 |",
        "|---|---:|---:|---:|---:|---:|---|",
    ]
    for row in rows:
        marks = [row["bfcl"], row["airline"], row["retail"], row["db"], row["os"]]
        cells = ["✓" if item is not None else "—" for item in marks]
        state = "完成" if row["complete"] else "运行中/待运行"
        lines.append(f"| {row['label']} | " + " | ".join(cells) + f" | {state} |")
    lines += [
        "",
        f"所有原始结果统一位于 `{root}`：每个模型目录下包含 `bfcl_v3/all/result`、`bfcl_v3/all/score`、`tau2/*/results.json`、`lifelong_*/test/summary.csv` 和逐任务轨迹；vLLM 与 runner 日志保存在同一模型目录。",
        "",
        "完整性判定要求：BFCL 生成与 score 各 17 个类别文件；tau2 task ID 数量完整且无重复；两个 Lifelong 子集各 500 个唯一 task ID。",
    ]
    if complete_count == len(rows):
        bfcl_winner = max(rows, key=lambda row: row["bfcl"]["overall"])
        tau_winner = max(rows, key=lambda row: row["tau_overall"][0] / row["tau_overall"][1])
        lifelong_winner = max(rows, key=lambda row: row["lifelong_macro"])
        merged = rows[-1]
        best_cluster_bfcl = max(rows[:-1], key=lambda row: row["bfcl"]["overall"])
        best_cluster_tau = max(rows[:-1], key=lambda row: row["tau_overall"][0] / row["tau_overall"][1])
        best_cluster_life = max(rows[:-1], key=lambda row: row["lifelong_macro"])
        lines += [
            "",
            "## 5. 结果摘要",
            "",
            f"- BFCL Overall 最佳：{bfcl_winner['label']}（{bfcl_winner['bfcl']['overall']:.2f}%）。Merged 相对最佳 cluster：{merged['bfcl']['overall'] - best_cluster_bfcl['bfcl']['overall']:+.2f} pp。",
            f"- tau2 Overall 最佳：{tau_winner['label']}（{tau_winner['tau_overall'][0] / tau_winner['tau_overall'][1] * 100:.2f}%）。Merged 相对最佳 cluster：{(merged['tau_overall'][0] / merged['tau_overall'][1] - best_cluster_tau['tau_overall'][0] / best_cluster_tau['tau_overall'][1]) * 100:+.2f} pp。",
            f"- Lifelong Macro Avg 最佳：{lifelong_winner['label']}（{lifelong_winner['lifelong_macro']:.2f}%）。Merged 相对最佳 cluster：{merged['lifelong_macro'] - best_cluster_life['lifelong_macro']:+.2f} pp。",
        ]
    report.parent.mkdir(parents=True, exist_ok=True)
    report.write_text("\n".join(lines) + "\n")
    print(report)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
