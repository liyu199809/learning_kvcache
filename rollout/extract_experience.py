#!/usr/bin/env python
"""Extract structured experience from an OPSD training parquet, then
(optionally) embed it and split the dataset into K clusters.

Stage 1 - extraction
--------------------
The dataset's privileged ``teacher_prompt`` column carries one of two
experience kinds, both produced by rollout/refine2swift.py:

  1. ``tool_chain``   - "[Reference plan]" blocks: the round-1 winning
     trajectory rendered as ordered steps of tool calls, preserving
     parallelism (calls made in one assistant turn share a step):

         Step 1: call tool({"k": "v"})
         Step 2 (parallel):
             - call toolA({...})
             - call toolB({...})

  2. ``advice``       - "[Expert advice]" blocks: one paragraph of teacher
     advice, either corrective (paired with the env-reset Notice, produced
     after a failed student round) or the content-free positive fallback.

Both are parsed back into structured records (task -> steps/calls,
task -> advice text) and written as JSONL, one record per dataset row.

Stage 2 - embedding + clustering (--num-clusters K, default 4)
--------------------------------------------------------------
Each record is turned into a text (see --embed-field), embedded with the
local Qwen/Qwen3-VL-Embedding-2B via offline vLLM (pooling runner, chat
template applied, text-only - no images), L2-normalized, and partitioned
with KMeans into K subsets. Outputs, under <data dir>/cluster_split/:

  * train_cluster{0..K-1}.parquet - the original parquet rows, split by
    cluster, directly consumable as training subsets.
  * clusters.json                  - K, sizes, per-cluster profile (type
    mix, top scenarios, top tools) for interpreting the split.
  * embeddings.npz                 - cached embeddings + text hashes, so
    re-running with a different K skips the GPU pass entirely.

The experience.jsonl records gain a "cluster" field.

Usage:
    python -m rollout.extract_experience \
        --train-file traj_data/awm_opsd_full_task1/train.parquet \
        --num-clusters 4 [--gpu 0] [--embed-field task+experience]
"""
from __future__ import annotations

import argparse
import collections
import hashlib
import json
import os
import re
import statistics
from pathlib import Path
from typing import Any, Dict, List, Optional

import numpy as np
import pandas as pd

PLAN_MARKER = "[Reference plan]"
ADVICE_MARKER = "[Expert advice]"
# Content-free positive fallback emitted when the teacher had nothing to teach
# (mirrors rollout/filter_hollow_opsd.py).
HOLLOW_MARKER = "This task is within your ability and can be solved directly."
# Suffixes that terminate the experience block (refine2swift.py appends one).
SUFFIX_RE = re.compile(r"\n\n(?:\[User Question\]|Notice:)", re.S)

# Step grammar (exact inverse of _render_success_trajectory):
#   "Step 3: call name({...})"            -> single-call step
#   "Step 3 (parallel):" + "    - call name({...})"  -> multi-call step
_STEP_RE = re.compile(r"^Step (\d+)( \(parallel\))?:\s*(.*)$")
_CALL_RE = re.compile(r"^\s*-\s*call\s+(.+?)\((.*)\)\s*$")
_INLINE_CALL_RE = re.compile(r"^call\s+(.+?)\((.*)\)\s*$")

# Default local copy of the embedding model (HF cache id as fallback).
DEFAULT_EMBED_MODEL_CANDIDATES = [
    "/mnt/storage/disk1/verl_data/base_model/Qwen3-VL-Embedding-2B",
    "Qwen/Qwen3-VL-Embedding-2B",
]
# Mirrors the reference vLLM deployment case's default instruction.
DEFAULT_EMBED_INSTRUCTION = "Represent the user's input."


# --------------------------------------------------------------------------
# Stage 1: parse teacher_prompt back into structured experience
# --------------------------------------------------------------------------

def _lenient_json(s: str) -> Any:
    try:
        return json.loads(s)
    except Exception:
        try:
            return json.loads(s.replace("'", '"'))
        except Exception:
            return s  # keep raw string rather than dropping the call


def parse_plan(text: str) -> List[Dict[str, Any]]:
    """Parse a '[Reference plan]' teacher_prompt body into structured steps.

    Returns [{"calls": [{"tool": str, "arguments": dict}, ...]}, ...] in
    execution order; calls inside one step were issued in parallel.
    """
    steps: List[Dict[str, Any]] = []
    current: Optional[List[Dict[str, Any]]] = None
    for line in text.splitlines():
        line = line.rstrip()
        if not line:
            continue
        m = _STEP_RE.match(line)
        if m:
            current = []
            steps.append({"calls": current})
            rest = m.group(3).strip()
            if rest:  # single-call step on the same line
                cm = _INLINE_CALL_RE.match(rest)
                if cm:
                    current.append({"tool": cm.group(1),
                                    "arguments": _lenient_json(cm.group(2))})
            continue
        cm = _CALL_RE.match(line)
        if cm and current is not None:  # indented "- call ..." under a step
            current.append({"tool": cm.group(1),
                            "arguments": _lenient_json(cm.group(2))})
    return steps


def _strip_suffix(teacher_prompt: str) -> str:
    """Drop the trailing '[User Question]<task>' / 'Notice:' tail."""
    m = SUFFIX_RE.search(teacher_prompt)
    return teacher_prompt[: m.start()] if m else teacher_prompt


def _task_text(row: pd.Series) -> str:
    extra = row.get("extra_info")
    if isinstance(extra, dict) and extra.get("problem"):
        return extra["problem"]
    if isinstance(extra, str):
        try:
            extra = json.loads(extra)
            if extra.get("problem"):
                return extra["problem"]
        except Exception:
            pass
    # Fallback: the last user turn of the student prompt.
    prompt = row.get("prompt")
    msgs = prompt if isinstance(prompt, list) else json.loads(prompt)
    for msg in reversed(msgs):
        if msg.get("role") == "user":
            return msg["content"]
    return ""


def _advice_body(teacher_prompt: str) -> str:
    """Extract the advice paragraph after '# Advice:' up to the suffix."""
    body = _strip_suffix(teacher_prompt)
    m = re.search(r"# Advice:\s*(.+)", body, re.S)
    return m.group(1).strip() if m else body.strip()


def extract_row(row: pd.Series) -> Dict[str, Any]:
    tp = row["teacher_prompt"]
    rec = {
        "scenario": row.get("scenario"),
        "task_idx": int(row.get("task_idx", -1)),
        "task": _task_text(row),
        "type": "tool_chain" if tp.startswith(PLAN_MARKER) else "advice",
    }
    if rec["type"] == "tool_chain":
        body = _strip_suffix(tp)
        # Drop the fixed header; keep only the Step lines.
        idx = body.find("Execute this plan yourself now!!")
        traj = body[idx + len("Execute this plan yourself now!!"):] if idx >= 0 else body
        rec["experience"] = {"steps": parse_plan(traj)}
    else:
        advice = _advice_body(tp)
        rec["experience"] = {
            "advice": advice,
            "advice_kind": "positive" if HOLLOW_MARKER in advice else "corrective",
        }
    return rec


# --------------------------------------------------------------------------
# Stage 2: embedding text construction
# --------------------------------------------------------------------------

def render_steps_text(steps: List[Dict[str, Any]]) -> str:
    """Render parsed steps as a compact 'tool(args)' chain for embedding."""
    lines = []
    for i, step in enumerate(steps, 1):
        calls = " | ".join(
            f"{c['tool']}({json.dumps(c['arguments'], ensure_ascii=False)})"
            for c in step["calls"])
        lines.append(f"Step {i}: {calls}")
    return "\n".join(lines)


def build_embed_text(rec: Dict[str, Any], mode: str = "task+experience") -> str:
    """Text fed to the embedding model for one experience record.

    mode: 'task' (task text only), 'experience' (advice / tool chain only),
    or 'task+experience' (both, the default).
    """
    exp = rec["experience"]
    if exp.get("steps") is not None:
        exp_text = "Tool call chain:\n" + render_steps_text(exp["steps"])
    else:
        exp_text = "Expert advice:\n" + exp["advice"]
    if mode == "task":
        return rec["task"]
    if mode == "experience":
        return exp_text
    return f"Task: {rec['task']}\n\n{exp_text}"


# --------------------------------------------------------------------------
# Stage 2: offline vLLM embedding (Qwen3-VL-Embedding, text-only)
# --------------------------------------------------------------------------

def embed_texts(texts: List[str],
                model: str,
                instruction: str = DEFAULT_EMBED_INSTRUCTION,
                gpu: int = 0,
                gpu_memory_utilization: float = 0.10,
                max_model_len: int = 8192,
                dtype: str = "bfloat16") -> np.ndarray:
    """Embed texts with Qwen3-VL-Embedding via offline vLLM pooling.

    Mirrors the reference deployment case: chat template applied with
    add_generation_prompt=True, the instruction living in the system turn,
    inputs are plain text (no multi_modal_data). Must be called AFTER
    CUDA_VISIBLE_DEVICES is set (vllm is imported lazily here).
    """
    from vllm import LLM, EngineArgs  # lazy: after CUDA_VISIBLE_DEVICES

    engine_args = EngineArgs(
        model=model,
        runner="pooling",
        dtype=dtype,
        trust_remote_code=True,
        gpu_memory_utilization=gpu_memory_utilization,
        max_model_len=max_model_len,
        enforce_eager=True,  # GPUs are shared; skip CUDA-graph memory
    )
    llm = LLM(**vars(engine_args))

    tokenizer = llm.get_tokenizer()
    prompts = []
    for text in texts:
        conversation = [
            {"role": "system",
             "content": [{"type": "text", "text": instruction}]},
            {"role": "user",
             "content": [{"type": "text", "text": text}]},
        ]
        prompts.append(tokenizer.apply_chat_template(
            conversation, tokenize=False, add_generation_prompt=True))

    outputs = llm.embed(prompts)
    return np.array([o.outputs.embedding for o in outputs], dtype=np.float32)


def _hashes(texts: List[str]) -> np.ndarray:
    return np.array([hashlib.sha256(t.encode()).hexdigest() for t in texts])


def embed_with_cache(texts: List[str], cache_path: Optional[Path], **kw) -> np.ndarray:
    """embed_texts with an npz cache keyed on the exact text list."""
    if cache_path is not None and cache_path.exists():
        cached = np.load(cache_path, allow_pickle=False)
        if np.array_equal(cached["hashes"], _hashes(texts)):
            print(f"reusing cached embeddings from {cache_path}")
            return cached["embeddings"]
        print("cached embeddings do not match current texts; re-embedding")
    emb = embed_texts(texts, **kw)
    if cache_path is not None:
        np.savez_compressed(cache_path, embeddings=emb, hashes=_hashes(texts))
        print(f"cached embeddings to {cache_path}")
    return emb


# --------------------------------------------------------------------------
# Stage 2: KMeans clustering + subset writing
# --------------------------------------------------------------------------

def cluster_embeddings(emb: np.ndarray, k: int,
                       seed: int = 42) -> np.ndarray:
    """L2-normalize, then KMeans-partition into k clusters."""
    from sklearn.cluster import KMeans

    norms = np.linalg.norm(emb, axis=1, keepdims=True)
    norms[norms == 0] = 1.0
    X = emb / norms
    km = KMeans(n_clusters=k, n_init=20, random_state=seed)
    return km.fit_predict(X).astype(int)


def cluster_profiles(records: List[Dict[str, Any]],
                     labels: np.ndarray, k: int) -> List[Dict[str, Any]]:
    """Interpretability summary per cluster: type mix, top scenarios/tools."""
    profiles = []
    for c in range(k):
        idx = [i for i, l in enumerate(labels) if l == c]
        recs = [records[i] for i in idx]
        types = collections.Counter(r["type"] for r in recs)
        scen = collections.Counter(r["scenario"] for r in recs)
        tools = collections.Counter(
            c_["tool"] for r in recs if r["type"] == "tool_chain"
            for s in r["experience"]["steps"] for c_ in s["calls"])
        n_steps = [len(r["experience"]["steps"]) for r in recs
                   if r["type"] == "tool_chain"]
        profiles.append({
            "cluster": c,
            "size": len(idx),
            "types": dict(types),
            "top_scenarios": scen.most_common(10),
            "top_tools": tools.most_common(10),
            "mean_steps": round(statistics.mean(n_steps), 2) if n_steps else 0.0,
        })
    return profiles


# --------------------------------------------------------------------------
# Driver
# --------------------------------------------------------------------------

def default_embed_model() -> str:
    for cand in DEFAULT_EMBED_MODEL_CANDIDATES:
        if Path(cand).exists():
            return cand
    return DEFAULT_EMBED_MODEL_CANDIDATES[-1]


def main() -> None:
    default_train = os.environ.get(
        "TRAIN_FILE",
        os.path.join(os.environ.get("DATA_DIR", "traj_data/awm_opsd_full_task1"),
                     "train.parquet"))
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--train-file", default=default_train)
    ap.add_argument("--output", default=None,
                    help="output JSONL (default: <train dir>/experience.jsonl)")
    ap.add_argument("--num-clusters", type=int, default=4,
                    help="K for KMeans split; 0 disables stage 2")
    ap.add_argument("--embed-field", choices=["task", "experience", "task+experience"],
                    default="task+experience", help="what text to embed")
    ap.add_argument("--embed-model", default=None,
                    help="embedding model path/id (default: local Qwen3-VL-Embedding-2B)")
    ap.add_argument("--embed-instruction", default=DEFAULT_EMBED_INSTRUCTION)
    ap.add_argument("--gpu", type=int, default=0,
                    help="GPU index for the embedding model (CUDA_VISIBLE_DEVICES)")
    ap.add_argument("--gpu-memory-utilization", type=float, default=0.10,
                    help="vLLM memory fraction; low default because GPUs are shared")
    ap.add_argument("--max-model-len", type=int, default=8192)
    ap.add_argument("--dtype", default="bfloat16")
    ap.add_argument("--cluster-seed", type=int, default=42)
    args = ap.parse_args()

    train_path = Path(args.train_file)
    df = pd.read_parquet(train_path)
    out_path = Path(args.output) if args.output else train_path.parent / "experience.jsonl"

    records: List[Dict[str, Any]] = []
    for _, row in df.iterrows():
        records.append(extract_row(row))

    labels: Optional[np.ndarray] = None
    if args.num_clusters > 0:
        split_dir = train_path.parent / "cluster_split"
        split_dir.mkdir(exist_ok=True)

        texts = [build_embed_text(r, args.embed_field) for r in records]
        longest = max(len(t) for t in texts)
        print(f"embedding {len(texts)} texts (longest {longest} chars) with "
              f"{args.embed_model or default_embed_model()} on GPU {args.gpu}")

        os.environ["CUDA_VISIBLE_DEVICES"] = str(args.gpu)
        emb = embed_with_cache(
            texts,
            cache_path=split_dir / "embeddings.npz",
            model=args.embed_model or default_embed_model(),
            instruction=args.embed_instruction,
            gpu=args.gpu,
            gpu_memory_utilization=args.gpu_memory_utilization,
            max_model_len=args.max_model_len,
            dtype=args.dtype,
        )
        print(f"embeddings: {emb.shape}")

        labels = cluster_embeddings(emb, args.num_clusters, args.cluster_seed)
        for rec, lab in zip(records, labels):
            rec["cluster"] = int(lab)

        # Split the ORIGINAL parquet rows by cluster - directly trainable.
        for c in range(args.num_clusters):
            sub = df.iloc[[i for i, l in enumerate(labels) if l == c]].reset_index(drop=True)
            sub_path = split_dir / f"train_cluster{c}.parquet"
            sub.to_parquet(sub_path)
            print(f"  cluster {c}: {len(sub)} rows -> {sub_path}")

        profiles = cluster_profiles(records, labels, args.num_clusters)
        meta = {
            "train_file": str(train_path),
            "embed_model": args.embed_model or default_embed_model(),
            "embed_field": args.embed_field,
            "embed_instruction": args.embed_instruction,
            "num_clusters": args.num_clusters,
            "cluster_seed": args.cluster_seed,
            "clusters": profiles,
        }
        (split_dir / "clusters.json").write_text(
            json.dumps(meta, ensure_ascii=False, indent=2))
        print(f"cluster profiles -> {split_dir / 'clusters.json'}")

    with open(out_path, "w") as f:
        for rec in records:
            f.write(json.dumps(rec, ensure_ascii=False) + "\n")

    # ---- summary ----
    n_by_type = collections.Counter(r["type"] for r in records)
    n_hollow = sum(r["experience"].get("advice_kind") == "positive" for r in records)
    print(f"\nread {len(df)} rows from {train_path}")
    print(f"wrote {len(records)} records to {out_path}")
    print(f"by type: {dict(n_by_type)} (advice: {n_hollow} positive / "
          f"{n_by_type['advice'] - n_hollow} corrective)")
    if labels is not None:
        sizes = collections.Counter(labels.tolist())
        print(f"clusters: {dict(sorted(sizes.items()))}")
    else:
        n_calls = [len(s["calls"]) for r in records if r["type"] == "tool_chain"
                   for s in r["experience"]["steps"]]
        if n_calls:
            print(f"tool_chain calls/trajectory: mean={statistics.mean(n_calls):.2f} "
                  f"max={max(n_calls)}")


if __name__ == "__main__":
    main()
