#!/usr/bin/env python3
"""Evaluate cleaned TACO final checkpoints, one DP=8 inference dataset at a time.

Uses the existing network-isolated official scoring path. CPU scoring can overlap
the next GPU generation stage; no two generation datasets run concurrently.
"""
from __future__ import annotations

import argparse
import concurrent.futures
import hashlib
import json
import os
from pathlib import Path
import signal
import socket
import subprocess
import threading
import time
import urllib.request

RUNS = {
    "on_on": "qwen3_5_4b_taco_condensed_v2",
    "off_off": "qwen3_5_4b_taco_condensed_v2_nothink",
    "off_on": "qwen3_5_4b_taco_condensed_v2_soff_ton",
}


def read_rows(path):
    return [json.loads(line) for line in Path(path).read_text().splitlines() if line.strip()]


def check_port_available(port):
    with socket.socket() as probe:
        # Match the serving socket: completed HTTP connections may be TIME_WAIT.
        probe.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        probe.bind(("127.0.0.1", port))


def merge_lcb(v5_dir, new_dir, destination):
    a, b = (json.loads((p / "manifest.json").read_text()) for p in (v5_dir, new_dir))
    assert a["revision"] == b["revision"] and a["thinking"] == b["thinking"]
    assert a["model"] == b["model"] and a["max_tokens"] == b["max_tokens"]
    for key in ("temperature", "top_p", "top_k", "min_p", "presence_penalty", "repetition_penalty", "seed", "num_samples"):
        assert a.get(key) == b.get(key), f"LCB merge sampling mismatch: {key}"
    assert a["release"] == "release_v5" and b["release"] == "release_v6"
    assert a["available_tasks"] == 880 and b["available_tasks"] == 1055
    v5, new = read_rows(v5_dir / "samples.jsonl"), read_rows(new_dir / "samples.jsonl")
    for rows, manifest, expected in ((v5, a, 880), (new, b, 175)):
        ids = [r["task_id"] for r in rows]
        assert len(ids) == len(set(ids)) == expected
        assert set(ids) == set(manifest["selected_tasks"])
        assert all(r["sample_index"] == 0 for r in rows)
    assert not ({r["task_id"] for r in v5} & {r["task_id"] for r in new})
    with destination.open("x") as output:
        for row in v5 + new:
            output.write(json.dumps(row, ensure_ascii=False) + "\n")


class Suite:
    def __init__(self, args):
        self.args = args
        self.max_tokens = getattr(args, "max_tokens", 16384)
        self.max_model_len = getattr(args, "max_model_len", 32768)
        self.lcb_only = getattr(args, "lcb_only", False)
        self.sampling = getattr(args, "sampling", "greedy")
        self.sampling_config = ({"temperature": 0.6, "top_p": 0.95, "top_k": 20,
                                 "min_p": 0.0, "presence_penalty": 0.0,
                                 "repetition_penalty": 1.0, "seed": 42}
                                if self.sampling == "coding" else {"temperature": 0, "top_p": 1})
        assert self.max_tokens > 0 and self.max_model_len > self.max_tokens
        self.root = Path(args.root).resolve()
        self.out = Path(args.output_root).resolve()
        if args.resume:
            assert self.out.is_dir(), "Resume requires an existing suite directory"
        else:
            self.out.mkdir(parents=True, exist_ok=False)
        self.lock = threading.Lock()
        self.server = None
        self.server_log = None
        self.futures = []
        self.env = {**os.environ, "TOKENIZERS_PARALLELISM": "false", "PYTHONUNBUFFERED": "1"}
        self.python = str(self.root / ".venv/bin/python")
        self.endpoint = f"http://127.0.0.1:{args.port}/v1"

    def event(self, stage, **details):
        row = {"time": time.strftime("%Y-%m-%d %H:%M:%S%z"), "stage": stage, **details}
        with self.lock:
            with (self.out / "events.jsonl").open("a") as f:
                f.write(json.dumps(row, ensure_ascii=False) + "\n")
            print(json.dumps(row, ensure_ascii=False), flush=True)

    def run_logged(self, command, log, **details):
        with log.open("x") as f:
            proc = subprocess.Popen(command, cwd=self.root, env=self.env, stdout=f, stderr=subprocess.STDOUT)
            self.event("process_started", pid=proc.pid, log=str(log), **details)
            return proc.wait()

    def stop_server(self):
        proc = self.server
        if proc is None:
            return
        # Only the process group created by this suite. No global pkill/ray stop.
        try:
            os.killpg(proc.pid, signal.SIGTERM)
        except ProcessLookupError:
            pass
        try:
            proc.wait(timeout=45)
        except subprocess.TimeoutExpired:
            os.killpg(proc.pid, signal.SIGKILL)
            proc.wait(timeout=30)
        try:
            os.killpg(proc.pid, signal.SIGKILL)
        except ProcessLookupError:
            pass
        self.server_log.close()
        self.server = None
        self.event("server_stopped", pid=proc.pid)
        time.sleep(5)

    def start_server(self, model_path, label):
        check_port_available(self.args.port)
        command = [str(self.root / ".venv/bin/vllm"), "serve", str(model_path),
                   "--served-model-name", label, "--host", "127.0.0.1", "--port", str(self.args.port),
                   "--data-parallel-size", "8", "--tensor-parallel-size", "1",
                   "--api-server-count", "8", "--distributed-executor-backend", "uni",
                   "--dtype", "bfloat16", "--max-model-len", str(self.max_model_len),
                   "--max-num-batched-tokens", "32768", "--max-num-seqs", "64",
                   "--gpu-memory-utilization", "0.85", "--seed", "42",
                   "--reasoning-parser", "qwen3", "--generation-config", "vllm"]
        log_path = self.out / f"server_{label}.log"
        attempt = 1
        while log_path.exists():
            log_path = self.out / f"server_{label}_resume_{attempt}.log"
            attempt += 1
        self.server_log = log_path.open("x")
        self.server = subprocess.Popen(
            command, cwd=self.root, env={**self.env, "CUDA_VISIBLE_DEVICES": "0,1,2,3,4,5,6,7"},
            stdout=self.server_log, stderr=subprocess.STDOUT, start_new_session=True)
        self.event("server_start", model=label, pid=self.server.pid, command=command)
        deadline = time.monotonic() + 1200
        while time.monotonic() < deadline:
            if self.server.poll() is not None:
                raise RuntimeError(f"Server exited: {label}; see server log")
            try:
                with urllib.request.urlopen(self.endpoint + "/models", timeout=5) as response:
                    data = json.load(response)
                if label in {m["id"] for m in data["data"]}:
                    self.event("server_ready", model=label)
                    return
            except (OSError, ValueError):
                pass
            time.sleep(5)
        raise TimeoutError(f"Server not ready: {label}")

    def common(self, model, mode):
        command = [self.python, "-m", "benchmark.eval.run_eval", "--model", model,
                "--openai-base-url", self.endpoint, "--api-key", "EMPTY",
                "--code-thinking", mode,
                "--temperature", str(self.sampling_config["temperature"]),
                "--top-p", str(self.sampling_config["top_p"]),
                "--llm-max-completion-tokens", str(self.max_tokens), "--step-timeout", "3600",
                "--concurrency", "512", "--code-n-samples", "1", "--code-pass-k", "1"]
        for key, value in self.sampling_config.items():
            if key not in ("temperature", "top_p"):
                command.extend(["--code-" + key.replace("_", "-"), str(value)])
        return command

    def check_sampling(self, manifest):
        assert manifest["max_tokens"] == self.max_tokens
        for key, value in self.sampling_config.items():
            assert manifest.get(key) == value, f"Resume sampling mismatch: {key}"

    def generate(self, model, mode, name, extra, parent, count):
        output = parent / (name + "_generation")
        retry_args = []
        if self.args.resume and output.exists():
            manifest = json.loads((output / "manifest.json").read_text())
            assert manifest["model"] == model and manifest["thinking"] == mode
            self.check_sampling(manifest)
            previous = read_rows(output / "samples.jsonl")
            assert len(previous) == len({r["task_id"] for r in previous}) <= count
            if len(previous) == count:
                assert all(r.get("error") in (None, "", "empty_final_answer") for r in previous)
                self.event("generation_reused", model=model, dataset=name, rows=count)
                return output
            assert all(not r.get("error") for r in previous), "Partial retries must not resample model failures"
            retry_args = ["--code-retry-errors-from", str(output / "samples.jsonl")]
            attempt = 1
            while (parent / f"{name}_generation_retry{attempt}").exists():
                attempt += 1
            output = parent / f"{name}_generation_retry{attempt}"
            self.event("generation_resume", model=model, dataset=name, preserved=len(previous),
                       missing=count-len(previous), source=retry_args[-1])
        command = self.common(model, mode) + extra + retry_args + ["--code-generate-only", "--output-dir", str(output)]
        rc = self.run_logged(command, parent / (output.name + ".log"),
                             model=model, mode=mode, dataset=name, kind="generation")
        if rc not in (0, 1) or not (output / "samples.jsonl").exists():
            raise RuntimeError(f"Generation failed: {model}/{mode}/{name}, rc={rc}")
        rows = read_rows(output / "samples.jsonl")
        assert len(rows) == len({r["task_id"] for r in rows}) == count
        infrastructure_errors = [r for r in rows if r.get("error") not in (None, "", "empty_final_answer")]
        if infrastructure_errors:
            raise RuntimeError(f"Infrastructure errors in {output}: {len(infrastructure_errors)}; preserve and retry")
        self.event("generation_complete", model=model, mode=mode, dataset=name, rows=len(rows),
                   empty_answers=sum(bool(r.get("error")) for r in rows),
                   truncated=sum(r.get("finish_reason") == "length" for r in rows))
        return output

    def score(self, model, mode, name, extra, samples, parent):
        output = parent / name
        command = self.common(model, mode) + extra + [
            "--code-samples", str(samples), "--code-eval-workers", "16",
            "--code-memory", "64g", "--code-eval-timeout", "14400", "--output-dir", str(output)]
        rc = self.run_logged(command, parent / (name + "_score.log"),
                             model=model, mode=mode, dataset=name, kind="scoring")
        if rc != 0 or not (output / "summary.json").exists():
            raise RuntimeError(f"Scoring failed: {output}, rc={rc}")
        summary = json.loads((output / "summary.json").read_text())
        self.event("scoring_complete", model=model, mode=mode, dataset=name,
                   metrics={k: v for k, v in summary["metrics"].items() if k != "detail"})
        return output

    def aggregate(self):
        rows = []
        for parent in sorted(self.out.glob("*/thinking_*")):
            for name in ("humaneval_plus", "mbpp_plus", "lcb_v5", "lcb_v6"):
                if self.lcb_only and not name.startswith("lcb_"):
                    continue
                summary = json.loads((parent / name / "summary.json").read_text())
                samples = read_rows(parent / name / "samples.jsonl")
                results = json.loads((parent / name / "results.json").read_text())["results"]
                scopes = [(name, samples, results)]
                if name == "lcb_v6":
                    new_manifest = json.loads((parent / "lcb_v6_new_generation/manifest.json").read_text())
                    new_ids = set(new_manifest["selected_tasks"])
                    scopes.append(("lcb_v6_new175", [r for r in samples if r["task_id"] in new_ids],
                                   [r for r in results if r["task_id"] in new_ids]))
                for dataset, selected, selected_results in scopes:
                    plus = dataset in ("humaneval_plus", "mbpp_plus")
                    passed = sum(r["plus_passed"] if plus else r["passed"][0] for r in selected_results)
                    n = len(selected)
                    assert n == len(selected_results)
                    tokens = [(r.get("usage") or {}).get("completion_tokens") for r in selected]
                    tokens = [t for t in tokens if t is not None]
                    row = {"model": parent.parent.name, "thinking": parent.name.removeprefix("thinking_"),
                           "dataset": dataset, "tasks": n, "passed": passed, "pass@1": passed / n,
                           "length_limit_count": sum(r.get("finish_reason") == "length" for r in selected),
                           "generation_errors": sum(bool(r.get("error")) for r in selected),
                           "mean_completion_tokens": sum(tokens) / len(tokens) if tokens else None,
                           "tokens_recorded": len(tokens), "source": str(parent / name / "summary.json")}
                    if dataset == name:
                        official = summary["metrics"]["plus"]["pass@1"] if plus else summary["metrics"]["pass@1"]
                        assert abs(row["pass@1"] - official) < 1e-8
                    rows.append(row)
        (self.out / "comparison.json").write_text(json.dumps(rows, indent=2))
        lines = ["# Cleaned TACO final-checkpoint coding evaluation", "",
                 f"All models: global_step_66; DP=8 / TP=1; {self.sampling}; max completion {self.max_tokens}; same benchmark prompts.",
                 f"Sampling: {json.dumps(self.sampling_config, sort_keys=True)}; one sample per task.",
                 "LCB v6 reuses identical v5 tasks plus 175 new tasks. Plus scores are enhanced-test pass@1.", "",
                 "| Model | Thinking | Dataset | Passed / total | Pass@1 | Mean completion tokens | Length limit |",
                 "|---|---|---|---:|---:|---:|---:|"]
        for r in rows:
            token_str = f"{r['mean_completion_tokens']:.1f}" if r["mean_completion_tokens"] is not None else "N/A"
            lines.append(f"| {r['model']} | {r['thinking']} | {r['dataset']} | {r['passed']}/{r['tasks']} | "
                         f"{r['pass@1']:.4f} | {token_str} | {r['length_limit_count']} |")
        (self.out / "comparison.md").write_text("\n".join(lines) + "\n")
        self.event("suite_complete", comparison=str(self.out / "comparison.md"))

    def run(self):
        metadata = {"runs": RUNS, "thinking_policy": self.args.thinking, "dp": 8, "tp": 1,
                    "max_tokens": self.max_tokens, "temperature": self.sampling_config["temperature"],
                    "top_p": self.sampling_config["top_p"], "seed": 42,
                    "generation_datasets_sequential": True, "cpu_scoring_concurrency": 2, "models": {}}
        if self.lcb_only or self.max_model_len != 32768:
            metadata.update(lcb_only=self.lcb_only, max_model_len=self.max_model_len)
        if self.sampling != "greedy":
            metadata.update(sampling_profile=self.sampling, sampling_parameters=self.sampling_config)
        for label, run in RUNS.items():
            source = self.root / "checkpoints/self_evolver_opsd_3way" / run
            assert (source / "latest_checkpointed_iteration.txt").read_text().strip() == "66"
            model = self.root / "workspace/eval_models/cleaned_taco_step66" / run
            weights = sorted(model.glob("*.safetensors"))
            assert weights, f"Missing merged weights: {model}"
            metadata["models"][label] = {"checkpoint": str(source / "global_step_66/actor"),
                                        "model_path": str(model),
                                        "weights": {p.name: {"bytes": p.stat().st_size,
                                                              "sha256": hashlib.file_digest(p.open("rb"), "sha256").hexdigest()}
                                                    for p in weights}}
        manifest_path = self.out / "suite_manifest.json"
        if manifest_path.exists():
            assert json.loads(manifest_path.read_text()) == metadata, "Resume configuration/weights changed"
        else:
            manifest_path.write_text(json.dumps(metadata, indent=2))
        with concurrent.futures.ThreadPoolExecutor(max_workers=2) as pool:
            try:
                for label, run in RUNS.items():
                    if label not in self.args.models:
                        continue
                    model = Path(metadata["models"][label]["model_path"])
                    modes = (["on", "off"] if self.args.thinking == "both" else
                             ["on" if label == "on_on" else "off"] if self.args.thinking == "training" else
                             [self.args.thinking])
                    expected_suites = {"humaneval_plus": 164, "mbpp_plus": 378, "lcb_v5": 880, "lcb_v6": 1055}
                    if self.lcb_only:
                        expected_suites = {k: v for k, v in expected_suites.items() if k.startswith("lcb_")}
                    completed = [self.out / label / ("thinking_" + mode) / name / "summary.json"
                                 for mode in modes for name in expected_suites]
                    if self.args.resume and all(path.exists() for path in completed):
                        for path in completed:
                            summary = json.loads(path.read_text())
                            assert summary["tasks"] == expected_suites[path.parent.name]
                            assert summary["model"] == label and not summary["subset"]
                            manifest = json.loads((path.parent / "manifest.json").read_text())
                            self.check_sampling(manifest)
                            assert manifest["thinking"] == path.parent.parent.name.removeprefix("thinking_")
                        self.event("completed_model_reused", model=label)
                        continue
                    self.start_server(model, label)
                    for mode in modes:
                        parent = self.out / label / ("thinking_" + mode)
                        parent.mkdir(parents=True, exist_ok=self.args.resume)
                        v5_generation = None
                        datasets = [
                            ("humaneval_plus", ["--benchmark", "humaneval+"], 164),
                            ("mbpp_plus", ["--benchmark", "mbpp+"], 378),
                            ("lcb_v5", ["--benchmark", "livecodebench", "--lcb-release", "v5"], 880),
                            ("lcb_v6_new", ["--benchmark", "livecodebench", "--lcb-release", "v6",
                                            "--tasks", "880-1054"], 175),
                        ]
                        for name, extra, count in datasets:
                            if self.lcb_only and not name.startswith("lcb_"):
                                continue
                            scored = parent / name / "summary.json"
                            if self.args.resume and name != "lcb_v6_new" and scored.exists():
                                summary = json.loads(scored.read_text())
                                manifest = json.loads((scored.parent / "manifest.json").read_text())
                                self.check_sampling(manifest)
                                assert summary["tasks"] == count and summary["model"] == label
                                assert manifest["thinking"] == mode and not summary["subset"]
                                if name == "lcb_v5":
                                    v5_generation = Path(manifest["samples_source"]).parent
                                self.event("scored_dataset_reused", model=label, mode=mode, dataset=name)
                                continue
                            output = self.generate(label, mode, name, extra, parent, count)
                            if name == "lcb_v5":
                                v5_generation = output
                            if name != "lcb_v6_new":
                                self.futures.append(pool.submit(self.score, label, mode, name, extra,
                                                                output / "samples.jsonl", parent))
                            else:
                                merged = parent / "lcb_v6_merged.samples.jsonl"
                                assert v5_generation is not None
                                merge_lcb(v5_generation, output, merged)
                                self.futures.append(pool.submit(self.score, label, mode, "lcb_v6",
                                                                ["--benchmark", "livecodebench", "--lcb-release", "v6"],
                                                                merged, parent))
                    self.stop_server()
            finally:
                self.stop_server()
            for future in self.futures:
                future.result()
        self.aggregate()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", default="/disk3/self_evolver")
    parser.add_argument("--output-root", required=True)
    parser.add_argument("--thinking", choices=("on", "off", "training", "both"), default="off")
    parser.add_argument("--port", type=int, default=9400)
    parser.add_argument("--max-tokens", type=int, default=16384)
    parser.add_argument("--max-model-len", type=int, default=32768)
    parser.add_argument("--lcb-only", action="store_true", help="Only evaluate LCB v5 and cumulative v6.")
    parser.add_argument("--sampling", choices=("greedy", "coding"), default="greedy",
                        help="coding: temperature .6, top_p .95, top_k 20, min_p 0, presence 0, repetition 1, seed 42.")
    parser.add_argument("--resume", action="store_true",
                        help="Reuse fully scored models after verifying the original configuration and weight hashes.")
    parser.add_argument("--models", nargs="+", choices=list(RUNS), default=list(RUNS),
                        help="Select unfinished models when continuing a suite; aggregation still includes all models.")
    args = parser.parse_args()
    suite = Suite(args)
    try:
        suite.run()
    except Exception as exc:
        suite.event("suite_failed", error=f"{type(exc).__name__}: {exc}")
        raise


if __name__ == "__main__":
    main()
