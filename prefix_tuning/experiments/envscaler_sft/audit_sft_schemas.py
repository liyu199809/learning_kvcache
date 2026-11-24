"""Match approved training records to archived environment schema metadata."""
from collections import Counter, defaultdict
import hashlib
import json
from pathlib import Path
import re

import pyarrow.parquet as pq

from prepare import APPROVED, OFFENSIVE, OUT, SOURCE, dump, lines

REPO = Path("/host-ssd/learning-kvcache-agent-world-pr")
ARCHIVE = REPO / "traj_data/opsd_mixed_1625_v1/train.parquet"
CAPTURE = REPO / "outputs"
DOMAINS = {"awm": "awm", "envscaler_rl": "envscaler"}
EPISODE = re.compile(r"^(.*)-(\d+)-[0-9a-f-]{36}$")
TAU2 = re.compile(r"airline|retail", re.I)


def canonical_tools(tools):
    return [tool if "function" in tool else {"type": "function", "function": {
        "name": tool["name"], "description": tool.get("description", ""),
        "parameters": tool["inputSchema"],
    }} for tool in tools]


def digest(tools):
    return hashlib.sha256(json.dumps(tools, sort_keys=True, ensure_ascii=False).encode()).hexdigest()


def schema_index():
    schema_rows = pq.read_table(ARCHIVE, columns=["data_source", "scenario", "task_idx", "tools"])
    index = {}
    for row in schema_rows.to_pylist():
        if row["data_source"] in DOMAINS:
            key = DOMAINS[row["data_source"]], row["scenario"], row["task_idx"]
            if key in index:
                raise ValueError(f"Duplicate metadata for {key}")
            index[key] = canonical_tools(json.loads(row["tools"]))
    return index


def parameters_match(messages, tools):
    schemas = {tool["function"]["name"]: tool["function"]["parameters"] for tool in tools}
    if len(schemas) != len(tools):
        return False, False
    uses_mcp_alias = False
    for message in messages:
        for call in message.get("tool_calls") or []:
            fn = call.get("function") or {}
            name = fn.get("name", "")
            args = fn.get("arguments")
            if isinstance(args, str):
                try:
                    args = json.loads(args)
                except ValueError:
                    return False, uses_mcp_alias
            if name not in schemas and name.startswith("mcp__episode__"):
                name = name.removeprefix("mcp__episode__")
                uses_mcp_alias = True
            schema = schemas.get(name)
            if schema is None or not isinstance(args, dict):
                return False, uses_mcp_alias
            if not set(args) <= set(schema.get("properties", {})):
                return False, uses_mcp_alias
            if not set(schema.get("required", [])) <= set(args):
                return False, uses_mcp_alias
    return True, uses_mcp_alias


def audit_record(source, key, messages, tools, origin, environments, stats):
    if not tools:
        return {"status": "tool_schema_missing"}
    stats[source]["schema_restored"] += 1
    matched, uses_alias = parameters_match(messages, tools)
    row = {"schema_source": origin, "schema_sha256": digest(tools), "mcp_alias_used": uses_alias}
    if not matched:
        row["status"] = "tool_schema_mismatch"
    elif source != "agent_world" and (TAU2.search(key) or any(
        TAU2.search(tool["function"]["name"]) for tool in tools
    )):
        row["status"] = "tau2_domain"
    elif OFFENSIVE.search(json.dumps([messages, tools], ensure_ascii=False)):
        row["status"] = "safety_screen"
    else:
        row["status"] = "eligible_before_length_and_dedup"
        environments[source][key] += 1
    return row


def main():
    index = schema_index()
    stats = defaultdict(Counter)
    environments = defaultdict(Counter)
    audit = []
    for number, row in enumerate(lines(SOURCE), 1):
        meta = row["task_source"]
        source = meta["dataset"]
        if source not in DOMAINS.values() or meta["status"] != "scored" or not meta["is_canonical"]:
            continue
        if row["rounds_rewards"][0] != 1:
            continue
        scenario, number_in_environment = EPISODE.fullmatch(meta["trajectory"].rsplit("/", 1)[-1]).groups()
        prefix = row["messages"][:2]
        for message in row["messages"][2:]:
            if message["role"] == "user":
                break
            prefix.append(message)
        captured = CAPTURE / meta["trajectory"] / "tools.json"
        if captured.is_file():
            tools = canonical_tools(json.loads(captured.read_text()))
            origin = str(captured)
        else:
            tools = index.get((source, scenario, int(number_in_environment)))
            origin = str(ARCHIVE) + "#tools"
        result = audit_record(source, scenario, prefix, tools, origin, environments, stats)
        stats[source][result["status"]] += 1
        audit.append({"source": source, "source_line": number, "environment": scenario, **result})

    for number, row in enumerate(lines(APPROVED), 1):
        source = "agent_world"
        result = audit_record(source, row["scenario"], row["prompt"] + row["continuation"],
                              canonical_tools(row["tools"]), str(APPROVED) + "#tools",
                              environments, stats)
        stats[source][result["status"]] += 1
        audit.append({"source": source, "source_line": number, "environment": row["scenario"], **result})

    summary = dict(stats)
    for source, cap in (("awm", 1), ("envscaler", 4)):
        summary[source]["environments"] = len(environments[source])
        summary[source]["capacity_before_length_and_dedup"] = sum(
            min(cap, count) for count in environments[source].values()
        )
    dump(OUT / "sft-schema-audit.jsonl", audit)
    (OUT / "sft-schema-audit-summary.json").write_text(json.dumps(summary, indent=2) + "\n")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
