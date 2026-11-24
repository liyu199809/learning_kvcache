"""Build next-expert-turn SFT rows from the two frozen trajectory sources."""
from collections import Counter
import hashlib
import json
import math
from pathlib import Path
import re

OUT = Path(__file__).resolve().parent
ROOT = OUT.parent
ADVICE = ROOT / "advice-ab"
DECISION = ADVICE / "agent-world-scale-20261122" / "decision8"
SOURCE = ROOT / "q35_v1_sft_all.jsonl"
APPROVED = DECISION / "agent-world-approved-257.jsonl"
OFFENSIVE = re.compile(
    r"\b(?:nmap|gobuster|nikto|sqlmap|metasploit|masscan|ffuf|hydra|dirb(?:uster)?|"
    r"exploit\w*|vulnerabilit\w*|port.scan\w*|pen.?test\w*|penetration.test\w*|"
    r"brute.force\w*|reverse.shell|shellcode|keylog\w*|ransomware|malware|"
    r"privilege.escalation|payload.inject\w*|code.inject\w*|command.inject\w*|"
    r"unauthorized.access|credential.stuffing|password.crack\w*|subdomain.enumerat\w*|"
    r"dns.reconnaissance|security.scan\w*|ssl.scan\w*|"
    r"phish\w*|botnet|prompt.injection|intrusion|zero.day|bypass\w*|attack\w*|"
    r"bioweapon\w*|pathogen\w*|viral.vector\w*|virus.engineer\w*|"
    r"virolog\w*|dna.synth\w*|gain.of.function|virulence|biological.agent\w*|"
    r"anthrax|smallpox|ricin|botulin\w*|viral.genome\w*|"
    r"weapon\w*|firearm\w*|explosiv\w*|bomb\w*|assassinat\w*|"
    r"shooting|detonat\w*|sabotag\w*)\b", re.I,
)
ROLE_MARKERS = re.compile(r"<\|[^\n]*?\|>|</?(?:think|tool_call|tool_response|function)(?:[=>\s])", re.I)


def lines(path):
    with path.open() as stream:
        for line in stream:
            yield json.loads(line)


def canonical_question(messages):
    text = next(m["content"] for m in messages if m["role"] == "user")
    if not isinstance(text, str) or not text.strip():
        raise ValueError("empty or nontext question")
    return " ".join(text.casefold().split())


def key(messages):
    return hashlib.sha256(canonical_question(messages).encode()).hexdigest()


def dump(path, rows):
    temporary = path.with_suffix(".tmp")
    with temporary.open("w") as stream:
        for row in rows:
            stream.write(json.dumps(row, ensure_ascii=False) + "\n")
    temporary.replace(path)


def load_approved():
    episodes = {}
    for batch in ("multiturn-v2", "multiturn-pilot"):
        episodes[batch] = {(r["scenario"], r["task_idx"]): r for r in lines(ADVICE / batch / "episodes.jsonl")}
    for row in lines(APPROVED):
        batch = row["source_batch"]
        trace = row.get("trace") or episodes[batch][row["scenario"], row["task_idx"]]["trace"]
        folder = DECISION / batch if batch == "recovery-pilot" else ADVICE / batch
        events = list(lines(folder / trace / "events.jsonl"))
        matches = [i for i, m in enumerate(events) if i > 1 and m["role"] == "user"
                   and m["content"] == row["teacher_prompt"]]
        if len(matches) != 1:
            raise ValueError(f'Ambiguous archived advice for {row["candidate"]}')
        yield row, events, matches[0]


def example(source, ident, domain, history, target, group, round_number, before, after):
    if len(history) < 3 or history[0]["role"] != "system" or history[1]["role"] != "user":
        return None
    if history[-1]["role"] not in {"assistant", "tool"} or not isinstance(target, str):
        return None
    if not target.strip() or ROLE_MARKERS.search(target):
        return None
    return {"source": source, "id": ident, "domain": domain, "group": group,
            "history": history, "target": target.strip(),
            "round": round_number, "before_score": before, "after_score": after}


def main():
    stats = Counter()
    splits = {"train": [], "validation": []}
    audited = list(load_approved())
    approved_groups = set()
    selected = set()

    def accept(source, ident, domain, events, index, round_number, before, after):
        group = key(events)
        row = example(source, ident, domain, events[:index], events[index]["content"], group, round_number, before, after)
        if row is None:
            stats[source, "invalid_experience"] += 1
            return
        fingerprint = hashlib.sha256(json.dumps([row["history"], row["target"]], sort_keys=True).encode()).digest()
        if fingerprint in selected:
            stats[source, "duplicate"] += 1
            return
        selected.add(fingerprint)
        split = "validation" if int(group[:8], 16) % 10 == 0 else "train"
        splits[split].append(row)
        stats[source, split] += 1
        return True

    for row, events, index in audited:
        source = "approved_257"
        stats[source, "candidate_rounds"] += 1
        before, after = row["before_score"], row["after_score"]
        if before is not None and after is not None and after > before:
            stats[source, "strict_gain"] += 1
        else:
            stats[source, "no_gain"] += 1
        if OFFENSIVE.search(json.dumps(events[:index + 1], ensure_ascii=False)):
            stats[source, "screened_rounds"] += 1
            continue
        if accept(source, row["candidate"], "agent_world", events, index, row["round"], before, after):
            approved_groups.add(key(events))

    for line_number, row in enumerate(lines(SOURCE), 1):
        metadata = row["task_source"]
        source = "q35_v1"
        events = row["messages"]
        turns = [i for i, m in enumerate(events) if i > 1 and m["role"] == "user"]
        rewards = row["rounds_rewards"]
        stats[source, "trajectories"] += 1
        if len(rewards) != len(turns) + 1 or any(not math.isfinite(v) for v in rewards):
            raise ValueError(f"Unaligned reward history at source line {line_number}")
        stats[source, "candidate_rounds"] += len(turns)
        improvements = [(number, index, rewards[number - 1], rewards[number])
                        for number, index in enumerate(turns, 1) if rewards[number] > rewards[number - 1]]
        stats[source, "strict_gain"] += len(improvements)
        stats[source, "no_gain"] += len(turns) - len(improvements)
        if not improvements:
            continue
        if not metadata.get("is_canonical"):
            stats[source, "noncanonical_gain"] += len(improvements)
            continue
        if OFFENSIVE.search(json.dumps(events, ensure_ascii=False)):
            stats[source, "screened_rounds"] += len(improvements)
            continue
        if key(events) in approved_groups:
            stats[source, "overlapping_gain"] += len(improvements)
            continue
        for number, index, before, after in improvements:
            accept(source, f"{line_number}:{index}", metadata["dataset"], events, index, number, before, after)

    groups = {split: {r["group"] for r in rows} for split, rows in splits.items()}
    assert not (groups["train"] & groups["validation"])
    for split, rows in splits.items():
        dump(OUT / f"{split}.jsonl", rows)
    for source, name in (("approved_257", "approved-eligible"), ("q35_v1", "q35_v1-improved")):
        dump(OUT / f"{name}.jsonl",
             ({"split": split, **row} for split, rows in splits.items() for row in rows if row["source"] == source))
    (OUT / "approved_257-improved.jsonl").unlink(missing_ok=True)
    counts = {source: {metric: count for (name, metric), count in stats.items() if name == source}
              for source in ("approved_257", "q35_v1")}
    counts["questions"] = {k: len(v) for k, v in groups.items()}
    (OUT / "data-summary.json").write_text(json.dumps(counts, ensure_ascii=False, indent=2) + "\n")
    print(json.dumps(counts, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
