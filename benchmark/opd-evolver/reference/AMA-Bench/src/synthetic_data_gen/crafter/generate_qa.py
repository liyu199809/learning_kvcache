import argparse
import json
import sys
import uuid
import re

# ---------------------------------------------------------------------------
# Layer 1: Game mechanics constants & observation parser
# ---------------------------------------------------------------------------

DIRECTION_MAP = {
    "Move North": "north",
    "Move South": "south",
    "Move West": "west",
    "Move East": "east",
}

OPPOSITE_MOVE = {
    "Move North": "Move South",
    "Move South": "Move North",
    "Move West": "Move East",
    "Move East": "Move West",
}

MOVE_ACTIONS = set(DIRECTION_MAP.keys())

CRAFT_ACTIONS = {
    "Make Wood Pickaxe", "Make Stone Pickaxe", "Make Iron Pickaxe",
    "Make Wood Sword", "Make Stone Sword", "Make Iron Sword",
}

PLACE_ACTIONS = {"Place Stone", "Place Table", "Place Furnace", "Place Plant"}

RESOURCE_TYPES = {"tree", "stone", "coal", "iron", "diamond"}
ENEMY_TYPES = {"zombie", "skeleton", "cow"}
TERRAIN_TYPES = {"grass", "path", "sand", "water", "lava"}

TOOL_REQUIRED = {
    "tree": None,
    "stone": "Wood Pickaxe",
    "coal": "Wood Pickaxe",
    "iron": "Stone Pickaxe",
    "diamond": "Iron Pickaxe",
}

RESOURCE_PRODUCT = {
    "tree": "wood",
    "stone": "stone",
    "coal": "coal",
    "iron": "iron",
    "diamond": "diamond",
}

CRAFT_PREREQUISITES = {
    "Make Wood Pickaxe": {
        "station": "table", "materials": ["wood"],
        "product": "Wood Pickaxe",
    },
    "Make Stone Pickaxe": {
        "station": "table", "materials": ["wood", "stone"],
        "product": "Stone Pickaxe",
    },
    "Make Iron Pickaxe": {
        "station": "table", "materials": ["wood", "stone", "iron"],
        "product": "Iron Pickaxe",
    },
    "Make Wood Sword": {
        "station": "table", "materials": ["wood"],
        "product": "Wood Sword",
    },
    "Make Stone Sword": {
        "station": "table", "materials": ["wood", "stone"],
        "product": "Stone Sword",
    },
    "Make Iron Sword": {
        "station": "table", "materials": ["wood", "stone", "iron"],
        "product": "Iron Sword",
    },
}

PLACE_ITEM = {
    "Place Table": "table",
    "Place Furnace": "furnace",
    "Place Plant": "plant",
    "Place Stone": "stone",
}

TOOL_TIER = {
    "Wood Pickaxe": 1, "Stone Pickaxe": 2, "Iron Pickaxe": 3,
    "Wood Sword": 1, "Stone Sword": 2, "Iron Sword": 3,
}

ACHIEVEMENT_ACTIONS = {
    "Place Table": "place_table",
    "Place Furnace": "place_furnace",
    "Place Plant": "place_plant",
    "Place Stone": "place_stone",
    "Make Wood Pickaxe": "make_wood_pickaxe",
    "Make Stone Pickaxe": "make_stone_pickaxe",
    "Make Iron Pickaxe": "make_iron_pickaxe",
    "Make Wood Sword": "make_wood_sword",
    "Make Stone Sword": "make_stone_sword",
    "Make Iron Sword": "make_iron_sword",
}

HARVEST_ACHIEVEMENT = {
    "tree": "collect_wood",
    "stone": "collect_stone",
    "coal": "collect_coal",
    "iron": "collect_iron",
    "diamond": "collect_diamond",
}

DEFEAT_ACHIEVEMENT = {
    "zombie": "defeat_zombie",
    "skeleton": "defeat_skeleton",
    "cow": "eat_cow",
}


def parse_observation(obs_text):
    items = []
    face = None
    for line in obs_text.split("\n"):
        line = line.strip()
        m = re.match(r"- (\w+) (\d+) steps? to your (.+)", line)
        if m:
            items.append({"type": m.group(1), "dist": int(m.group(2)), "dir": m.group(3)})
        fm = re.match(r"You face (\w+) at your front", line)
        if fm:
            face = fm.group(1)
    return {"items": items, "face": face}


def _last_move_direction(trajectory, step_idx):
    for i in range(step_idx - 1, -1, -1):
        if trajectory[i]["action"] in DIRECTION_MAP:
            return DIRECTION_MAP[trajectory[i]["action"]]
    return None


# ---------------------------------------------------------------------------
# Layer 2: Event detectors
# ---------------------------------------------------------------------------

def detect_harvests(trajectory):
    harvests = []
    for i in range(len(trajectory) - 1):
        if trajectory[i]["action"] != "Do":
            continue
        obs_before = parse_observation(trajectory[i]["observation"])
        obs_after = parse_observation(trajectory[i + 1]["observation"])
        face = obs_before["face"]
        if not face or face in TERRAIN_TYPES:
            continue

        faced_before = [it for it in obs_before["items"] if it["type"] == face and it["dist"] == 1]
        faced_after = [it for it in obs_after["items"] if it["type"] == face and it["dist"] == 1]

        success = bool(faced_before) and not bool(faced_after)
        is_resource = face in RESOURCE_TYPES
        is_enemy = face in ENEMY_TYPES

        harvests.append({
            "step": i,
            "target": face,
            "success": success,
            "is_resource": is_resource,
            "is_enemy": is_enemy,
            "obs_before": obs_before,
            "obs_after": obs_after,
        })
    return harvests


def detect_approach_sequences(trajectory):
    sequences = []
    i = 0
    while i < len(trajectory):
        if trajectory[i]["action"] not in MOVE_ACTIONS:
            i += 1
            continue
        start = i
        while i < len(trajectory) and trajectory[i]["action"] in MOVE_ACTIONS:
            i += 1
        if i < len(trajectory) and trajectory[i]["action"] == "Do":
            moves = [trajectory[j]["action"] for j in range(start, i)]
            sequences.append({
                "start_step": start,
                "do_step": i,
                "moves": moves,
                "num_moves": len(moves),
            })
    return sequences


def detect_craft_place_events(trajectory):
    events = []
    for i in range(len(trajectory)):
        action = trajectory[i]["action"]
        if action not in CRAFT_ACTIONS and action not in PLACE_ACTIONS:
            continue
        obs_before = parse_observation(trajectory[i]["observation"])

        obs_after = None
        if i + 1 < len(trajectory):
            obs_after = parse_observation(trajectory[i + 1]["observation"])

        if action in PLACE_ACTIONS:
            placed = PLACE_ITEM[action]
            success = False
            if obs_after:
                success = any(it["type"] == placed and it["dist"] <= 1 for it in obs_after["items"])
            events.append({
                "step": i,
                "action": action,
                "kind": "place",
                "placed": placed,
                "success": success,
                "obs_before": obs_before,
            })
        else:
            station_nearby = any(
                it["type"] == "table" and it["dist"] <= 1
                for it in obs_before["items"]
            )
            events.append({
                "step": i,
                "action": action,
                "kind": "craft",
                "product": CRAFT_PREREQUISITES[action]["product"],
                "station_nearby": station_nearby,
                "obs_before": obs_before,
            })
    return events


def detect_combat(trajectory):
    combats = []
    i = 0
    while i < len(trajectory):
        if trajectory[i]["action"] != "Do":
            i += 1
            continue
        obs = parse_observation(trajectory[i]["observation"])
        face = obs["face"]
        if not face or face not in ENEMY_TYPES:
            i += 1
            continue
        enemy = face
        start = i
        steps = [i]
        i += 1
        while i < len(trajectory) and trajectory[i]["action"] == "Do":
            o = parse_observation(trajectory[i]["observation"])
            if o["face"] != enemy:
                break
            steps.append(i)
            i += 1

        after_obs = parse_observation(trajectory[i]["observation"]) if i < len(trajectory) else None
        enemy_gone = True
        if after_obs:
            enemy_gone = not any(it["type"] == enemy and it["dist"] <= 1 for it in after_obs["items"])
        combats.append({
            "steps": steps,
            "start_step": start,
            "end_step": steps[-1],
            "enemy": enemy,
            "num_hits": len(steps),
            "enemy_defeated": enemy_gone,
        })
    return combats


def detect_water_interactions(trajectory):
    interactions = []
    for i in range(len(trajectory)):
        if trajectory[i]["action"] != "Do":
            continue
        obs = parse_observation(trajectory[i]["observation"])
        if obs["face"] == "water":
            interactions.append({"step": i, "obs_before": obs})
    return interactions


def detect_sleep_sequences(trajectory):
    sequences = []
    i = 0
    while i < len(trajectory):
        if trajectory[i]["action"] != "Sleep":
            i += 1
            continue
        start = i
        while i < len(trajectory) and trajectory[i]["action"] in ("Sleep", "Noop"):
            i += 1
        sequences.append({
            "start_step": start,
            "end_step": i - 1,
            "num_steps": i - start,
            "wake_step": i if i < len(trajectory) else None,
        })
    return sequences


def detect_observation_changes(trajectory):
    changes = []
    for i in range(1, len(trajectory)):
        if trajectory[i]["action"] not in MOVE_ACTIONS:
            continue
        prev_obs = parse_observation(trajectory[i - 1]["observation"])
        curr_obs = parse_observation(trajectory[i]["observation"])
        prev_types = {it["type"] for it in prev_obs["items"]}
        curr_types = {it["type"] for it in curr_obs["items"]}
        new_types = curr_types - prev_types
        gone_types = prev_types - curr_types
        if len(prev_obs["items"]) == 0 and len(curr_obs["items"]) >= 2:
            changes.append({
                "step": i, "action": trajectory[i]["action"],
                "kind": "empty_to_rich",
                "new_types": new_types,
                "items_before": len(prev_obs["items"]),
                "items_after": len(curr_obs["items"]),
            })
        elif len(curr_obs["items"]) == 0 and len(prev_obs["items"]) >= 2:
            changes.append({
                "step": i, "action": trajectory[i]["action"],
                "kind": "rich_to_empty",
                "gone_types": gone_types,
                "items_before": len(prev_obs["items"]),
                "items_after": len(curr_obs["items"]),
            })
        elif len(new_types) >= 2:
            changes.append({
                "step": i, "action": trajectory[i]["action"],
                "kind": "new_entities",
                "new_types": new_types,
                "items_before": len(prev_obs["items"]),
                "items_after": len(curr_obs["items"]),
            })
    return changes


def detect_oscillations(trajectory, window=5):
    """Find short windows of move actions that contain opposing-direction pairs.

    Used by type-D to ask 'which actions were forward-progress vs counter-productive'.
    Returns windows of 4-6 consecutive moves where at least one opposing pair appears.
    """
    oscillations = []
    i = 0
    while i < len(trajectory):
        if trajectory[i]["action"] not in MOVE_ACTIONS:
            i += 1
            continue
        run_start = i
        while i < len(trajectory) and trajectory[i]["action"] in MOVE_ACTIONS:
            i += 1
        run_end = i - 1
        run_len = run_end - run_start + 1
        if run_len < 4:
            continue
        moves = [trajectory[j]["action"] for j in range(run_start, run_end + 1)]
        has_opposing = False
        for j in range(len(moves) - 1):
            for k in range(j + 1, len(moves)):
                if OPPOSITE_MOVE.get(moves[j]) == moves[k]:
                    has_opposing = True
                    break
            if has_opposing:
                break
        if has_opposing:
            oscillations.append({
                "start_step": run_start,
                "end_step": run_end,
                "moves": moves,
                "num_moves": run_len,
            })
    return oscillations


def infer_achievements(events):
    unlocked = set()
    first_time = []
    repeated = []

    harvest_steps = {}
    for h in events["successful_harvests"]:
        harvest_steps[h["step"]] = h["target"]
    combat_end_steps = {}
    for c in events["combats"]:
        if c["enemy_defeated"]:
            combat_end_steps[c["end_step"]] = c["enemy"]
    craft_place_steps = {}
    for ev in events["craft_place"]:
        craft_place_steps[ev["step"]] = ev["action"]
    water_steps = {w["step"] for w in events["water_interactions"]}
    wake_steps = set()
    for s in events["sleep_sequences"]:
        if s["wake_step"] is not None:
            wake_steps.add(s["wake_step"])

    all_steps = set(harvest_steps) | set(combat_end_steps) | set(craft_place_steps) | water_steps | wake_steps
    for step in sorted(all_steps):
        achievement = None
        action = None
        if step in harvest_steps:
            target = harvest_steps[step]
            achievement = HARVEST_ACHIEVEMENT.get(target)
            action = "Do"
        elif step in combat_end_steps:
            enemy = combat_end_steps[step]
            achievement = DEFEAT_ACHIEVEMENT.get(enemy)
            action = "Do"
        elif step in craft_place_steps:
            act = craft_place_steps[step]
            achievement = ACHIEVEMENT_ACTIONS.get(act)
            action = act
        elif step in water_steps:
            achievement = "collect_drink"
            action = "Do"
        elif step in wake_steps:
            achievement = "wake_up"
            action = "wake"

        if not achievement:
            continue
        if achievement not in unlocked:
            unlocked.add(achievement)
            first_time.append({"step": step, "achievement": achievement, "action": action, "reward": 1.0})
        else:
            repeated.append({"step": step, "achievement": achievement, "action": action, "reward": 0.0})

    return {"first_time": first_time, "repeated": repeated, "unlocked": unlocked}


def build_tool_timeline(craft_events):
    tools = []
    inventory = []
    for ev in craft_events:
        if ev["kind"] == "craft":
            inventory.append(ev["product"])
            tools.append({"step": ev["step"], "product": ev["product"],
                          "inventory_after": list(inventory)})
    return tools


def analyze_trajectory(trajectory):
    """Infer interesting moments by diffing consecutive observations.

    The raw trajectory only has (turn_idx, action, observation) — no events,
    rewards, or inventory.  This function reconstructs what happened by
    comparing observations before and after each action.
    """
    harvests = detect_harvests(trajectory)
    approaches = detect_approach_sequences(trajectory)
    craft_place = detect_craft_place_events(trajectory)
    combats = detect_combat(trajectory)
    tool_timeline = build_tool_timeline(craft_place)
    water_interactions = detect_water_interactions(trajectory)
    sleep_sequences = detect_sleep_sequences(trajectory)
    observation_changes = detect_observation_changes(trajectory)
    oscillations = detect_oscillations(trajectory)

    successful_harvests = [h for h in harvests if h["success"] and h["is_resource"]]
    failed_harvests = [h for h in harvests if not h["success"] and h["is_resource"]]

    events = {
        "harvests": harvests,
        "successful_harvests": successful_harvests,
        "failed_harvests": failed_harvests,
        "approaches": approaches,
        "craft_place": craft_place,
        "combats": combats,
        "tool_timeline": tool_timeline,
        "water_interactions": water_interactions,
        "sleep_sequences": sleep_sequences,
        "observation_changes": observation_changes,
        "oscillations": oscillations,
    }
    events["achievements"] = infer_achievements(events)
    return events


# ---------------------------------------------------------------------------
# Layer 3: Question generators (4 AMA-Bench types: A/B/C/D)
#
# Every question + answer references at least 4 distinct trajectory step
# numbers — single-step or 2-step templates have been removed.  Question
# prose mirrors AMA-Bench's crafter rows (HF rows 6–11, file
# `test/open_end_qa_set.jsonl`).  Per-template AMA-Bench reference rows are
# noted in comments.
# ---------------------------------------------------------------------------

MIN_STEP_REFS = 4


def _step_refs(text):
    """Return the set of distinct step numbers referenced in `text`.

    Recognizes 'step N', 'steps N, M, K', and 'steps N-M' / 'step N to M'
    range forms (range expanded to its endpoints — not all interior steps,
    so this stays a strict count).
    """
    refs = set()
    for m in re.finditer(r"step[s]?\s+(\d+)(?:\s*[,]\s*(\d+))*", text.lower()):
        for g in m.groups():
            if g is not None:
                refs.add(int(g))
    # Pick up trailing numbers in lists like "steps 5, 7, 9, 11".
    for m in re.finditer(r"\bsteps?\b[^.]*?(\d+(?:\s*[,]\s*\d+){2,})", text.lower()):
        for n in re.findall(r"\d+", m.group(1)):
            refs.add(int(n))
    # Range forms: "steps X-Y", "step X through Y", "step X to Y".
    for m in re.finditer(
        r"step[s]?\s+(\d+)\s*(?:-|–|through|to)\s*(\d+)", text.lower()
    ):
        refs.add(int(m.group(1)))
        refs.add(int(m.group(2)))
    return refs


def _mk(question, answer, qtype):
    return {
        "question": question,
        "answer": answer,
        "type": qtype,
        "question_uuid": str(uuid.uuid4()),
    }


def _alt_move(action):
    opts = [m for m in MOVE_ACTIONS if m != action]
    return opts[0] if opts else "Move North"


def _pick(seed, options):
    """Deterministically pick one of `options` from an integer `seed`."""
    return options[seed % len(options)]


# ---------------------------------------------------------------------------
# Type A — Recall / counterfactual reasoning.
#
# AMA-Bench A questions are dominated by counterfactual reasoning: "Explain how
# X was the direct and necessary cause for this success, and what would have
# happened if the agent had performed Y instead." (See HF rows 6 qa_pair[0],
# 6 qa_pair[1], 7 qa_pair[6].)  Pure-recall questions are rarer ("What action
# did the agent perform at step X..." — row 6 qa_pair[3]).
# ---------------------------------------------------------------------------

def generate_type_a(trajectory, events):
    """Type A — Recall / counterfactual reasoning.  Every template cites >=4
    distinct trajectory steps in both question and answer."""
    candidates = []

    # A1: Approach + harvest counterfactual.  Walks the full approach
    # sequence (start..do_step) and asks what would have happened if a
    # middle move had been changed.  Step refs: every move step + do_step
    # + mid_step + 1.  Requires num_moves >= 3 (so >=4 step refs).
    # Mirrors HF row 6 qa_pair[0] (uuid 28b50395).
    for ap in events["approaches"]:
        if ap["num_moves"] < 3 or ap["num_moves"] > 8:
            continue
        do_step = ap["do_step"]
        if do_step >= len(trajectory):
            continue
        harvest = next(
            (h for h in events["successful_harvests"] if h["step"] == do_step),
            None,
        )
        if not harvest:
            continue
        target = harvest["target"]
        product = RESOURCE_PRODUCT.get(target, target)
        moves = ap["moves"]
        start = ap["start_step"]
        last_move = moves[-1]
        mid_idx = len(moves) // 2
        mid_step = start + mid_idx
        mid_move = moves[mid_idx]
        alt = _alt_move(mid_move)
        question = _pick(do_step, [
            f"At step {do_step}, the agent successfully uses the 'Do' action "
            f"to harvest a {target}, having reached this position through "
            f"{ap['num_moves']} movement actions across steps {start} to "
            f"{do_step - 1}. If the agent had instead performed '{alt}' at "
            f"step {mid_step}, what different outcome would have occurred "
            f"at step {do_step}, and why?",

            f"The agent harvests a {target} at step {do_step} after "
            f"{ap['num_moves']} preceding movement actions from step "
            f"{start} through step {do_step - 1}. If the agent had "
            f"performed '{alt}' at step {mid_step} instead of the move it "
            f"actually took, what different outcome would have followed at "
            f"step {do_step}?",

            f"Across steps {start} through {do_step}, the agent navigates "
            f"{ap['num_moves']} moves and then a successful 'Do' that "
            f"harvests a {target}. Explain how the actual move at step "
            f"{mid_step} contributed to this success, and what would have "
            f"happened at step {do_step} if the agent had performed '{alt}' "
            f"at step {mid_step} instead.",
        ])
        candidates.append(_mk(
            question,

            f"The {ap['num_moves']}-move sequence at steps {start} through "
            f"{do_step - 1} was a precise approach maneuver: each move "
            f"closed the distance to the {target}, and the final "
            f"'{last_move}' at step {do_step - 1} placed the agent directly "
            f"adjacent to and facing the {target} ('You face {target} at "
            f"your front'). The 'Do' at step {do_step} could only succeed "
            f"because of this exact positioning. If the agent had performed "
            f"'{alt}' at step {mid_step} instead of '{mid_move}', its grid "
            f"position from step {mid_step + 1} onward would have diverged "
            f"from the actual trajectory. The remaining moves at steps "
            f"{mid_step + 1} through {do_step - 1} would not have produced "
            f"the adjacency required at step {do_step - 1}, and the 'Do' "
            f"at step {do_step} would have targeted an empty tile, failing "
            f"to collect the {product}.",
            "A",
        ))

    # A2: Multi-failure tool counterfactual — repeated `Do` on a deposit
    # that the agent lacks the tool for.  Cites every failure step.
    # Requires >=3 failures (so >=3 explicit step numbers in Q, plus
    # repeated references in A → >=4 distinct refs overall).
    # Mirrors HF row 6 qa_pair[1] (uuid fb184ff3).
    failed_groups = {}
    for h in events["failed_harvests"]:
        failed_groups.setdefault(h["target"], []).append(h)
    for target, group in failed_groups.items():
        required = TOOL_REQUIRED.get(target)
        if not required or len(group) < 3:
            continue
        fail_steps = [h["step"] for h in group[:6]]
        steps_str = ", ".join(str(s) for s in fail_steps)
        first_fail = fail_steps[0]
        last_fail = fail_steps[-1]
        later_steps = ", ".join(str(s) for s in fail_steps[1:])
        question = _pick(first_fail, [
            f"In steps {steps_str}, the agent uses the 'Do' action on a "
            f"{target} deposit but fails to collect it on every attempt, "
            f"receiving no reward. If the agent's inventory had contained "
            f"a {required} before step {first_fail}, what different outcome "
            f"would have occurred across all {len(fail_steps)} attempts, "
            f"and why is this tool a critical factor?",

            f"At steps {steps_str}, the agent repeatedly attempts the 'Do' "
            f"action on a {target} deposit but fails each time. If the "
            f"agent had crafted and equipped a {required} before step "
            f"{first_fail}, what different outcome would have followed at "
            f"step {first_fail} and at the subsequent attempts?",
        ])
        candidates.append(_mk(
            question,

            f"If the agent had possessed a {required}, each of the 'Do' "
            f"actions at steps {steps_str} would have succeeded instead of "
            f"failing. After step {first_fail}, the {target} deposit would "
            f"have been removed and one unit added to inventory; the "
            f"subsequent 'Do' actions at steps {later_steps} would have "
            f"continued to yield {target} from neighboring deposits. The "
            f"persistent failure across all {len(fail_steps)} attempts "
            f"spanning steps {first_fail} through {last_fail} is direct "
            f"evidence the tool requirement was not met: Crafter's strict "
            f"tool hierarchy requires at least a {required} to mine "
            f"{target}, and lower-tier tools or bare hands have no effect.",
            "A",
        ))

    # A3: Multi-hit combat counterfactual — what if the agent had fled?
    # Requires num_hits >= 3 (so each hit step + start + post-combat = >=4).
    # Mirrors HF row 7 qa_pair[6] (zombie counterfactual).
    for c in events["combats"]:
        if c["num_hits"] < 3:
            continue
        start = c["start_step"]
        if start == 0:
            continue
        enemy = c["enemy"]
        opp_move = "Move North"
        if start > 0 and trajectory[start - 1]["action"] in MOVE_ACTIONS:
            opp_move = OPPOSITE_MOVE[trajectory[start - 1]["action"]]
        outcome = "defeated" if c["enemy_defeated"] else "failed to defeat"
        steps_str = ", ".join(str(s) for s in c["steps"])
        post_step = c["end_step"] + 1
        later_hits = ", ".join(str(s) for s in c["steps"][1:])
        question = _pick(start, [
            f"At steps {steps_str}, the agent fights a {enemy}, performing "
            f"{c['num_hits']} consecutive 'Do' actions and {outcome} it "
            f"(the {enemy} disappears by step {post_step}). If the agent "
            f"had instead performed '{opp_move}' at step {start} to flee "
            f"the encounter, what different outcome would have unfolded "
            f"across steps {start}, {c['end_step']}, and {post_step}?",

            f"The agent attacks a {enemy} {c['num_hits']} times in a row "
            f"at steps {steps_str}, finally defeating it by step "
            f"{post_step}. If the agent had performed '{opp_move}' at step "
            f"{start} instead of engaging, what would have happened to the "
            f"{enemy} and the agent across steps {start} through "
            f"{post_step}?",
        ])
        candidates.append(_mk(
            question,

            f"If the agent had performed '{opp_move}' at step {start}, it "
            f"would have moved away from the {enemy} without engaging. The "
            f"{enemy} would have remained alive throughout steps {start} "
            f"to {post_step}, unharmed and free to threaten the agent "
            f"later. The {c['num_hits']} 'Do' actions the agent actually "
            f"performed at steps {steps_str} each reduced the {enemy}'s "
            f"hidden HP; the intermediate hits at steps {later_hits} were "
            f"necessary cumulative damage, and only the final hit at step "
            f"{c['end_step']} brought HP to zero, removing the {enemy} "
            f"from the world by step {post_step}. Fleeing trades offensive "
            f"resolution for safety but leaves the threat intact.",
            "A",
        ))

    # A6: Sleep cycle counterfactual.  Cites: start, start+1, end, wake.
    # Requires sleep cycle of >=3 steps so the cited interior step is
    # meaningful.
    for s in events["sleep_sequences"]:
        if s["num_steps"] < 3 or s["wake_step"] is None:
            continue
        start = s["start_step"]
        end = s["end_step"]
        wake = s["wake_step"]
        question = _pick(start, [
            f"Between steps {start} and {end}, the agent enters a sleeping "
            f"cycle of {s['num_steps']} 'Sleep'/'Noop' actions, finally "
            f"waking at step {wake}. If the agent had broken the cycle by "
            f"performing an active 'Move' action at step {start + 1} "
            f"instead, what different outcome would have unfolded across "
            f"steps {start + 1}, {end}, and {wake}?",

            f"From step {start} through step {end}, the agent sleeps "
            f"continuously and wakes at step {wake}. If the agent had "
            f"interrupted the sleep with a 'Move' at step {start + 1}, "
            f"what different outcome would have occurred at step {wake} "
            f"compared to the actual fully-restored state?",
        ])
        candidates.append(_mk(
            question,

            f"If the agent had performed an active action at step "
            f"{start + 1}, the sleep cycle would have ended prematurely "
            f"without fully restoring energy. The remaining sleep at steps "
            f"{start + 2} through {end} would not have happened; instead "
            f"the agent would have continued depleting energy through each "
            f"subsequent active action. By step {wake}, the agent would "
            f"have been increasingly low on energy rather than fully "
            f"restored. The actual sleep cycle from step {start} to step "
            f"{end} replenished the agent's hidden energy stat, and waking "
            f"at step {wake} returned control with full energy — a "
            f"necessary prerequisite for sustained productive play.",
            "A",
        ))

    return candidates


# ---------------------------------------------------------------------------
# Type B — Causal Inference / critical-action / optimality.
#
# AMA-Bench B questions identify the critical action that enabled an outcome,
# or ask why a chosen action was strategically optimal compared to other
# moves.  Distinctive phrases: "What was the critical positioning action ..."
# (HF row 6 qa_pair[3], uuid 2fff9024) and "Why was X the only optimal action
# at Step N for achieving this reward, compared to moving in any other
# direction?" (HF row 6 qa_pair[4], uuid 430a0337).
# ---------------------------------------------------------------------------

def generate_type_b(trajectory, events):
    """Type B — Causal Inference / critical-action / optimality.  Every
    template cites >=4 distinct trajectory steps in both question and
    answer."""
    candidates = []

    # B1: Critical-positioning across the full approach.  Cites every move
    # step + do_step.  Requires num_moves >= 3 for >=4 step refs.
    # Mirrors HF row 6 qa_pair[3] (uuid 2fff9024) but extended to include
    # the full approach context rather than just the (N-1, N) pair.
    for ap in events["approaches"]:
        if ap["num_moves"] < 3 or ap["num_moves"] > 8:
            continue
        do_step = ap["do_step"]
        if do_step >= len(trajectory):
            continue
        harvest = next(
            (h for h in events["successful_harvests"] if h["step"] == do_step),
            None,
        )
        if not harvest:
            continue
        target = harvest["target"]
        moves = ap["moves"]
        start = ap["start_step"]
        last_move = moves[-1]
        approach_steps = ", ".join(str(s) for s in range(start, do_step))
        earlier_steps = approach_steps[:approach_steps.rfind(",")] or str(start)
        question = _pick(do_step, [
            f"At step {do_step}, the agent performs a 'Do' action that "
            f"successfully harvests a {target}, having approached through "
            f"{ap['num_moves']} movement actions across steps {start} "
            f"through {do_step - 1}. What was the critical positioning "
            f"action taken at step {do_step - 1} that enabled this success, "
            f"and why would the 'Do' action have failed without it?",

            f"The agent harvests a {target} at step {do_step} after a "
            f"{ap['num_moves']}-move approach spanning steps {start} "
            f"through {do_step - 1}. Within this approach, what was the "
            f"critical positioning action that enabled the success at step "
            f"{do_step}, and why would the 'Do' have failed without it?",

            f"At step {do_step}, the agent successfully harvests a "
            f"{target}. Looking at the {ap['num_moves']} preceding "
            f"movement actions across steps {start}-{do_step - 1}, what "
            f"was the critical positioning action that enabled this "
            f"success, and what would have happened without it?",
        ])
        candidates.append(_mk(
            question,

            f"The critical positioning action was '{last_move}' at step "
            f"{do_step - 1}: it was the final adjacency-creating move that "
            f"placed the agent directly adjacent to and facing the "
            f"{target} ('You face {target} at your front'). The earlier "
            f"moves at steps {earlier_steps} were necessary preparation — "
            f"they closed the bulk of the distance to the {target} — but "
            f"none of them on their own placed the agent in the harvesting "
            f"tile. According to Crafter's mechanics, the 'Do' action only "
            f"affects the single tile the agent is directly facing; "
            f"without '{last_move}' at step {do_step - 1}, the agent would "
            f"still have been one tile away after the approach, and the "
            f"'Do' at step {do_step} would have had no effect on the "
            f"{target}.",
            "B",
        ))

    # B2: "Only optimal action" framing for approach to first-time
    # achievement.  Cites all approach steps + do_step.
    # Mirrors HF row 6 qa_pair[4] (uuid 430a0337).
    first_time_steps = {ft["step"]: ft for ft in events["achievements"]["first_time"]}
    for ap in events["approaches"]:
        if ap["num_moves"] < 3 or ap["num_moves"] > 8:
            continue
        do_step = ap["do_step"]
        if do_step not in first_time_steps:
            continue
        if do_step == 0:
            continue
        do_obs = parse_observation(trajectory[do_step]["observation"])
        face = do_obs["face"]
        if not face or (face in TERRAIN_TYPES and face != "water"):
            continue
        moves = ap["moves"]
        start = ap["start_step"]
        last_move = moves[-1]
        last_dir = DIRECTION_MAP[last_move]
        approach_steps = ", ".join(str(s) for s in range(start, do_step))
        ach = first_time_steps[do_step]["achievement"]
        earlier_steps = approach_steps[:approach_steps.rfind(",")] or str(start)
        question = _pick(do_step, [
            f"The agent performs {ap['num_moves']} movement actions across "
            f"steps {approach_steps}, all yielding 0.0 reward, and then a "
            f"'Do' at step {do_step} that yields +1.0 (unlocking '{ach}'). "
            f"Why was '{last_move}' at step {do_step - 1} the only optimal "
            f"action for this reward, compared to moving in any other "
            f"direction?",

            f"At step {do_step}, the agent earns a +1.0 reward by "
            f"performing 'Do' (unlocking '{ach}'), preceded by "
            f"{ap['num_moves']} moves across steps {start} through "
            f"{do_step - 1}. Why was '{last_move}' at step {do_step - 1} "
            f"the only optimal final move for triggering this reward?",
        ])
        candidates.append(_mk(
            question,

            f"'{last_move}' at step {do_step - 1} was the only optimal "
            f"action because the 'Do' action in Crafter only works when "
            f"the agent is directly adjacent to and facing the target. "
            f"After the approach moves at steps {earlier_steps}, the agent "
            f"was positioned exactly one tile away from the {face}; the "
            f"'{last_move}' at step {do_step - 1} placed the agent facing "
            f"{last_dir}, with the {face} exactly 1 step in front. Any "
            f"other move at step {do_step - 1} would have changed the "
            f"agent's facing direction, leaving the subsequent 'Do' at "
            f"step {do_step} targeting an empty tile and yielding no "
            f"reward.",
            "B",
        ))

    # B3: Repeated-achievement decomposition — first time vs many subsequent.
    # Cites first_step + every repeated step.  Requires >=3 repetitions.
    rep_by_ach = {}
    for rp in events["achievements"]["repeated"]:
        rep_by_ach.setdefault(rp["achievement"], []).append(rp["step"])
    for ach, rep_steps in rep_by_ach.items():
        if len(rep_steps) < 3:
            continue
        first_step = next(
            (ft["step"] for ft in events["achievements"]["first_time"]
             if ft["achievement"] == ach), None,
        )
        if first_step is None:
            continue
        rep_steps_show = rep_steps[:6]
        steps_str = ", ".join(str(s) for s in rep_steps_show)
        question = _pick(first_step, [
            f"The '{ach}' achievement was unlocked at step {first_step} "
            f"with a +1.0 reward. The agent then performed the same "
            f"successful action again at steps {steps_str}, but each "
            f"repetition yielded 0.0 reward. Why does only the action at "
            f"step {first_step} earn the reward, even though the actions "
            f"at steps {steps_str} produced identical game effects?",

            f"At step {first_step}, the agent earns a +1.0 reward for "
            f"unlocking '{ach}'; subsequent identical actions at steps "
            f"{steps_str} all yield 0.0. Why is only the first "
            f"occurrence rewarded?",
        ])
        candidates.append(_mk(
            question,

            f"Crafter awards +1.0 only the first time each achievement is "
            f"unlocked. The action at step {first_step} was the first "
            f"occurrence of '{ach}', triggering the achievement and "
            f"earning the reward. The subsequent successful actions at "
            f"steps {steps_str} each produced the same game effect "
            f"(resource collected, entity defeated, etc.) but the '{ach}' "
            f"flag was already set after step {first_step}, so the "
            f"achievement could not be re-awarded. The 0.0 reward at each "
            f"of steps {steps_str} reflects this hidden achievement-"
            f"tracking state, not any failure of the action itself.",
            "B",
        ))

    # B4: Crafting prerequisite chain — wood + (other materials) + table +
    # craft.  Cites all relevant prereq harvest steps + table + craft.
    # Requires >=4 cited steps total.
    place_tables = [
        e for e in events["craft_place"]
        if e["action"] == "Place Table" and e["success"]
    ]
    crafts = [
        e for e in events["craft_place"]
        if e["kind"] == "craft" and e["station_nearby"]
    ]
    for craft in crafts:
        preceding_table = next(
            (pt for pt in place_tables if pt["step"] < craft["step"]),
            None,
        )
        if not preceding_table:
            continue
        materials = CRAFT_PREREQUISITES[craft["action"]]["materials"]
        # Find the latest preceding harvest of each required material.
        prereq_harvests = []
        for mat in materials:
            mat_target = mat if mat != "wood" else "tree"
            relevant = [
                h for h in events["successful_harvests"]
                if h["target"] == mat_target and h["step"] < craft["step"]
            ]
            if relevant:
                prereq_harvests.append((mat, relevant[-1]["step"]))
        if not prereq_harvests:
            continue
        cited_steps = sorted(
            {h[1] for h in prereq_harvests}
            | {preceding_table["step"], craft["step"]}
        )
        if len(cited_steps) < 4:
            continue
        prereq_desc = "; ".join(
            f"harvested {mat} at step {st}" for mat, st in prereq_harvests
        )
        all_steps_str = ", ".join(str(s) for s in cited_steps)
        question = _pick(craft["step"], [
            f"At step {craft['step']}, the agent successfully crafts a "
            f"{craft['product']}, depending on a chain of prerequisite "
            f"actions across steps {all_steps_str}: the agent "
            f"{prereq_desc}, and placed a crafting table at step "
            f"{preceding_table['step']}. Why was this multi-step chain a "
            f"critical dependency, and what would have failed if any "
            f"single one of steps {all_steps_str} had been omitted?",

            f"The agent crafts a {craft['product']} at step "
            f"{craft['step']}, the final action in a chain spanning steps "
            f"{all_steps_str}. Why was every step of this chain "
            f"({prereq_desc}, then 'Place Table' at step "
            f"{preceding_table['step']}) a strict precondition for the "
            f"craft at step {craft['step']}?",
        ])
        candidates.append(_mk(
            question,

            f"Each step in the chain at {all_steps_str} satisfied a "
            f"strict precondition for the '{craft['action']}' recipe. The "
            f"harvest(s) put the required materials "
            f"({', '.join(materials)}) into the agent's hidden inventory; "
            f"without these, the recipe's material check would have "
            f"failed at step {craft['step']}. The 'Place Table' at step "
            f"{preceding_table['step']} created the crafting station the "
            f"recipe requires; without it, no station would have been "
            f"adjacent at step {craft['step']} and the action would have "
            f"had no effect. Crafter enforces this chain strictly — the "
            f"{craft['product']} can only be obtained by completing every "
            f"step at {all_steps_str} in order.",
            "B",
        ))

    return candidates


# ---------------------------------------------------------------------------
# Type C — State Updating / hidden state inference.
#
# AMA-Bench C questions ask "what crucial unobserved change occurred" or "what
# hidden state can be inferred", typically about inventory, energy, thirst,
# HP, or coordinate state.  Distinctive phrasing: "what crucial, unobserved
# change occurred in the agent's internal state at this moment, and why is
# this change a fundamental prerequisite ..." (HF row 6 qa_pair[6]).
# ---------------------------------------------------------------------------

def generate_type_c(trajectory, events):
    """Type C — State Updating / hidden state inference.  Every template
    cites >=4 distinct trajectory steps in both question and answer."""
    candidates = []

    # C1: Inventory chain — harvest → place table → craft.  Cites all three
    # event steps + a fourth (post-craft observation OR a second harvest).
    # Mirrors HF row 6 qa_pair[6] but extended over a multi-step chain.
    place_tables = [
        e for e in events["craft_place"]
        if e["action"] == "Place Table" and e["success"]
    ]
    crafts = [
        e for e in events["craft_place"]
        if e["kind"] == "craft" and e["station_nearby"]
    ]
    wood_harvests = [
        h for h in events["successful_harvests"] if h["target"] == "tree"
    ]
    for craft in crafts:
        preceding_table = next(
            (pt for pt in place_tables if pt["step"] < craft["step"]),
            None,
        )
        if not preceding_table:
            continue
        wood_before_table = [h["step"] for h in wood_harvests if h["step"] < preceding_table["step"]]
        if len(wood_before_table) < 2:
            continue
        wood1, wood2 = wood_before_table[0], wood_before_table[1]
        table_step = preceding_table["step"]
        craft_step = craft["step"]
        product = craft["product"]
        question = _pick(craft_step, [
            f"Across steps {wood1}, {wood2}, {table_step}, and "
            f"{craft_step}, the agent's hidden inventory undergoes a chain "
            f"of state updates: it harvests wood at steps {wood1} and "
            f"{wood2}, places a crafting table at step {table_step}, and "
            f"crafts a {product} at step {craft_step}. What unobserved "
            f"inventory state can be inferred at each of these four "
            f"moments, and how does the chain explain the success of the "
            f"craft at step {craft_step}?",

            f"At step {craft_step}, the agent crafts a {product}. Looking "
            f"back at the chain spanning steps {wood1}, {wood2}, "
            f"{table_step}, and {craft_step}, what hidden inventory "
            f"updates occurred at each of these moments, and why was the "
            f"full chain necessary for the craft to succeed?",
        ])
        candidates.append(_mk(
            question,

            f"After step {wood1}, the agent's hidden inventory contained "
            f"1 wood (added by harvesting the tree). After step {wood2}, "
            f"it contained 2 wood. At step {table_step}, the 'Place Table' "
            f"consumed 1 wood to materialize a crafting table on the map "
            f"(inventory: 1 wood remaining). At step {craft_step}, the "
            f"'{craft['action']}' consumed the remaining wood and produced "
            f"a {product} stored back into inventory. The success at step "
            f"{craft_step} required two preconditions both established by "
            f"the prior chain: an adjacent crafting table (placed at step "
            f"{table_step}) and sufficient wood in inventory (collected "
            f"across steps {wood1} and {wood2}). Without any one of these "
            f"hidden state updates, the craft at step {craft_step} would "
            f"have failed.",
            "C",
        ))

    # C2: Multi-failure tool inference.  Cites every failure step + the
    # surrounding context.  Requires >=4 failures (so >=4 explicit step refs).
    # Mirrors HF row 6 qa_pair[2] / row 8.
    failed_groups = {}
    for h in events["failed_harvests"]:
        failed_groups.setdefault(h["target"], []).append(h)
    for target, group in failed_groups.items():
        required = TOOL_REQUIRED.get(target)
        if not required or len(group) < 4:
            continue
        fail_steps = [h["step"] for h in group[:6]]
        steps_str = ", ".join(str(s) for s in fail_steps)
        first = fail_steps[0]
        last = fail_steps[-1]
        question = _pick(first, [
            f"At steps {steps_str}, the agent repeatedly attempts the 'Do' "
            f"action on a {target} deposit but fails to collect any "
            f"resources across all {len(fail_steps)} attempts spanning "
            f"steps {first} through {last}. Based on Crafter's tool "
            f"progression rules, what can be inferred about the agent's "
            f"hidden inventory state that explains this persistent "
            f"failure?",

            f"In steps {first}-{last}, the agent's 'Do' action on a "
            f"{target} deposit has no effect at any of the "
            f"{len(fail_steps)} attempts ({steps_str}). What hidden "
            f"inventory state can be inferred, and why does that state "
            f"prevent any progress across this entire span?",
        ])
        candidates.append(_mk(
            question,

            f"It can be inferred that, throughout steps {first} to {last}, "
            f"the agent's hidden inventory does not contain a {required} "
            f"(or any higher-tier pickaxe). According to the game "
            f"mechanics, {target} cannot be mined with bare hands or "
            f"lower-tier tools. The fact that the 'Do' action had no "
            f"effect at every one of steps {steps_str} — even though the "
            f"agent was correctly positioned — is direct evidence that the "
            f"tool prerequisite was unmet at every one of those moments. "
            f"This unmet state persists across the entire span and "
            f"prevents progress on this resource branch of the tech tree.",
            "C",
        ))

    # C3: Hidden HP reduction from multi-hit combat.  Requires num_hits >=3
    # so each hit step + post-combat = >=4 step refs.
    # Mirrors HF row 8 cow-combat qa_pair[?].
    for c in events["combats"]:
        if c["num_hits"] < 3 or not c["enemy_defeated"]:
            continue
        enemy = c["enemy"]
        steps_str = ", ".join(str(s) for s in c["steps"])
        post_step = c["end_step"] + 1
        early_steps = ", ".join(str(s) for s in c["steps"][:-1])
        question = _pick(c["start_step"], [
            f"At steps {steps_str}, the agent uses the 'Do' action on a "
            f"{enemy}. The first {c['num_hits'] - 1} hits at steps "
            f"{early_steps} produce no reward and no visible change, but "
            f"the final hit at step {c['end_step']} yields a +1.0 reward "
            f"and the {enemy}'s disappearance by step {post_step}. What "
            f"unobserved state of the {enemy} must have been changing "
            f"across steps {steps_str}, and why did the cumulative change "
            f"finally trigger the reward at step {c['end_step']}?",

            f"The agent attacks a {enemy} {c['num_hits']} times at steps "
            f"{steps_str}, with no visible change at steps {early_steps} "
            f"but a +1.0 reward and disappearance at step {post_step}. "
            f"What hidden state of the {enemy} was being modified at each "
            f"of these steps?",
        ])
        candidates.append(_mk(
            question,

            f"The unobserved state being changed was the {enemy}'s hidden "
            f"health value. In Crafter, creatures have an internal HP that "
            f"is reduced by 'Do' (attack) actions but is never shown in "
            f"observations. The first {c['num_hits'] - 1} attacks at "
            f"steps {early_steps} each successfully reduced the {enemy}'s "
            f"HP — the lack of visible change at those steps does not "
            f"mean the hits failed, only that HP had not yet reached "
            f"zero. The final attack at step {c['end_step']} was the "
            f"killing blow that brought HP to 0; this triggered both the "
            f"{enemy}'s removal from the world (visible by step "
            f"{post_step}) and the "
            f"'{DEFEAT_ACHIEVEMENT.get(enemy, 'defeat')}' achievement "
            f"reward.",
            "C",
        ))

    # C4: Hidden energy restoration during sleep.  Cites: start, start+1
    # (interior Noop), end, wake.
    for s in events["sleep_sequences"]:
        if s["num_steps"] < 3 or s["wake_step"] is None:
            continue
        start = s["start_step"]
        end = s["end_step"]
        wake = s["wake_step"]
        question = _pick(start, [
            f"From step {start} through step {end}, the agent enters a "
            f"sleeping state and performs only 'Sleep'/'Noop' actions, "
            f"finally waking at step {wake}. What critical hidden resource "
            f"is being restored across the interior steps (e.g. step "
            f"{start + 1} through step {end}), and why is this "
            f"restoration strategically necessary for the agent to resume "
            f"productive actions at step {wake}?",

            f"Throughout steps {start} to {end}, the agent stays in a "
            f"sleeping state and finally wakes at step {wake}. What "
            f"hidden resource is being incrementally restored each turn "
            f"between step {start + 1} and step {end}, and why does the "
            f"agent only wake once this restoration is complete?",
        ])
        candidates.append(_mk(
            question,

            f"The hidden resource being restored is the agent's energy. "
            f"According to Crafter's mechanics, most actions consume "
            f"energy and once energy is depleted the agent's effectiveness "
            f"collapses. The sleep cycle that began at step {start} "
            f"continues incrementing energy each turn through step "
            f"{start + 1} and onward to step {end}; the seemingly idle "
            f"Noop actions are actually advancing the energy stat toward "
            f"full. The agent wakes at step {wake} only when energy is "
            f"fully restored. This is why the agent then becomes able to "
            f"take productive actions from step {wake} onward — the "
            f"hidden energy prerequisite for non-Sleep actions has been "
            f"satisfied.",
            "C",
        ))

    # C5: Multi-water hidden thirst.  Cites every water-drink step.
    # Requires >=4 water interactions.
    if len(events["water_interactions"]) >= 4:
        water_steps = [w["step"] for w in events["water_interactions"][:6]]
        steps_str = ", ".join(str(s) for s in water_steps)
        first = water_steps[0]
        last = water_steps[-1]
        question = _pick(first, [
            f"At steps {steps_str}, the agent performs 'Do' actions while "
            f"facing water tiles, and the water tiles do not disappear. "
            f"The agent returns to drink {len(water_steps)} times "
            f"spanning steps {first} through {last}. What unobserved "
            f"internal state of the agent was being depleted between "
            f"these drinks, and why does the repeated drinking pattern at "
            f"steps {steps_str} reveal a continuous survival mechanic?",

            f"The agent drinks from water at steps {steps_str}, "
            f"returning {len(water_steps)} times across steps {first}-"
            f"{last}. What hidden survival stat must have been depleting "
            f"between these visits to motivate each return?",
        ])
        candidates.append(_mk(
            question,

            f"The agent's hidden thirst (drink) meter was being depleted "
            f"between each of the drinks at steps {steps_str}. In "
            f"Crafter, thirst decreases over time regardless of action; "
            f"once it reaches zero the agent begins losing health. Each "
            f"'Do' on water tops the meter back up but does not consume "
            f"the water tile (water is permanent terrain). The need to "
            f"revisit and drink at all of steps {steps_str} is direct "
            f"evidence that thirst was non-trivially low each time the "
            f"agent returned, showing the depletion–replenishment cycle "
            f"in action across the span from step {first} to step "
            f"{last}.",
            "C",
        ))

    return candidates


# ---------------------------------------------------------------------------
# Type D — State Abstraction / multi-step strategic purpose.
#
# AMA-Bench D questions reason about sequences of actions: "What was the
# specific strategic purpose of these movements, and why were they all
# necessary ..." (HF row 6 qa_pair[?]); productive vs counter-productive
# breakdowns (HF row 7, row 8); oscillation analysis (HF row 9).
# ---------------------------------------------------------------------------

def generate_type_d(trajectory, events):
    """Type D — State Abstraction / multi-step strategic purpose.  Every
    template cites >=4 distinct trajectory steps in both question and
    answer."""
    candidates = []

    # D1: Approach maneuver — multi-move sequence ending in successful Do.
    # Cap at 8 moves (AMA-Bench D questions list 3–5).  Cites every move
    # step + do_step inline, so >=4 step refs when num_moves >= 3.
    for ap in events["approaches"]:
        if ap["num_moves"] < 3 or ap["num_moves"] > 8:
            continue
        do_step = ap["do_step"]
        if do_step >= len(trajectory):
            continue
        do_obs = parse_observation(trajectory[do_step]["observation"])
        face = do_obs["face"]
        if not face or (face in TERRAIN_TYPES and face != "water"):
            continue
        moves = ap["moves"]
        start = ap["start_step"]
        move_steps = ", ".join(str(start + i) for i in range(len(moves)))
        question = _pick(do_step, [
            f"In the sequence from step {start} to step {do_step}, the "
            f"agent performs {ap['num_moves']} movement actions before a "
            f"'Do' at step {do_step} that makes the {face} disappear. What "
            f"was the specific strategic purpose of these "
            f"{ap['num_moves']} movements, and why were they all necessary "
            f"to enable the outcome at step {do_step}?",

            f"Between steps {start} and {do_step}, the agent performs "
            f"{ap['num_moves']} movement actions and then a 'Do' at step "
            f"{do_step} that successfully removes a {face}. What strategic "
            f"purpose did these movements at steps {move_steps} serve, and "
            f"why was each one required for the success at step {do_step}?",

            f"The agent's actions across steps {start}-{do_step} consist "
            f"of {ap['num_moves']} movement actions followed by a 'Do' at "
            f"step {do_step}. Describe the strategic purpose of these "
            f"movements and explain why they were all necessary "
            f"prerequisites for the outcome at step {do_step}.",
        ])
        candidates.append(_mk(
            question,

            f"The strategic purpose of the movements at steps {move_steps} "
            f"was to navigate the agent from its starting position to a "
            f"tile directly adjacent to the target {face}. These movements "
            f"were not random exploration; they were a precise approach "
            f"maneuver. Each step closed a portion of the distance: the "
            f"early moves covered the bulk, and the final move at step "
            f"{do_step - 1} placed the agent in the harvesting tile. They "
            f"were all necessary because the 'Do' action in Crafter only "
            f"affects the tile directly in front of the agent. Without "
            f"this exact positioning at step {do_step - 1}, the 'Do' "
            f"action at step {do_step} would have targeted an empty tile "
            f"with no effect.",
            "D",
        ))

    # D2: Oscillation breakdown — productive vs counter-productive.  Cites
    # each move step inline.  Cap at 8 moves.
    for osc in events["oscillations"]:
        if osc["num_moves"] < 4 or osc["num_moves"] > 8:
            continue
        start = osc["start_step"]
        end = osc["end_step"]
        # Identify which steps are part of an opposing pair.
        wasted_steps = []
        moves = osc["moves"]
        for j in range(len(moves) - 1):
            for k in range(j + 1, len(moves)):
                if OPPOSITE_MOVE.get(moves[j]) == moves[k]:
                    wasted_steps.append(start + j)
                    wasted_steps.append(start + k)
        wasted_steps = sorted(set(wasted_steps))
        wasted_str = ", ".join(str(s) for s in wasted_steps[:4])
        all_step_str = ", ".join(str(start + i) for i in range(len(moves)))
        question = _pick(start, [
            f"Between steps {start} and {end}, the agent performs "
            f"{osc['num_moves']} movement actions but makes no observable "
            f"progress. Which of these actions across steps {start} "
            f"through {end} were part of a direct, forward-progress path, "
            f"and which were counter-productive or simply corrective "
            f"maneuvers?",

            f"The agent makes a series of {osc['num_moves']} moves "
            f"between steps {start} and {end} (across steps "
            f"{all_step_str}) without observable progress. Identify which "
            f"actions made forward progress and which were "
            f"counter-productive, and explain why this sequence is "
            f"considered a poor strategy.",
        ])
        candidates.append(_mk(
            question,

            f"None of the actions across steps {start} through {end} made "
            f"meaningful net forward progress. The sequence is a poor "
            f"strategy because the agent engages in 'oscillating' "
            f"movement: the moves at steps {wasted_str} include opposing "
            f"pairs (e.g., a Move North at one step is later cancelled by "
            f"a Move South). These opposing pairs return the agent to a "
            f"previously occupied tile without gaining new information or "
            f"approaching any resource. Efficient exploration requires "
            f"sustained directional movement; the agent's behavior across "
            f"steps {start}-{end} is the opposite of that, wasting turns "
            f"on backtracking.",
            "D",
        ))

    # D3: Single productive action in a 4-step window — extended to 4 steps.
    # Cites step-2, step-1, step, step+1 (4 distinct refs).
    for ch in events["observation_changes"]:
        if ch["kind"] != "empty_to_rich":
            continue
        step = ch["step"]
        if step < 3 or step + 1 >= len(trajectory):
            continue
        s_minus2 = step - 2
        s_minus1 = step - 1
        s_plus1 = step + 1
        prev2_act = trajectory[s_minus2]["action"]
        prev_act = trajectory[s_minus1]["action"]
        next_act = trajectory[s_plus1]["action"]
        if prev2_act not in MOVE_ACTIONS or prev_act not in MOVE_ACTIONS or next_act not in MOVE_ACTIONS:
            continue
        new_types = sorted(ch.get("new_types", set()))
        type_list = ", ".join(new_types) if new_types else "multiple objects"
        question = _pick(step, [
            f"In the sequence from step {s_minus2} to step {s_plus1}, "
            f"the agent performs four movement actions but only one "
            f"provides new environmental information. Identify this "
            f"single productive action across steps {s_minus2}-{s_plus1} "
            f"and explain why the surrounding movements at steps "
            f"{s_minus2}, {s_minus1}, and {s_plus1} were unproductive.",

            f"Across steps {s_minus2}, {s_minus1}, {step}, and "
            f"{s_plus1}, the agent makes four moves; only one reveals "
            f"new environmental information. Identify this productive "
            f"action and explain why the others were unproductive.",
        ])
        candidates.append(_mk(
            question,

            f"The only productive action was '{ch['action']}' at step "
            f"{step}: it shifted the agent to a coordinate whose 7-tile "
            f"scanning radius now overlaps the locations of {type_list}, "
            f"replacing the empty observation at step {s_minus1} with a "
            f"rich one at step {step}. The '{prev2_act}' at step "
            f"{s_minus2} and '{prev_act}' at step {s_minus1} both left "
            f"the agent in positions with empty observations — neither "
            f"revealed any objects. The '{next_act}' at step {s_plus1} "
            f"moved the agent off the informative tile, likely returning "
            f"to an empty observation. Across the four-step window only "
            f"the action at step {step} produced new map information, "
            f"because only its specific destination tile fell within "
            f"scanning range of {type_list}.",
            "D",
        ))

    # D4: Full crafting chain — wood harvest(s) + place table + craft.
    # Cites 4 widely-separated steps.
    place_events = [
        e for e in events["craft_place"]
        if e["action"] == "Place Table" and e["success"]
    ]
    craft_events = [
        e for e in events["craft_place"] if e["kind"] == "craft"
    ]
    wood_harvests = [
        h for h in events["successful_harvests"] if h["target"] == "tree"
    ]
    if len(wood_harvests) >= 2 and place_events and craft_events:
        first_wood = wood_harvests[0]["step"]
        second_wood = wood_harvests[1]["step"]
        first_table = place_events[0]["step"]
        first_craft = craft_events[0]["step"]
        if first_wood < second_wood < first_table < first_craft:
            product = craft_events[0]["product"]
            question = _pick(first_craft, [
                f"Across steps {first_wood}, {second_wood}, "
                f"{first_table}, and {first_craft}, the agent executes a "
                f"strategic chain: it harvests wood at steps {first_wood} "
                f"and {second_wood}, places a crafting table at step "
                f"{first_table}, and crafts a {product} at step "
                f"{first_craft}. Describe the strategic dependency chain "
                f"across these four steps and explain why each was a "
                f"strict prerequisite for the next.",

                f"At step {first_craft}, the agent crafts a {product} — "
                f"the final action in a strategic chain that began at "
                f"step {first_wood} (wood harvest), continued at step "
                f"{second_wood} (second wood harvest), and step "
                f"{first_table} (place table). Why is each link in this "
                f"chain a strict prerequisite for the next?",
            ])
            candidates.append(_mk(
                question,

                f"This four-step chain is the fundamental early-game "
                f"progression in Crafter, with each step strictly "
                f"dependent on the previous. Step {first_wood}: harvest "
                f"wood (the only resource collectible with bare hands). "
                f"Step {second_wood}: harvest a second wood — needed "
                f"because both placing a table and crafting consume wood. "
                f"Step {first_table}: spend wood to place a crafting "
                f"table, the required station for tool recipes. Step "
                f"{first_craft}: with a table adjacent and remaining wood "
                f"in inventory, craft a {product}. Without enough wood "
                f"(steps {first_wood} and {second_wood}) there is no "
                f"table, and without a table (step {first_table}) there "
                f"is no tool. The chain cannot be reordered or skipped.",
                "D",
            ))

    # D5: Reward-vs-no-reward decomposition over an approach to first-time
    # achievement.  Cites every approach step + do_step.
    first_time_steps = {ft["step"]: ft for ft in events["achievements"]["first_time"]}
    for ap in events["approaches"]:
        if ap["num_moves"] < 3 or ap["num_moves"] > 8:
            continue
        do_step = ap["do_step"]
        if do_step not in first_time_steps:
            continue
        ach = first_time_steps[do_step]["achievement"]
        moves = ap["moves"]
        start = ap["start_step"]
        last_move = moves[-1]
        approach_steps = ", ".join(str(s) for s in range(start, do_step))
        earlier_steps = approach_steps[:approach_steps.rfind(",")] or str(start)
        question = _pick(do_step, [
            f"The agent performs {ap['num_moves']} movement actions "
            f"across steps {approach_steps}, all of which yield 0.0 "
            f"reward. In contrast, the 'Do' action at step {do_step} "
            f"yields +1.0 (unlocking '{ach}'). Why was '{last_move}' at "
            f"step {do_step - 1} strategically critical, while the "
            f"earlier movements at steps {earlier_steps} were not "
            f"directly rewarded?",

            f"At step {do_step}, the agent earns a +1.0 reward "
            f"(unlocking '{ach}') after {ap['num_moves']} earlier moves "
            f"at steps {approach_steps}, all of which yielded 0.0. Why "
            f"was the move at step {do_step - 1} the strategically "
            f"critical one, and why didn't the earlier moves earn any "
            f"reward?",
        ])
        candidates.append(_mk(
            question,

            f"'{last_move}' at step {do_step - 1} was strategically "
            f"critical because it was the necessary precondition for the "
            f"reward-triggering 'Do' action — it placed the agent "
            f"directly adjacent to and facing the target. The +1.0 "
            f"reward at step {do_step} was for the '{ach}' achievement, "
            f"which requires the 'Do' action to succeed at this exact "
            f"position. Movement actions never directly produce rewards "
            f"in Crafter — they reposition the agent but do not trigger "
            f"achievements. The earlier movements at steps "
            f"{earlier_steps} were preparatory navigation; each closed "
            f"part of the distance to the target but none of them "
            f"produced the adjacency. Only the final move at step "
            f"{do_step - 1} created the precondition that 'Do' at step "
            f"{do_step} could exploit.",
            "D",
        ))

    return candidates


# ---------------------------------------------------------------------------
# Layer 4: Question selector
# ---------------------------------------------------------------------------

def _question_signature(q):
    """Extract a rough semantic signature to avoid near-duplicate questions."""
    text = q["question"].lower()
    for resource in ("tree", "wood", "stone", "coal", "iron", "diamond"):
        if resource in text:
            return (q["type"], resource)
    for enemy in ("zombie", "skeleton", "cow"):
        if enemy in text:
            return (q["type"], enemy)
    if "reward" in text:
        return (q["type"], "reward")
    if "sleep" in text or "energy" in text:
        return (q["type"], "sleep")
    if "water" in text or "drink" in text or "thirst" in text:
        return (q["type"], "water")
    if "observation" in text or "field of view" in text or "see nothing" in text or "empty" in text:
        return (q["type"], "fov")
    if "health" in text or "hp" in text or "hit" in text:
        return (q["type"], "hp")
    if "craft" in text or "table" in text:
        return (q["type"], "craft")
    if "exploration" in text or "first resource" in text:
        return (q["type"], "explore")
    return (q["type"], "other")


def _qa_step_refs(q):
    """All step numbers referenced anywhere in the question or answer."""
    return _step_refs(q["question"] + " " + q["answer"])


def _pick_from_pool(pool, quota, global_used_steps):
    used_sigs = set()
    chosen = []
    # Filter pool: every QA must reference >= MIN_STEP_REFS distinct step
    # numbers across question + answer (combined).  This is the safety net
    # for the per-template guarantees in Layer 3.
    eligible = [q for q in pool if len(_qa_step_refs(q)) >= MIN_STEP_REFS]
    for q in eligible:
        sig = _question_signature(q)
        step_refs = _qa_step_refs(q)
        if step_refs & global_used_steps:
            continue
        if sig in used_sigs:
            continue
        chosen.append(q)
        global_used_steps |= step_refs
        used_sigs.add(sig)
        if len(chosen) >= quota:
            break
    if len(chosen) < quota:
        chosen_uuids = {c["question_uuid"] for c in chosen}
        for q in eligible:
            if q["question_uuid"] in chosen_uuids:
                continue
            step_refs = _qa_step_refs(q)
            if step_refs & global_used_steps:
                continue
            chosen.append(q)
            global_used_steps |= step_refs
            if len(chosen) >= quota:
                break
    return chosen


def select_questions(candidates_by_type, target=12):
    ALL_TYPES = list(candidates_by_type.keys())
    available = [t for t in ALL_TYPES if candidates_by_type.get(t)]
    if not available:
        return []

    per_type = max(1, target // len(available))
    max_per_type = max(2, target // max(3, len(available) - 1))
    selected = []
    type_counts = {t: 0 for t in ALL_TYPES}
    global_used_steps = set()

    for qtype in available:
        pool = candidates_by_type[qtype]
        chosen = _pick_from_pool(pool, per_type, global_used_steps)
        selected.extend(chosen)
        type_counts[qtype] += len(chosen)

    if len(selected) < target:
        selected_uuids = {q["question_uuid"] for q in selected}
        remaining = []
        for qtype in available:
            for q in candidates_by_type[qtype]:
                if q["question_uuid"] not in selected_uuids and len(_qa_step_refs(q)) >= MIN_STEP_REFS:
                    remaining.append(q)
        for q in remaining:
            if len(selected) >= target:
                break
            if type_counts[q["type"]] >= max_per_type:
                continue
            step_refs = _qa_step_refs(q)
            if step_refs & global_used_steps:
                continue
            selected.append(q)
            global_used_steps |= step_refs
            type_counts[q["type"]] += 1

    # Last-resort backfill: drop the no-step-overlap requirement but still
    # enforce the >=MIN_STEP_REFS floor so we never emit a single-step QA.
    if len(selected) < target:
        selected_uuids = {q["question_uuid"] for q in selected}
        for qtype in available:
            for q in candidates_by_type[qtype]:
                if len(selected) >= target:
                    break
                if q["question_uuid"] in selected_uuids:
                    continue
                if len(_qa_step_refs(q)) < MIN_STEP_REFS:
                    continue
                selected.append(q)
                selected_uuids.add(q["question_uuid"])

    return selected[:target]


# ---------------------------------------------------------------------------
# Main entry point
# ---------------------------------------------------------------------------

DEFAULT_INPUT = "extracted_above_avg_trajectories.json"
DEFAULT_OUTPUT = "extracted_above_avg_trajectories_ama.json"

TASK_DESCRIPTION = (
    "Crafter-style survival/crafting environment: the agent gathers "
    "resources, crafts tools, and navigates a small world to achieve "
    "objectives under constraints."
)


def generate_qa_rich(trajectory, target=12):
    events = analyze_trajectory(trajectory)
    candidates_by_type = {
        "A": generate_type_a(trajectory, events),
        "B": generate_type_b(trajectory, events),
        "C": generate_type_c(trajectory, events),
        "D": generate_type_d(trajectory, events),
    }
    return select_questions(candidates_by_type, target=target)


def build_ama_episode(ep_idx, ep_in, target=12):
    trajectory = ep_in["trajectory"]
    qa_pairs = generate_qa_rich(trajectory, target=target)
    return {
        "episode_id": ep_idx,
        "task": TASK_DESCRIPTION,
        "task_type": "crafter",
        "domain": "Game",
        "success": False,
        "num_turns": ep_in.get("num_steps", len(trajectory)),
        "total_tokens": 0,
        "trajectory": trajectory,
        "qa_pairs": qa_pairs,
    }


def main():
    parser = argparse.ArgumentParser(
        description="Generate AMA-Bench-style Q&A pairs from curated Crafter "
                    "trajectories and emit AMA-Bench-shaped JSON.",
        epilog=f"""\
default behavior:
  Reads {DEFAULT_INPUT} (27 curated above-average heuristic-agent
  episodes) and writes {DEFAULT_OUTPUT} in AMA-Bench schema.

input format (per episode):
  episode      int  — original episode index (preserved internally only)
  num_steps    int  — number of trajectory turns
  trajectory   list — list of turns with keys: turn_idx, action, observation

output schema (AMA-Bench compatible — see
https://huggingface.co/datasets/AMA-bench/AMA-bench/test/open_end_qa_set.jsonl):
  episode_id    int   — sequential 0..N-1 across the file
  task          str   — Crafter task description
  task_type     str   — "crafter"
  domain        str   — "Game"
  success       bool  — false (heuristic-agent episodes have no reward signal)
  num_turns     int   — copied from input num_steps
  total_tokens  int   — 0 (heuristic agents do not consume LLM tokens)
  trajectory    list  — passed through unchanged
  qa_pairs      list  — list of {{question, answer, type, question_uuid}}
                         where type ∈ {{A, B, C, D}} matching AMA-Bench's
                         taxonomy: A=Recall (counterfactual in practice),
                         B=Causal Inference, C=State Updating,
                         D=State Abstraction.

examples:
  # Default: read {DEFAULT_INPUT} -> {DEFAULT_OUTPUT}
  python generate_qa.py

  # Emit JSONL (one episode per line, true AMA-Bench format)
  python generate_qa.py --jsonl

  # Custom input/output
  python generate_qa.py --input my_episodes.json --output my_ama.json

  # Different number of QA pairs per episode (default 12, matching AMA-Bench)
  python generate_qa.py --target 10
""",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument(
        "--input", "-i", default=DEFAULT_INPUT,
        help=f"Input trajectory JSON file (default: {DEFAULT_INPUT})",
    )
    parser.add_argument(
        "--output", "-o", default=None,
        help="Output filename (default: <input_stem>_ama.json, "
             f"i.e. {DEFAULT_OUTPUT} for the default input). "
             "When --jsonl is set, the .json suffix is replaced with .jsonl.",
    )
    parser.add_argument(
        "--jsonl", action="store_true",
        help="Emit one episode per line (AMA-Bench native format) instead of "
             "a JSON array.",
    )
    parser.add_argument(
        "--target", type=int, default=12,
        help="Target number of Q&A pairs per episode (default: 12, matching "
             "AMA-Bench)",
    )
    args = parser.parse_args()

    with open(args.input) as f:
        episodes = json.load(f)

    out_path = args.output
    if out_path is None:
        stem = args.input[:-5] if args.input.endswith(".json") else args.input
        out_path = f"{stem}_ama.{'jsonl' if args.jsonl else 'json'}"
    elif args.jsonl and out_path.endswith(".json"):
        out_path = out_path[:-5] + ".jsonl"

    out_episodes = [
        build_ama_episode(idx, ep, target=args.target)
        for idx, ep in enumerate(episodes)
    ]

    with open(out_path, "w") as f:
        if args.jsonl:
            for ep in out_episodes:
                f.write(json.dumps(ep))
                f.write("\n")
        else:
            json.dump(out_episodes, f, indent=2)

    total_qa = sum(len(ep["qa_pairs"]) for ep in out_episodes)
    type_counts = {}
    for ep in out_episodes:
        for q in ep["qa_pairs"]:
            type_counts[q["type"]] = type_counts.get(q["type"], 0) + 1
    dist = ", ".join(f"{k}={v}" for k, v in sorted(type_counts.items()))
    print(
        f"{args.input} -> {out_path}: "
        f"{len(out_episodes)} episodes, {total_qa} qa_pairs ({dist})"
    )


if __name__ == "__main__":
    main()
