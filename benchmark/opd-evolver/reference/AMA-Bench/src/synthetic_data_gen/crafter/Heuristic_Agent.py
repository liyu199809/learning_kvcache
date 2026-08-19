import json
import warnings
import numpy as np

warnings.filterwarnings("ignore", category=DeprecationWarning)
import crafter

ACTION_NAMES = [
    "Noop", "Move West", "Move East", "Move North", "Move South",
    "Do", "Sleep", "Place Stone", "Place Table", "Place Furnace",
    "Place Plant", "Make Wood Pickaxe", "Make Stone Pickaxe",
    "Make Iron Pickaxe", "Make Wood Sword", "Make Stone Sword",
    "Make Iron Sword",
]

VITAL_KEYS = ["health", "food", "drink", "energy"]
RESOURCE_KEYS = ["sapling", "wood", "stone", "coal", "iron", "diamond"]
TOOL_KEYS = [
    "wood_pickaxe", "stone_pickaxe", "iron_pickaxe",
    "wood_sword", "stone_sword", "iron_sword",
]


def direction_label(dx, dy):
    """Return a compass label for a relative offset."""
    if dx == 0 and dy == 0:
        return "here"
    parts = []
    if dy < 0:
        parts.append("north")
    elif dy > 0:
        parts.append("south")
    if dx < 0:
        parts.append("west")
    elif dx > 0:
        parts.append("east")
    return "-".join(parts)


def build_observation(env):
    """Build a text observation from the current game state."""
    player = env._player
    pos = player.pos
    facing = player.facing

    scan_range = 7
    items = []
    for dy in range(-scan_range, scan_range + 1):
        for dx in range(-scan_range, scan_range + 1):
            if dx == 0 and dy == 0:
                continue
            check = pos + np.array([dx, dy])
            mat, obj = env._world[check]
            dist = abs(dx) + abs(dy)
            d_label = direction_label(dx, dy)

            if obj is not None and type(obj).__name__ != "Player":
                obj_name = type(obj).__name__.lower()
                items.append((dist, obj_name, d_label))
            if mat is not None and mat not in ("grass", "path"):
                items.append((dist, mat, d_label))

    items.sort(key=lambda x: x[0])

    seen = set()
    unique_items = []
    for dist, name, d_label in items:
        key = (name, d_label)
        if key not in seen:
            seen.add(key)
            unique_items.append((dist, name, d_label))
        if len(unique_items) >= 12:
            break

    lines = []
    if unique_items:
        lines.append("You see:")
        for dist, name, d_label in unique_items:
            step_word = "step" if dist == 1 else "steps"
            lines.append(f"- {name} {dist} {step_word} to your {d_label}")
    else:
        lines.append("You see nothing nearby.")

    front_pos = pos + np.array(facing)
    front_mat, front_obj = env._world[front_pos]
    if front_obj is not None and type(front_obj).__name__ != "Player":
        front_desc = type(front_obj).__name__.lower()
    elif front_mat and front_mat != "grass":
        front_desc = front_mat
    else:
        front_desc = "grass"
    lines.append(f"\nYou face {front_desc} at your front.")

    obs = "\n".join(lines)
    if player.sleeping:
        obs = "You are sleeping, and will not be able take actions until energy is full.\n\n" + obs
    return obs


# ---- Smart Agent Helpers ----

def _find_nearest(env, materials=None, obj_types=None, scan_range=20):
    """Find nearest tile with matching material or object type."""
    player = env._player
    pos = player.pos
    best = None
    best_dist = float('inf')
    for dy in range(-scan_range, scan_range + 1):
        for dx in range(-scan_range, scan_range + 1):
            if dx == 0 and dy == 0:
                continue
            dist = abs(dx) + abs(dy)
            if dist >= best_dist:
                continue
            check = pos + np.array([dx, dy])
            mat, obj = env._world[check]
            if materials and mat in materials:
                best = (dx, dy)
                best_dist = dist
            if obj_types and obj is not None and type(obj).__name__ in obj_types:
                best = (dx, dy)
                best_dist = dist
    return best


def _act_interact(dx, dy, facing):
    """Action to approach and interact with tile at relative (dx, dy)."""
    dist = abs(dx) + abs(dy)
    if dist == 1:
        if tuple(facing) == (dx, dy):
            return 5  # Do
        return {(1, 0): 2, (-1, 0): 1, (0, 1): 4, (0, -1): 3}[(dx, dy)]
    if abs(dx) > abs(dy):
        return 2 if dx > 0 else 1
    if abs(dy) > abs(dx):
        return 4 if dy > 0 else 3
    return np.random.choice([2 if dx > 0 else 1, 4 if dy > 0 else 3])


def _act_navigate(dx, dy):
    """Move toward (dx, dy) without interacting."""
    if dx == 0 and dy == 0:
        return 0
    if abs(dx) >= abs(dy):
        return 2 if dx > 0 else 1
    return 4 if dy > 0 else 3


def _near_mat(env, pos, mat_name):
    """Check if material exists within 3x3 area around pos."""
    for dy in [-1, 0, 1]:
        for dx in [-1, 0, 1]:
            mat, _ = env._world[pos + np.array([dx, dy])]
            if mat == mat_name:
                return True
    return False


def _avoid_lava(env, pos, action):
    """Replace movement into lava with a safe alternative."""
    deltas = {1: (-1, 0), 2: (1, 0), 3: (0, -1), 4: (0, 1)}
    if action not in deltas:
        return action
    dx, dy = deltas[action]
    mat, _ = env._world[pos + np.array([dx, dy])]
    if mat != "lava":
        return action
    for alt in [1, 2, 3, 4]:
        if alt == action:
            continue
        adx, ady = deltas[alt]
        amat, _ = env._world[pos + np.array([adx, ady])]
        if amat != "lava":
            return alt
    return 0


def _get_goal(inv, ach, table_pos, furnace_pos, sapling_tries):
    """Determine current tech tree goal."""
    has_sword = any(inv[s] > 0 for s in ["wood_sword", "stone_sword", "iron_sword"])

    if inv["wood"] < 5 and table_pos is None:
        return "collect_wood"
    if table_pos is None and inv["wood"] >= 2:
        return "place_table"
    if inv["wood_pickaxe"] == 0:
        return "make_wood_pickaxe" if inv["wood"] >= 1 else "collect_wood"
    if inv["wood_sword"] == 0:
        return "make_wood_sword" if inv["wood"] >= 1 else "collect_wood"

    if not ach.get("collect_drink", 0):
        return "collect_drink"

    if inv["drink"] <= 7:
        return "collect_drink"
    if inv["food"] <= 6:
        return "eat_cow"

    if inv["sapling"] == 0 and not ach.get("place_plant", 0) and sapling_tries < 30:
        return "collect_sapling"
    if inv["sapling"] > 0 and not ach.get("place_plant", 0):
        return "place_plant"

    if inv["stone"] < 5 and inv["wood_pickaxe"] > 0:
        return "collect_stone"
    if inv["coal"] < 2 and inv["wood_pickaxe"] > 0:
        return "collect_coal"
    if inv["stone_pickaxe"] == 0 and inv["stone"] >= 1 and inv["wood"] >= 1:
        return "make_stone_pickaxe"
    if inv["stone_sword"] == 0 and inv["stone"] >= 1 and inv["wood"] >= 1:
        return "make_stone_sword"
    if furnace_pos is None and inv["stone"] >= 4:
        return "place_furnace"

    if inv["iron"] < 1 and inv["stone_pickaxe"] > 0:
        return "collect_iron"
    if inv["iron_pickaxe"] == 0 and inv["iron"] >= 1 and inv["coal"] >= 1 and inv["wood"] >= 1:
        return "make_iron_pickaxe"
    if inv["iron_sword"] == 0 and inv["iron"] >= 1 and inv["coal"] >= 1 and inv["wood"] >= 1:
        return "make_iron_sword"

    if inv["diamond"] == 0 and inv["iron_pickaxe"] > 0:
        return "collect_diamond"

    if not ach.get("eat_cow", 0):
        return "eat_cow"
    if not ach.get("defeat_zombie", 0) and has_sword:
        return "defeat_zombie"
    if not ach.get("defeat_skeleton", 0) and has_sword:
        return "defeat_skeleton"
    if not ach.get("wake_up", 0) and inv["energy"] < 9:
        return "sleep_goal"

    return "explore"


def _execute_goal(goal, env, pos, facing, inv, table_pos, furnace_pos, enemy):
    """Execute current goal. Returns action index."""

    if goal == "collect_wood":
        t = _find_nearest(env, materials={"tree"})
        return _act_interact(t[0], t[1], facing) if t else np.random.choice([1, 2, 3, 4])

    if goal == "place_table":
        front = pos + np.array(facing)
        mat, obj = env._world[front]
        if mat in ("grass", "sand", "path") and obj is None:
            return 8  # Place Table
        t = _find_nearest(env, materials={"tree"})
        return _act_interact(t[0], t[1], facing) if t else np.random.choice([1, 2, 3, 4])

    if goal in ("make_wood_pickaxe", "make_wood_sword", "make_stone_pickaxe", "make_stone_sword"):
        action_map = {
            "make_wood_pickaxe": 11, "make_wood_sword": 14,
            "make_stone_pickaxe": 12, "make_stone_sword": 15,
        }
        if _near_mat(env, pos, "table"):
            return action_map[goal]
        if table_pos is not None:
            return _act_navigate(table_pos[0] - pos[0], table_pos[1] - pos[1])
        return np.random.choice([1, 2, 3, 4])

    if goal in ("make_iron_pickaxe", "make_iron_sword"):
        action_idx = 13 if goal == "make_iron_pickaxe" else 16
        if _near_mat(env, pos, "table") and _near_mat(env, pos, "furnace"):
            return action_idx
        target = table_pos if table_pos is not None else furnace_pos
        if target is not None:
            return _act_navigate(target[0] - pos[0], target[1] - pos[1])
        return np.random.choice([1, 2, 3, 4])

    if goal == "collect_drink":
        w = _find_nearest(env, materials={"water"})
        return _act_interact(w[0], w[1], facing) if w else np.random.choice([1, 2, 3, 4])

    if goal == "collect_sapling":
        front = pos + np.array(facing)
        mat, obj = env._world[front]
        if mat == "grass" and obj is None:
            return 5  # Do on grass for 10% sapling chance
        g = _find_nearest(env, materials={"grass"}, scan_range=5)
        return _act_interact(g[0], g[1], facing) if g else np.random.choice([1, 2, 3, 4])

    if goal == "place_plant":
        front = pos + np.array(facing)
        mat, obj = env._world[front]
        if mat == "grass" and obj is None:
            return 10  # Place Plant
        g = _find_nearest(env, materials={"grass"}, scan_range=5)
        return _act_navigate(g[0], g[1]) if g else np.random.choice([1, 2, 3, 4])

    if goal == "collect_stone":
        s = _find_nearest(env, materials={"stone"})
        return _act_interact(s[0], s[1], facing) if s else np.random.choice([1, 2, 3, 4])

    if goal == "collect_coal":
        c = _find_nearest(env, materials={"coal"})
        if c:
            return _act_interact(c[0], c[1], facing)
        s = _find_nearest(env, materials={"stone"})
        return _act_navigate(s[0], s[1]) if s else np.random.choice([1, 2, 3, 4])

    if goal == "place_furnace":
        if table_pos is not None:
            tdist = abs(pos[0] - table_pos[0]) + abs(pos[1] - table_pos[1])
            if tdist > 3:
                return _act_navigate(table_pos[0] - pos[0], table_pos[1] - pos[1])
        front = pos + np.array(facing)
        mat, obj = env._world[front]
        if mat in ("grass", "sand", "path") and obj is None:
            return 9  # Place Furnace
        t = _find_nearest(env, materials={"tree"}, scan_range=5)
        return _act_interact(t[0], t[1], facing) if t else np.random.choice([1, 2, 3, 4])

    if goal == "collect_iron":
        i = _find_nearest(env, materials={"iron"}, scan_range=30)
        if i:
            return _act_interact(i[0], i[1], facing)
        s = _find_nearest(env, materials={"stone"}, scan_range=30)
        return _act_navigate(s[0], s[1]) if s else np.random.choice([1, 2, 3, 4])

    if goal == "collect_diamond":
        d = _find_nearest(env, materials={"diamond"}, scan_range=30)
        if d:
            return _act_interact(d[0], d[1], facing)
        s = _find_nearest(env, materials={"stone"}, scan_range=30)
        return _act_navigate(s[0], s[1]) if s else np.random.choice([1, 2, 3, 4])

    if goal == "eat_cow":
        c = _find_nearest(env, obj_types={"Cow"})
        return _act_interact(c[0], c[1], facing) if c else np.random.choice([1, 2, 3, 4])

    if goal in ("defeat_zombie", "defeat_skeleton"):
        target_type = "Zombie" if goal == "defeat_zombie" else "Skeleton"
        e = _find_nearest(env, obj_types={target_type})
        return _act_interact(e[0], e[1], facing) if e else np.random.choice([1, 2, 3, 4])

    if goal == "sleep_goal":
        if not enemy:
            return 6  # Sleep
        return np.random.choice([1, 2, 3, 4])

    # explore
    if inv["drink"] <= 6:
        w = _find_nearest(env, materials={"water"})
        if w:
            return _act_interact(w[0], w[1], facing)
    if inv["food"] <= 6:
        c = _find_nearest(env, obj_types={"Cow"})
        if c:
            return _act_interact(c[0], c[1], facing)
    if inv["wood"] < 3:
        t = _find_nearest(env, materials={"tree"})
        if t:
            return _act_interact(t[0], t[1], facing)
    return np.random.choice([1, 2, 3, 4])


def play_episode_smart(env, max_steps=1000):
    """Play one episode with a smart goal-directed agent."""
    obs = env.reset()
    trajectory = []

    prev_info = None
    table_pos = None
    furnace_pos = None
    plant_step = None
    stuck_count = 0
    last_pos = None
    sapling_tries = 0

    for step in range(max_steps):
        observation = build_observation(env)
        player = env._player
        pos = player.pos.copy()
        facing = player.facing

        if prev_info:
            inv = prev_info["inventory"]
            ach = prev_info["achievements"]
        else:
            inv = dict.fromkeys(VITAL_KEYS, 9)
            inv.update(dict.fromkeys(RESOURCE_KEYS + TOOL_KEYS, 0))
            ach = {}

        if table_pos is not None:
            mat, _ = env._world[table_pos]
            if mat != "table":
                table_pos = None
        if furnace_pos is not None:
            mat, _ = env._world[furnace_pos]
            if mat != "furnace":
                furnace_pos = None

        cp = tuple(pos)
        stuck_count = stuck_count + 1 if cp == last_pos else 0
        last_pos = cp

        enemy = _find_nearest(env, obj_types={"Zombie", "Skeleton"}, scan_range=4)
        has_sword = any(inv[s] > 0 for s in ["wood_sword", "stone_sword", "iron_sword"])
        action = None

        # P0: Unstuck
        if stuck_count > 5:
            action = np.random.choice([1, 2, 3, 4])
            stuck_count = 0

        # P1: Vital management
        if action is None and inv["drink"] <= 3:
            w = _find_nearest(env, materials={"water"})
            if w:
                action = _act_interact(w[0], w[1], facing)

        if action is None and inv["food"] <= 3:
            c = _find_nearest(env, obj_types={"Cow"})
            if c:
                action = _act_interact(c[0], c[1], facing)

        if action is None and inv["energy"] <= 3 and not enemy:
            action = 6  # Sleep

        # P2: Combat
        if action is None and enemy:
            edx, edy = enemy
            edist = abs(edx) + abs(edy)
            if edist <= 2:
                if has_sword:
                    action = _act_interact(edx, edy, facing)
                elif inv["health"] <= 4:
                    if abs(edx) >= abs(edy):
                        action = 1 if edx > 0 else 2
                    else:
                        action = 3 if edy > 0 else 4
                else:
                    action = _act_interact(edx, edy, facing)

        # P3: Tech tree goal
        if action is None:
            goal = _get_goal(inv, ach, table_pos, furnace_pos, sapling_tries)
            if goal == "collect_sapling":
                sapling_tries += 1
            action = _execute_goal(goal, env, pos, facing, inv, table_pos, furnace_pos, enemy)

        if action is None:
            action = np.random.choice([1, 2, 3, 4])

        action = _avoid_lava(env, pos, action)
        action_name = ACTION_NAMES[action]
        obs, reward, done, info = env.step(action)

        # Track placements
        if action == 8:  # Place Table
            check = pos + np.array(facing)
            mat, _ = env._world[check]
            if mat == "table":
                table_pos = check.copy()
        elif action == 9:  # Place Furnace
            check = pos + np.array(facing)
            mat, _ = env._world[check]
            if mat == "furnace":
                furnace_pos = check.copy()
        elif action == 10 and plant_step is None:  # Place Plant
            check = pos + np.array(facing)
            _, obj = env._world[check]
            if obj is not None and type(obj).__name__ == "Plant":
                plant_step = step

        curr_info = {
            "inventory": dict(info["inventory"]),
            "achievements": dict(info["achievements"]),
        }

        entry = {
            "turn_idx": step,
            "action": action_name,
            "observation": observation,
        }
        trajectory.append(entry)
        prev_info = curr_info

        if done:
            break

    return trajectory


if __name__ == "__main__":
    num_episodes = 10

    raw_env = crafter.Env()
    raw_env = crafter.Recorder(
        raw_env,
        directory="./heuristic_play_recordings",
        save_stats=True,
        save_video=True,
        save_episode=True,
    )

    all_data = []

    for ep in range(num_episodes):
        print(f"Playing episode {ep + 1}/{num_episodes}...")
        trajectory = play_episode_smart(raw_env, max_steps=1000)

        episode_data = {
            "episode": ep,
            "num_steps": len(trajectory),
            "trajectory": trajectory,
        }
        all_data.append(episode_data)

        print(f"  Steps: {len(trajectory)}")

    raw_env.close()

    with open("heuristic_crafter_trajectories.json", "w") as f:
        json.dump(all_data, f, indent=2)

    print(f"\nSaved {len(all_data)} episodes to heuristic_crafter_trajectories.json")
    print(f"Recordings saved to ./heuristic_play_recordings/")
