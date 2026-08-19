import json
import os
import warnings

warnings.filterwarnings("ignore", category=DeprecationWarning)
import crafter

from Heuristic_Agent import play_episode_smart

NUM_TRIALS = 10
NUM_EPISODES = 10
OUTPUT_DIR = "trial_trajectories"


if __name__ == "__main__":
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    for trial in range(NUM_TRIALS):
        print(f"\n{'='*40}")
        print(f"Trial {trial + 1}/{NUM_TRIALS}")
        print(f"{'='*40}")

        rec_dir = f"{OUTPUT_DIR}/heuristic_trial_{trial + 1}_recordings"
        raw_env = crafter.Env()
        raw_env = crafter.Recorder(
            raw_env,
            directory=rec_dir,
            save_stats=True,
            save_video=False,
            save_episode=True,
        )

        all_data = []

        for ep in range(NUM_EPISODES):
            print(f"  Episode {ep + 1}/{NUM_EPISODES}...", end=" ")
            trajectory = play_episode_smart(raw_env, max_steps=1000)

            episode_data = {
                "episode": ep,
                "num_steps": len(trajectory),
                "trajectory": trajectory,
            }
            all_data.append(episode_data)

            print(f"Steps: {len(trajectory)}")

        out_file = f"{OUTPUT_DIR}/heuristic_trial_{trial + 1}_trajectories.json"
        with open(out_file, "w") as f:
            json.dump(all_data, f, indent=2)

        steps = [ep["num_steps"] for ep in all_data]
        print(f"  Avg: {sum(steps)/len(steps):.1f}, Best: {max(steps)}")
        print(f"  Saved to {out_file}")
