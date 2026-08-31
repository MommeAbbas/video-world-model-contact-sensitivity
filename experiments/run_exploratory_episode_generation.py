"""
Generates the exploratory episode batch (seeds 0-9) via
src/episode_generation.py::run_episode and saves it to
outputs/pilot_episodes.pkl, the path every downstream script expects.

Output schema: dict keyed by integer seed, each value the per-timestep log
dict returned by run_episode, consumed downstream via `episodes[seed]`.
"""
import os
import pickle
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from src.episode_generation import run_episode, find_contact_events

OUT_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "outputs")


def main():
    os.makedirs(OUT_DIR, exist_ok=True)
    all_logs = {}
    total_onsets, total_releases = 0, 0
    for seed in range(10):
        log = run_episode(seed=seed, n_cycles=3)
        onsets, releases = find_contact_events(log["contact_per_object"], obj_index=0)
        valid_onsets = [t for t in onsets if t - 5 >= 0 and t + 6 <= len(log["action"]) - 1]
        valid_releases = [t for t in releases if t - 5 >= 0 and t + 6 <= len(log["action"]) - 1]
        print(f"seed={seed} len={len(log['action'])} onsets={onsets} releases={releases} "
              f"valid_onsets={valid_onsets} valid_releases={valid_releases}")
        total_onsets += len(valid_onsets)
        total_releases += len(valid_releases)
        all_logs[seed] = log
    print("total valid onsets:", total_onsets, "valid releases:", total_releases)

    out_path = os.path.join(OUT_DIR, "pilot_episodes.pkl")
    with open(out_path, "wb") as f:
        pickle.dump(all_logs, f)
    print(f"saved exploratory episodes to {out_path}")


if __name__ == "__main__":
    main()
