"""
Matched predictive-sensitivity experiment, release (paper Section "Causal
Sensitivity Around Contact Transitions", Table 1: Release / Exploratory and
Release / Confirmation). Reuses the identical onset pipeline
(src/matched_sensitivity.py: match_controls / run_event / all_control_candidates)
unchanged. The only change from the onset experiment is which contact
transitions are extracted: release (True->False) instead of onset
(False->True). No new episodes are generated -- this reads the already-saved
pilot_episodes.pkl (seeds 0-9) and pilot_episodes_confirmation.pkl
(seeds 100-109).

Usage: python run_matched_sensitivity_release.py [exploratory|confirmation]
"""
import json
import os
import pickle
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src import episode_generation as ep
from src import simulator_interface as lib
from src.matched_sensitivity import all_control_candidates, match_controls, run_event
from src.model_loading import load_models, pick_device

OUT_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "outputs")


def run_release_experiment(episodes_pkl, output_name, label):
    device = pick_device()
    print(f"Using device: {device}")
    tokenizer, model = load_models(device)

    with open(os.path.join(OUT_DIR, episodes_pkl), "rb") as f:
        episodes = pickle.load(f)

    release_events = []
    for seed, log in episodes.items():
        _, releases = ep.find_contact_events(log["contact_per_object"], obj_index=0)
        for t in releases:
            if t - 5 >= 0 and t + 6 <= len(log["action"]) - 1:
                release_events.append((seed, t))

    control_pool = all_control_candidates(episodes)
    print(f"\n[{label}] valid release events: {len(release_events)}, control pool: {len(control_pool)}")

    env = lib.build_env(seed=0)
    matches = match_controls(release_events, control_pool, episodes, env)

    results = []
    for i in range(len(release_events)):
        seed, t = release_events[i]
        (cseed, ct), dist, ocov, ccov = matches[i]
        print(f"\n--- [{label}] pair {i+1}/{len(release_events)}: release seed={seed} t={t}  <->  "
              f"control seed={cseed} t={ct}  (match dist={dist:.3f}) ---")
        r_release = run_event(env, tokenizer, model, device, episodes[seed], t, "release", seed)
        r_control = run_event(env, tokenizer, model, device, episodes[cseed], ct, "control", cseed)
        r_release["pair_id"] = i
        r_control["pair_id"] = i
        r_release["match_distance"] = float(dist)
        r_control["match_distance"] = float(dist)
        results.append(r_release)
        results.append(r_control)
        print(f"  release: eps={r_release['epsilon_mm']}mm hamming={r_release['injection_hamming']}/16 "
              f"downstream={r_release['downstream_divergence']}")
        print(f"  control: eps={r_control['epsilon_mm']}mm hamming={r_control['injection_hamming']}/16 "
              f"downstream={r_control['downstream_divergence']}")

    with open(os.path.join(OUT_DIR, output_name), "w") as f:
        json.dump(results, f, indent=2)
    print(f"\n[{label}] Saved {len(results)} event records ({len(release_events)} pairs) to outputs/{output_name}")
    return results


if __name__ == "__main__":
    which = sys.argv[1] if len(sys.argv) > 1 else "exploratory"
    if which == "exploratory":
        run_release_experiment("pilot_episodes.pkl", "rq2_release_exploratory_full.json",
                                 "EXPLORATORY seeds 0-9")
    elif which == "confirmation":
        run_release_experiment("pilot_episodes_confirmation.pkl", "rq2_release_confirmation_full.json",
                                 "CONFIRMATION seeds 100-109")
    else:
        raise ValueError(which)
