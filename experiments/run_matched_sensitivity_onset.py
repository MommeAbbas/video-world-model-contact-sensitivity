"""
Matched predictive-sensitivity experiment, onset, exploratory batch (paper
Table 1: Onset / Exploratory). For every onset event, runs the calibrated
adaptive-epsilon intervention (src/matched_sensitivity.py) and its
nearest-neighbor-matched free-motion control, recording the full
identity/injection/covariate/outcome fields for both.
"""
import json
import os
import pickle
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src import simulator_interface as lib
from src.matched_sensitivity import all_control_candidates, match_controls, run_event
from src.model_loading import load_models, pick_device

OUT_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "outputs")


def main():
    device = pick_device()
    print(f"Using device: {device}")
    tokenizer, model = load_models(device)

    with open(os.path.join(OUT_DIR, "pilot_episodes.pkl"), "rb") as f:
        episodes = pickle.load(f)
    with open(os.path.join(OUT_DIR, "rq1_pilot_events.json")) as f:
        events = json.load(f)

    onset_events = [(s, t) for s, t in events["onset_events"]]
    control_pool = all_control_candidates(episodes)
    print(f"onset events: {len(onset_events)}, control candidate pool: {len(control_pool)}")

    env = lib.build_env(seed=0)
    matches = match_controls(onset_events, control_pool, episodes, env)

    results = []
    for i in range(len(onset_events)):
        seed, t = onset_events[i]
        (cseed, ct), dist, ocov, ccov = matches[i]
        print(f"\n--- pair {i+1}/{len(onset_events)}: onset seed={seed} t={t}  <->  control seed={cseed} t={ct}  "
              f"(match dist={dist:.3f}) ---")
        r_onset = run_event(env, tokenizer, model, device, episodes[seed], t, "onset", seed)
        r_control = run_event(env, tokenizer, model, device, episodes[cseed], ct, "control", cseed)
        r_onset["pair_id"] = i
        r_control["pair_id"] = i
        r_onset["match_distance"] = float(dist)
        r_control["match_distance"] = float(dist)
        results.append(r_onset)
        results.append(r_control)
        print(f"  onset:   eps={r_onset['epsilon_mm']}mm hamming={r_onset['injection_hamming']}/16 "
              f"downstream={r_onset['downstream_divergence']}")
        print(f"  control: eps={r_control['epsilon_mm']}mm hamming={r_control['injection_hamming']}/16 "
              f"downstream={r_control['downstream_divergence']}")

        if (i + 1) == 20:
            print(f"\ninterim checkpoint at {i+1} pairs")
            _save(results, "rq2_scaled_interim20.json")

    _save(results, "rq2_scaled_full.json")
    print(f"\nSaved {len(results)} event records ({len(onset_events)} pairs) to outputs/rq2_scaled_full.json")


def _save(results, name):
    with open(os.path.join(OUT_DIR, name), "w") as f:
        json.dump(results, f, indent=2)
    import csv
    csv_path = os.path.join(OUT_DIR, name.replace(".json", ".csv"))
    keys = [k for k in results[0].keys() if k not in ("changed_token_indices", "epsilon_search_trace", "downstream_divergence")]
    keys += ["downstream_div_1", "downstream_div_2", "downstream_div_3"]
    with open(csv_path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=keys)
        w.writeheader()
        for r in results:
            row = {k: r.get(k) for k in keys if k in r}
            for j in range(3):
                row[f"downstream_div_{j+1}"] = r["downstream_divergence"][j]
            w.writerow(row)


if __name__ == "__main__":
    main()
