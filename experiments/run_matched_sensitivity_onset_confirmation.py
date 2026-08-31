"""
Matched predictive-sensitivity experiment, onset, independent confirmation
batch (paper Section "Causal Sensitivity Around Contact Transitions", Table 1:
Onset / Confirmation). Generates 10 genuinely new episodes (seeds 100-109,
disjoint from the exploratory seeds 0-9) and runs the identical pipeline used
for the exploratory batch (src/episode_generation.py, src/matched_sensitivity.py) --
this batch was not inspected before the exploratory analysis was complete, per
the paper's pre-registered independent-replication rule.

Does not print or compute any aggregate contact-effect estimate during
collection -- only raw per-event numbers. All regression/bootstrap analysis
happens afterward, in analysis/analyze_matched_sensitivity.py.
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

CONFIRMATION_SEEDS = list(range(100, 110))  # 10 new seeds, disjoint from exploratory 0-9


def main():
    device = pick_device()
    print(f"Using device: {device}")
    tokenizer, model = load_models(device)

    print(f"\nGenerating {len(CONFIRMATION_SEEDS)} NEW episodes "
          f"(seeds {CONFIRMATION_SEEDS[0]}-{CONFIRMATION_SEEDS[-1]}), n_cycles=3, "
          f"via src/episode_generation.py::run_episode (unchanged)...")
    episodes = {}
    for seed in CONFIRMATION_SEEDS:
        log = ep.run_episode(seed=seed, n_cycles=3)
        episodes[seed] = log
        onsets, releases = ep.find_contact_events(log["contact_per_object"], obj_index=0)
        valid_onsets = [t for t in onsets if t - 5 >= 0 and t + 6 <= len(log["action"]) - 1]
        print(f"  seed={seed} len={len(log['action'])} onsets={onsets} valid_onsets={valid_onsets}")

    with open(os.path.join(OUT_DIR, "pilot_episodes_confirmation.pkl"), "wb") as f:
        pickle.dump(episodes, f)

    onset_events = []
    for seed, log in episodes.items():
        onsets, _ = ep.find_contact_events(log["contact_per_object"], obj_index=0)
        for t in onsets:
            if t - 5 >= 0 and t + 6 <= len(log["action"]) - 1:
                onset_events.append((seed, t))

    control_pool = all_control_candidates(episodes)
    print(f"\nTotal valid onset events across confirmation episodes: {len(onset_events)}")
    print(f"Control candidate pool size: {len(control_pool)}")

    if len(onset_events) == 0:
        print("NO VALID ONSET EVENTS in the confirmation batch -- reporting transparently, "
              "not altering selection rules to compensate. Stopping here.")
        with open(os.path.join(OUT_DIR, "rq2_confirmation_full.json"), "w") as f:
            json.dump([], f)
        return

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

    with open(os.path.join(OUT_DIR, "rq2_confirmation_full.json"), "w") as f:
        json.dump(results, f, indent=2)
    print(f"\nSaved {len(results)} event records ({len(onset_events)} pairs) to "
          f"outputs/rq2_confirmation_full.json")
    print("\nCollection complete. No contact-effect estimate computed during this run -- "
          "run analysis/analyze_matched_sensitivity.py separately for analysis.")


if __name__ == "__main__":
    main()
