"""
Isolated-release-subset construction (paper Section "Causal Sensitivity
Around Contact Transitions": restricting to windows containing only the
target transition). Two steps: isolated_release_events extracts (seed,
t_release) identifiers whose [tau-5, tau+6] window is in bounds and contains
exactly one cube transition; filter_to_isolated_pairs joins those to the
existing release results at the pair level, keeping each event's original
matched control unchanged. No rematching, no new inference, no new pair IDs.
"""
import json
import os
import pickle
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from src import episode_generation as ep

OUT_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "outputs")


# Step 1: identify isolated release events.

def all_transitions(log):
    onsets, releases = ep.find_contact_events(log["contact_per_object"], obj_index=0)
    return sorted([(t, "onset") for t in onsets] + [(t, "release") for t in releases])


def isolated_release_events(episodes):
    """Returns list of (seed, t) for release events whose [t-5,t+6] window is
    (a) in bounds (t-5>=0 and t+6<=episode_length-1) and (b) contains exactly
    one cube transition (the release itself).
    """
    isolated = []
    for seed, log in episodes.items():
        _, releases = ep.find_contact_events(log["contact_per_object"], obj_index=0)
        trans = all_transitions(log)
        ep_len = len(log["action"])
        for tau in releases:
            ws, we = tau - 5, tau + 6
            if ws < 0 or we > ep_len - 1:
                continue
            n_in_window = sum(1 for (t, _) in trans if ws <= t <= we)
            if n_in_window == 1:
                isolated.append((seed, tau))
    return isolated


# Step 2: join isolated identifiers to existing release results.

def load(name):
    with open(os.path.join(OUT_DIR, name)) as f:
        return json.load(f)


def filter_to_isolated_pairs(rows, isolated_ids):
    """rows: full release+control record list for one batch.
    isolated_ids: list of {"seed":..,"t":..} for isolated release events.
    Returns the subset of rows (release + its existing paired control)
    whose release event matches one of isolated_ids, by (seed,t) join.
    """
    iso_set = {(d["seed"], d["t"]) for d in isolated_ids}
    release_rows = [r for r in rows if r["event_type"] == "release"]
    matched_release = [r for r in release_rows if (r["seed"], r["t"]) in iso_set]
    matched_pair_ids = {r["pair_id"] for r in matched_release}
    # sanity: exactly one release row per isolated (seed,t) should match
    assert len(matched_release) == len(iso_set), (
        f"join mismatch: {len(matched_release)} release rows matched vs {len(iso_set)} requested ids, "
        f"matched=({[(r['seed'], r['t']) for r in matched_release]}) requested=({sorted(iso_set)})")
    kept = [r for r in rows if r["pair_id"] in matched_pair_ids]
    return kept, matched_pair_ids


def print_audit_table(rows, batch_label):
    pair_ids = sorted(set(r["pair_id"] for r in rows))
    print(f"\nRetained-pair audit table: {batch_label}")
    header = (f"{'pair_id':>7s} {'rel_seed':>8s} {'rel_t':>6s} {'ctrl_seed':>9s} {'ctrl_t':>7s} "
              f"{'match_dist':>10s} {'ham_rel':>7s} {'ham_ctrl':>8s} {'D_rel':>7s} {'D_ctrl':>7s} {'DeltaD':>8s}")
    print(header)
    deltas = []
    episodes_repr = set()
    for pid in pair_ids:
        pr = [r for r in rows if r["pair_id"] == pid]
        rel = [r for r in pr if r["event_type"] == "release"][0]
        ctl = [r for r in pr if r["event_type"] == "control"][0]
        dd = rel["mean_downstream_divergence"] - ctl["mean_downstream_divergence"]
        deltas.append((pid, rel["seed"], dd))
        episodes_repr.add(rel["seed"])
        print(f"{pid:>7d} {rel['seed']:>8d} {rel['t']:>6d} {ctl['seed']:>9d} {ctl['t']:>7d} "
              f"{rel['match_distance']:>10.4f} {rel['injection_hamming']:>7d} {ctl['injection_hamming']:>8d} "
              f"{rel['mean_downstream_divergence']:>7.2f} {ctl['mean_downstream_divergence']:>7.2f} {dd:>+8.3f}")
    print(f"\nn retained pairs = {len(pair_ids)}, distinct release episodes = {len(episodes_repr)}")
    import collections
    per_ep = collections.Counter(r["seed"] for r in rows if r["event_type"] == "release")
    print(f"retained pairs per episode: {dict(sorted(per_ep.items()))}")
    return deltas


def main():
    with open(os.path.join(OUT_DIR, "pilot_episodes.pkl"), "rb") as f:
        exploratory = pickle.load(f)
    with open(os.path.join(OUT_DIR, "pilot_episodes_confirmation.pkl"), "rb") as f:
        confirmation = pickle.load(f)

    iso_exp = isolated_release_events(exploratory)
    iso_conf = isolated_release_events(confirmation)

    print("Isolated (transition-clean) release events, window=[tau-5,tau+6]:")
    print(f"\nexploratory (seeds 0-9): {len(iso_exp)} isolated release events")
    for seed, t in iso_exp:
        print(f"  (seed={seed}, t={t})")
    print(f"\nconfirmation (seeds 100-109): {len(iso_conf)} isolated release events")
    for seed, t in iso_conf:
        print(f"  (seed={seed}, t={t})")
    print(f"\ncombined: {len(iso_exp) + len(iso_conf)} isolated release events")

    ids = {
        "exploratory": [{"seed": int(s), "t": int(t)} for s, t in iso_exp],
        "confirmation": [{"seed": int(s), "t": int(t)} for s, t in iso_conf],
    }
    with open(os.path.join(OUT_DIR, "rq2_isolated_release_ids.json"), "w") as f:
        json.dump(ids, f, indent=2)
    print("\nSaved identifiers to outputs/rq2_isolated_release_ids.json")

    exp_full = load("rq2_release_exploratory_full.json")
    conf_full = load("rq2_release_confirmation_full.json")

    exp_kept, exp_pids = filter_to_isolated_pairs(exp_full, ids["exploratory"])
    conf_kept, conf_pids = filter_to_isolated_pairs(conf_full, ids["confirmation"])

    print(f"exploratory: {len(exp_pids)} isolated pairs retained (of {len(set(r['pair_id'] for r in exp_full))} total)")
    print(f"confirmation: {len(conf_pids)} isolated pairs retained (of {len(set(r['pair_id'] for r in conf_full))} total)")

    print_audit_table(exp_kept, "exploratory isolated subset")
    print_audit_table(conf_kept, "confirmation isolated subset")

    # Re-index confirmation pair_id to avoid collision with exploratory pair_ids.
    import copy
    max_pid = max(r["pair_id"] for r in exp_kept)
    combined = copy.deepcopy(exp_kept)
    for r in conf_kept:
        r2 = copy.deepcopy(r)
        r2["pair_id"] = r["pair_id"] + max_pid + 1
        combined.append(r2)

    exp_seeds = set(r["seed"] for r in exp_kept)
    conf_seeds = set(r["seed"] for r in conf_kept)
    print(f"\nSeed disjointness check: exploratory seeds={sorted(exp_seeds)}, "
          f"confirmation seeds={sorted(conf_seeds)}, overlap={exp_seeds & conf_seeds}")

    with open(os.path.join(OUT_DIR, "rq2_release_isolated_exploratory.json"), "w") as f:
        json.dump(exp_kept, f, indent=2)
    with open(os.path.join(OUT_DIR, "rq2_release_isolated_confirmation.json"), "w") as f:
        json.dump(conf_kept, f, indent=2)
    with open(os.path.join(OUT_DIR, "rq2_release_isolated_combined.json"), "w") as f:
        json.dump(combined, f, indent=2)
    print(f"\nSaved isolated-subset datasets: "
          f"{len(exp_kept)} exploratory events, {len(conf_kept)} confirmation events, "
          f"{len(combined)} combined events ({len(set(r['pair_id'] for r in combined))} pairs)")


if __name__ == "__main__":
    main()
