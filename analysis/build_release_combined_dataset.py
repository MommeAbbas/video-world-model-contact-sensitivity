"""
Combines the release exploratory and confirmation batches into
outputs/rq2_release_combined_full.json, re-indexing confirmation pair_ids
to avoid collision with exploratory pair_ids. Used by
analyze_matched_sensitivity.py and bootstrap_covariate_adjusted_effect.py
for the paper's combined release statistics (Table 1, n=98/49 pairs/20
episodes). Pure join/re-index of two already-computed result files; no
rematching or new inference.
"""
import copy
import json
import os

OUT_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "outputs")


def main():
    with open(os.path.join(OUT_DIR, "rq2_release_exploratory_full.json")) as f:
        exp = json.load(f)
    with open(os.path.join(OUT_DIR, "rq2_release_confirmation_full.json")) as f:
        conf = json.load(f)

    max_pair_id = max(r["pair_id"] for r in exp)
    combined = copy.deepcopy(exp)
    for r in conf:
        r2 = copy.deepcopy(r)
        r2["pair_id"] = r["pair_id"] + max_pair_id + 1
        combined.append(r2)

    exp_seeds = set(r["seed"] for r in exp)
    conf_seeds = set(r["seed"] for r in conf)
    print("exp seeds", sorted(exp_seeds))
    print("conf seeds", sorted(conf_seeds))
    print("overlap:", exp_seeds & conf_seeds)
    print("exp n_events", len(exp), "n_pairs", len(set(r["pair_id"] for r in exp)))
    print("conf n_events", len(conf), "n_pairs", len(set(r["pair_id"] for r in conf)))
    print("combined n_events", len(combined), "n_pairs", len(set(r["pair_id"] for r in combined)))

    with open(os.path.join(OUT_DIR, "rq2_release_combined_full.json"), "w") as f:
        json.dump(combined, f, indent=2)
    print("saved combined release dataset")


if __name__ == "__main__":
    main()
