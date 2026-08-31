"""
Combines the onset exploratory and confirmation batches into
outputs/rq2_combined_full.json, re-indexing confirmation pair_ids to avoid
collision with exploratory pair_ids. This is the dataset behind the paper's
Table 1 onset-combined row (64 pairs, 128 events, 20 episodes): Model 1
beta_contact ~= -0.95, 95% cluster CI ~= [-1.74, -0.19]; matched-pair
Delta D ~= -0.83, cluster CI ~= [-1.78, +0.03]. Pure join/re-index of two
already-computed result files, mirroring build_release_combined_dataset.py.
"""
import copy
import json
import os

OUT_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "outputs")


def main():
    with open(os.path.join(OUT_DIR, "rq2_scaled_full.json")) as f:
        orig = json.load(f)
    with open(os.path.join(OUT_DIR, "rq2_confirmation_full.json")) as f:
        conf = json.load(f)

    max_pair_id = max(r["pair_id"] for r in orig)
    combined = copy.deepcopy(orig)
    for r in conf:
        r2 = copy.deepcopy(r)
        r2["pair_id"] = r["pair_id"] + max_pair_id + 1
        combined.append(r2)

    print("orig n_events", len(orig), "n_pairs", len(set(r["pair_id"] for r in orig)))
    print("conf n_events", len(conf), "n_pairs", len(set(r["pair_id"] for r in conf)))
    print("combined n_events", len(combined), "n_pairs", len(set(r["pair_id"] for r in combined)))

    orig_seeds = set(r["seed"] for r in orig)
    conf_seeds = set(r["seed"] for r in conf)
    print("orig seeds", sorted(orig_seeds))
    print("conf seeds", sorted(conf_seeds))
    print("overlap:", orig_seeds & conf_seeds)

    with open(os.path.join(OUT_DIR, "rq2_combined_full.json"), "w") as f:
        json.dump(combined, f, indent=2)
    print("saved combined dataset")


if __name__ == "__main__":
    main()
