"""
Eligibility + transition-contamination audit for the event-centered temporal
profile (paper Section "Temporal Structure of Predictive Sensitivity").
Pure contact-boolean window inspection, no model inference. Run and
inspected before any intervention, per the pre-registered eligibility rule:
a record is "clean" only if its [t_i-2, t_i+3] window is in-bounds and
contains no other contact transition besides the event being aligned to.
"""
import json
import os
import pickle
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from src import episode_generation as ep

OUT_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "outputs")
OFFSETS = [-3, -2, -1, 0, 1, 2, 3]


def all_transitions(log, obj_index=0):
    onsets, releases = ep.find_contact_events(log["contact_per_object"], obj_index=obj_index)
    return sorted([(t, "onset") for t in onsets] + [(t, "release") for t in releases])


def audit_batch(episodes, batch_label, obj_index=0):
    records = []
    for seed, log in episodes.items():
        onsets, releases = ep.find_contact_events(log["contact_per_object"], obj_index=obj_index)
        trans = all_transitions(log, obj_index=obj_index)
        ep_len = len(log["action"])
        for etype, event_list in [("onset", onsets), ("release", releases)]:
            for tau in event_list:
                for dt in OFFSETS:
                    t_i = tau + dt
                    ws, we = t_i - 2, t_i + 3
                    eligible = ws >= 0 and we <= ep_len - 1
                    contaminated = None
                    neighbor = None
                    if eligible:
                        others = [(t, tt) for (t, tt) in trans if ws <= t <= we and t != tau]
                        contaminated = len(others) > 0
                        neighbor = others[0] if others else None
                    records.append({
                        "batch": batch_label, "event_type": etype, "seed": int(seed), "tau": int(tau),
                        "dt": dt, "t_i": int(t_i), "eligible": bool(eligible),
                        "contaminated": contaminated, "neighbor": neighbor, "clean": eligible and not contaminated,
                    })
    return records


def summarize(records):
    from collections import defaultdict
    counts = defaultdict(lambda: {"eligible": 0, "clean": 0})
    for r in records:
        key = (r["event_type"], r["batch"], r["dt"])
        if r["eligible"]:
            counts[key]["eligible"] += 1
            if r["clean"]:
                counts[key]["clean"] += 1
    return counts


def main():
    with open(os.path.join(OUT_DIR, "pilot_episodes.pkl"), "rb") as f:
        exploratory = pickle.load(f)
    with open(os.path.join(OUT_DIR, "pilot_episodes_confirmation.pkl"), "rb") as f:
        confirmation = pickle.load(f)

    records = audit_batch(exploratory, "exploratory") + audit_batch(confirmation, "confirmation")
    counts = summarize(records)

    for etype in ["onset", "release"]:
        print(f"\n=== {etype.upper()} ===")
        print(f"{'offset':>7s} | {'explor elig/clean':>18s} | {'confirm elig/clean':>19s} | {'combined clean':>14s}")
        for dt in OFFSETS:
            e = counts[(etype, "exploratory", dt)]
            c = counts[(etype, "confirmation", dt)]
            print(f"{dt:>7d} | {e['eligible']:>7d} / {e['clean']:>7d}         | "
                  f"{c['eligible']:>7d} / {c['clean']:>7d}          | {e['clean']+c['clean']:>14d}")

    with open(os.path.join(OUT_DIR, "rq2_event_centered_audit.json"), "w") as f:
        json.dump(records, f, indent=2)
    print(f"\nSaved {len(records)} audit records to outputs/rq2_event_centered_audit.json")


if __name__ == "__main__":
    main()
