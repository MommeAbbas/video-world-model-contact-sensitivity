"""
Thin entry point around the existing final analysis implementation
(analyze_matched_sensitivity.analyze), applied to the isolated-release
subset built by build_isolated_release_subset.py. This is the paper's
Section "Causal Sensitivity Around Contact Transitions" isolated-window
robustness check (14 of 49 pairs, 10 of 20 episodes): beta ~= +2.64,
95% CI ~= [+0.52, +4.97]; Delta D ~= +2.79 (Model 1 / matched-pair estimates
from the combined isolated subset).

Does not duplicate or modify analyze()'s statistical implementation --
these are the exact same calls used for the full (non-isolated) release
datasets, applied to a different, already-constructed input file.
"""
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from analyze_matched_sensitivity import load, analyze


def main():
    rows_exp = load("rq2_release_isolated_exploratory.json")
    analyze(rows_exp, label="ISOLATED EXPLORATORY (n=7 pairs)", treatment_label="release")

    rows_conf = load("rq2_release_isolated_confirmation.json")
    analyze(rows_conf, label="ISOLATED CONFIRMATION (n=7 pairs)", treatment_label="release")

    rows_comb = load("rq2_release_isolated_combined.json")
    analyze(rows_comb, label="ISOLATED COMBINED (n=14 pairs)", treatment_label="release")


if __name__ == "__main__":
    main()
