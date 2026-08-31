"""
Thin entry point around analyze_matched_sensitivity.analyze, applied to the
isolated-release subset built by build_isolated_release_subset.py: the
paper's isolated-window robustness check (14 of 49 pairs, 10 of 20
episodes), beta ~= +2.64, 95% CI ~= [+0.52, +4.97], Delta D ~= +2.79. Reuses
the exact same calls as the full (non-isolated) release datasets.
"""
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from analyze_matched_sensitivity import load, analyze


def main():
    rows_exp = load("rq2_release_isolated_exploratory.json")
    analyze(rows_exp, label="isolated exploratory (n=7 pairs)", treatment_label="release")

    rows_conf = load("rq2_release_isolated_confirmation.json")
    analyze(rows_conf, label="isolated confirmation (n=7 pairs)", treatment_label="release")

    rows_comb = load("rq2_release_isolated_combined.json")
    analyze(rows_comb, label="isolated combined (n=14 pairs)", treatment_label="release")


if __name__ == "__main__":
    main()
