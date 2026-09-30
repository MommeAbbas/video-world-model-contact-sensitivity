# Predictive Sensitivity Across Contact Transitions in a Video World Model

This project studies whether an action-conditioned video world model's local
predictive sensitivity to a small state perturbation changes systematically
around robot-object contact transitions. We apply a simulator-grounded,
controlled displacement to a manipulated object and measure how much it
shifts the model's predicted continuation, using
[iVideoGPT](https://github.com/thuml/iVideoGPT) on a from-source RoboSuite
pushing environment reconstructed to match its training setup. Sensitivity
rises through contact onset and falls through release. A matched
intervention at a fixed pre-transition offset finds a release-approach
effect that independently replicates on held-out episodes; the analogous
onset effect does not replicate.

## Method

For a chosen timestep: restore the simulator state, apply a calibrated
displacement to the manipulated object, render and tokenize the resulting
frame, splice it into an otherwise identical autoregressive prefix with
future actions held fixed, and compare the two model continuations. Both
come from the same model and prefix history, so this isolates local
predictive sensitivity to the perturbation, not a causal account of contact.

## Results

- Local predictive sensitivity shows opposing temporal profiles around
  contact transitions: rising through onset, falling through release.
- The release-approach matched effect replicates on held-out confirmation
  episodes (combined: beta ≈ +2.53, 95% CI [+1.61, +3.53]); the analogous
  onset effect does not replicate.
- Natural rollout error alone does not reveal this structure; it takes the
  intervention to see it.

## Repository structure

```
src/                Core environment and intervention implementation
experiments/        Experiment entry points
analysis/           Statistical analyses
figures/            Paper figure generation
validation/         Validation and sanity checks
reference_outputs/  Lightweight outputs from the reported experiments
third_party/        Pinned upstream iVideoGPT submodule
```

## Setup

```bash
git clone --recurse-submodules <repository-url>
cd video-world-model-contact-sensitivity
python3 -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
```

If cloned without `--recurse-submodules`, run
`git submodule update --init --recursive` (pinned to commit
`d601d5cac9e96c6aa0c17cb37ed6a7c7ca1fb210`). Download the pretrained
checkpoint `thuml/ivideogpt-vp2-robosuite-64-act-cond` from the
[iVideoGPT model collection](https://huggingface.co/collections/thuml/ivideogpt-674c59cae32231024d82d6c5)
into `checkpoints/ivideogpt-vp2-robosuite-64-act-cond/` (`tokenizer/`,
`transformer/`); `checkpoints/` is gitignored.

## Reproducing the experiments

Run from the repository root with the checkpoint in place. Exploratory
episodes use seeds 0-9 and the confirmation batch uses seeds 100-109;
environment randomness is not fully seed-determined, so regenerated
episodes will not be bit-exact replicas of the original runs.

**1. Episode generation and calibration**
```bash
python experiments/run_exploratory_episode_generation.py
python experiments/run_perturbation_calibration.py
```

**2. Natural rollout-error analysis**
```bash
python experiments/run_natural_rollout_error.py
python analysis/analyze_natural_rollout_error.py
python analysis/bootstrap_natural_rollout_effect.py
python analysis/analyze_confound_adjusted_rollout.py
python analysis/bootstrap_confound_adjusted_effect.py
```

**3. Matched predictive sensitivity**
```bash
python experiments/run_matched_sensitivity_onset.py
python analysis/analyze_matched_sensitivity.py
python experiments/run_matched_sensitivity_onset_confirmation.py
python analysis/build_onset_combined_dataset.py
python experiments/run_matched_sensitivity_release.py exploratory
python experiments/run_matched_sensitivity_release.py confirmation
python analysis/build_release_combined_dataset.py
python analysis/bootstrap_covariate_adjusted_effect.py
python analysis/build_isolated_release_subset.py
python analysis/analyze_isolated_release_subset.py
```
The combined datasets are analyzed with:
```python
from analysis.analyze_matched_sensitivity import load, analyze
analyze(load("rq2_combined_full.json"), label="onset combined")
analyze(load("rq2_release_combined_full.json"), label="release combined", treatment_label="release")
```

**4. Event-centered sensitivity and interpretation**
```bash
python analysis/audit_event_centered_eligibility.py
python experiments/run_event_centered_sensitivity.py
python analysis/analyze_event_centered_sensitivity.py
python validation/validate_cube_tracker.py
python analysis/select_qualitative_examples.py
python analysis/analyze_physical_interpretability.py
```

**5. Figures**
```bash
python figures/plot_paper_figures.py
python figures/plot_qualitative_rollouts.py
```

## Validation

Standalone sanity checks, independent of the statistical pipeline (the first
two require a real RoboSuite demonstration clip, per the format documented
in `validate_checkpoint_loading.py`):
```bash
python validation/validate_checkpoint_loading.py
python validation/validate_causal_splice.py
python validation/validate_cube_tracker.py
```

## Reference outputs

`reference_outputs/` contains lightweight summaries, validation artifacts,
and final figures so the reported results can be inspected directly; large
raw episode logs and per-intervention records are excluded.

## Upstream

`third_party/iVideoGPT` is an unmodified, pinned submodule of
[thuml/iVideoGPT](https://github.com/thuml/iVideoGPT), using the pretrained
checkpoint `thuml/ivideogpt-vp2-robosuite-64-act-cond`. This repository
contains the environment reconstruction, the intervention pipeline, and the
experiments, analyses, and validation checks built on top of it.

## Citation

```bibtex
@inproceedings{
abbas2026predictive,
title={Predictive Sensitivity Across Contact Transitions in a Video World Model},
author={Mohammed Abbas},
booktitle={NeurIPS 2026 Workshop on Physical Understanding for Decision-Making: Bridging Foundation Models and Reliable Agents},
year={2026},
url={https://openreview.net/forum?id=gVEyYKFVac}
}
```

## License

This repository's own code is MIT licensed (`LICENSE`). `third_party/iVideoGPT`
remains under its own upstream license (`third_party/iVideoGPT/LICENSE`).
