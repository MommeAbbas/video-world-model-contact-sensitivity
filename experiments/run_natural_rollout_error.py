"""
Natural rollout-error experiment (paper Section "Contact-Conditioned Rollout
Dynamics"): for each valid contact onset/release event, generate a real
iVideoGPT rollout from a fixed 2-context-frame window anchored 5 steps before
the event, compare each of the 10 predicted future frames against the actual
simulator ground-truth frame at that timestep, and align the resulting error
curve to the event timestep. Also generates matched non-contact (free-motion)
windows at the same horizon structure as a control.

Window layout (segment_length=12, context=2, matches the checkpoint exactly):
  segment_start = t_event - 5
  frames  = [segment_start, ..., segment_start+11]   (12 absolute timesteps)
  context = frames[0], frames[1]
  future  = frames[2..11]  -> 10 predicted frames, relative time k-3 for k=0..9
  so t_event sits at relative time 0 (segment position 5, future index 3).

Action shift: episode logs store action[t] = the action applied to reach
frame t from frame t-1 (a "look-back" convention from env.step() bookkeeping).
The model's HeadModelWithAction.generate()/forward() convention is
action[t] = action taken FROM frame t (a "look-ahead" convention); the
per-segment action fed to the model is a +1 shift of the logged action array
(see src/causal_intervention.py::build_segment).

Error metric: mean absolute pixel error (0-1 scale) between generated and
ground-truth frame, per future position.
"""
import json
import os
import pickle
import sys

import numpy as np
import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src import episode_generation as ep
from src import causal_splice as sl
from src.causal_intervention import build_segment
from src.model_loading import load_models, pick_device, CONTEXT_LENGTH, SEGMENT_LENGTH

PILOT_PATH = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
                           "outputs", "pilot_episodes.pkl")
OUT_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "outputs")


@torch.no_grad()
def rollout_and_score(tokenizer, model, device, full_segment, model_actions):
    # Tokenize the FULL 12-frame segment (context + ground-truth future),
    # then keep only the context tokens as gen_input -- tokenizer.tokenize()
    # requires future_length > 0 internally (compressive_vq_model.py's
    # cond_encoder call), so we can't tokenize context frames alone.
    full_t = torch.from_numpy(full_segment).unsqueeze(0).to(device)  # 1,12,3,64,64
    action_t = torch.from_numpy(model_actions).unsqueeze(0).to(device)  # 1,12,4

    tokens, _ = tokenizer.tokenize(full_t, CONTEXT_LENGTH)
    gen_input = tokens[:, :CONTEXT_LENGTH * (16 * 16 + 1)]

    gen_tokens, _ = sl.generate_frames(
        model, gen_input, action_t, start_frame=0, end_frame_exclusive=SEGMENT_LENGTH - CONTEXT_LENGTH,
        do_sample=False,
    )
    recon = tokenizer.detokenize(gen_tokens[:, :-1], CONTEXT_LENGTH).clamp(0, 1)  # 1,12,3,64,64
    pred_future = recon[0, CONTEXT_LENGTH:].cpu().numpy()  # 10,3,64,64
    return pred_future


def per_frame_mae(pred, gt):
    return np.abs(pred - gt).reshape(pred.shape[0], -1).mean(axis=1)


def find_matched_controls(all_logs, n_needed, exclude_events, rng):
    """Finds windows with the same 12-step structure where NO contact
    transition (onset or release, for the cube) occurs anywhere within
    [t-5, t+6], at a similar rollout horizon (t not too close to episode
    start/end) -- a free-motion control matched in structure, not in the
    same episode/other-object identity.
    """
    candidates = []
    for seed, log in all_logs.items():
        col = log["contact_per_object"][:, 0].astype(bool)
        T = len(col)
        for t in range(5, T - 6):
            if (seed, t) in exclude_events:
                continue
            window = col[t - 5: t + 7]
            if window.any() == window.all():  # all same value: no transition in window
                candidates.append((seed, t))
    rng.shuffle(candidates)
    return candidates[:n_needed]


def main():
    device = pick_device()
    print(f"Using device: {device}")
    tokenizer, model = load_models(device)

    with open(PILOT_PATH, "rb") as f:
        all_logs = pickle.load(f)

    onset_events, release_events = [], []
    for seed, log in all_logs.items():
        onsets, releases = ep.find_contact_events(log["contact_per_object"], obj_index=0)
        for t in onsets:
            if t - 5 >= 0 and t + 6 <= len(log["action"]) - 1:
                onset_events.append((seed, t))
        for t in releases:
            if t - 5 >= 0 and t + 6 <= len(log["action"]) - 1:
                release_events.append((seed, t))

    print(f"Valid onset events: {len(onset_events)}, valid release events: {len(release_events)}")

    rng = np.random.default_rng(0)
    exclude = set(onset_events) | set(release_events)
    control_events = find_matched_controls(all_logs, max(len(onset_events), len(release_events)), exclude, rng)
    print(f"Matched non-contact control windows found: {len(control_events)}")

    def run_batch(events, label):
        curves = []
        for i, (seed, t) in enumerate(events):
            log = all_logs[seed]
            built = build_segment(log, t)
            if built is None:
                continue
            context, model_actions, gt_future = built
            full_segment = np.concatenate([context, gt_future], axis=0)
            pred_future = rollout_and_score(tokenizer, model, device, full_segment, model_actions)
            err = per_frame_mae(pred_future, gt_future)
            curves.append(err)
            print(f"  [{label} {i+1}/{len(events)}] seed={seed} t={t} err={np.round(err,4).tolist()}")
        return np.array(curves)  # (n_events, 10)

    print("\nRunning rollouts for ONSET events...")
    onset_curves = run_batch(onset_events, "onset")
    print("\nRunning rollouts for RELEASE events...")
    release_curves = run_batch(release_events, "release")
    print("\nRunning rollouts for CONTROL (non-contact) windows...")
    control_curves = run_batch(control_events, "control")

    np.savez(os.path.join(OUT_DIR, "rq1_pilot_curves.npz"),
              onset=onset_curves, release=release_curves, control=control_curves)
    with open(os.path.join(OUT_DIR, "rq1_pilot_events.json"), "w") as f:
        json.dump({"onset_events": onset_events, "release_events": release_events,
                   "control_events": control_events}, f)
    print(f"\nSaved curves to {OUT_DIR}/rq1_pilot_curves.npz")


if __name__ == "__main__":
    main()
