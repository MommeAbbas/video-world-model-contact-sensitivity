"""
Checkpoint sanity check: load the pretrained iVideoGPT checkpoint and run one
normal (unmodified) generation pass on a real RoboSuite clip, confirming the
architecture, checkpoint, and local device (MPS or CPU) work together cleanly
(no NaN/Inf) before any perturbation experiment is run.

Requires a real RoboSuite action-conditioned demonstration clip, which is not
part of this repository (see README "Environment / data generation"). Point
DEMO_NPZ at any clip with the same (observation, action) layout as the
original VP2/RoboSuite pushing task; this check is a one-time sanity check,
not part of the reproduction pipeline for any reported paper result.
"""
import os
import sys
import time

import numpy as np
import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from src.model_loading import load_models, pick_device, CONTEXT_LENGTH, SEGMENT_LENGTH, RESOLUTION

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DEMO_NPZ = os.path.join(REPO_ROOT, "demo_data", "door-lock", "20240515T053942_2_100.npz")


def load_real_robosuite_clip(segment_length=SEGMENT_LENGTH, resolution=RESOLUTION, start=0):
    """mbrl/demonstrations/*.npz store 'observation' as (T, 9, H, W) frame-stacked
    uint8 (3 stacked RGB frames per timestep; channels [6:9] are the current frame,
    verified by exact pixel match against the previous timestep's [3:6]/[0:3] slices).
    'action' is (T, 4) float32 -- real robosuite delta-pose/gripper actions.
    """
    d = np.load(DEMO_NPZ, allow_pickle=True)
    obs = d["observation"]  # (T, 9, H, W) uint8
    act = d["action"]  # (T, 4) float32
    imgs = obs[start:start + segment_length, 6:9]  # (T, 3, H, W) current-frame channels
    imgs = torch.from_numpy(imgs).float() / 255.0  # T,C,H,W in [0,1]
    if imgs.shape[-1] != resolution:
        imgs = torch.nn.functional.interpolate(imgs, size=(resolution, resolution), mode="bilinear",
                                                 align_corners=False)
    actions = torch.from_numpy(act[start:start + segment_length]).float()
    return imgs, actions  # (T,C,H,W), (T,4)


def check_finite(name, t):
    t = t.float()
    n_nan = torch.isnan(t).sum().item()
    n_inf = torch.isinf(t).sum().item()
    print(f"  [{name}] shape={tuple(t.shape)} dtype={t.dtype} nan={n_nan} inf={n_inf} "
          f"min={t.min().item():.4f} max={t.max().item():.4f}")
    return n_nan == 0 and n_inf == 0


def main():
    device = pick_device()
    print(f"Using device: {device}")

    print("Loading tokenizer + HeadModelWithAction from pretrained checkpoint...")
    t0 = time.time()
    tokenizer, model = load_models(device)
    print(f"  loaded in {time.time() - t0:.1f}s")

    print("Loading real RoboSuite demonstration clip (real pixels + real actions)...")
    imgs, actions = load_real_robosuite_clip()
    print(f"  imgs {imgs.shape}, actions {actions.shape}")

    pixel_values = imgs.unsqueeze(0).to(device)  # 1,T,C,H,W
    actions_t = actions.unsqueeze(0).to(device)  # 1,T,4

    print("Tokenizing full clip (context + future) to get ground-truth token layout...")
    with torch.no_grad():
        tokens, labels = tokenizer.tokenize(pixel_values, CONTEXT_LENGTH)
    ok = check_finite("tokens", tokens)
    gen_input = tokens[:, :CONTEXT_LENGTH * (16 * 16 + 1)]
    print(f"  gen_input shape: {gen_input.shape} (expect [1, {CONTEXT_LENGTH * 257}])")

    max_new_tokens = (1 + 4 * 4) * (SEGMENT_LENGTH - CONTEXT_LENGTH) - 1
    print(f"Generating {SEGMENT_LENGTH - CONTEXT_LENGTH} future frames "
          f"(max_new_tokens={max_new_tokens}) with greedy decoding...")
    t0 = time.time()
    with torch.no_grad():
        generated_tokens = model.generate(
            gen_input,
            do_sample=False,
            max_new_tokens=max_new_tokens,
            pad_token_id=50256,
            action=actions_t,
        )
    print(f"  generated in {time.time() - t0:.1f}s")
    ok &= check_finite("generated_tokens", generated_tokens)
    print(f"  generated_tokens shape: {generated_tokens.shape} "
          f"(expect [1, {gen_input.shape[1] + max_new_tokens}])")

    print("Detokenizing generated tokens back to pixel space...")
    with torch.no_grad():
        recon = tokenizer.detokenize(generated_tokens, CONTEXT_LENGTH).clamp(0.0, 1.0)
    ok &= check_finite("recon_frames", recon)
    print(f"  recon shape: {recon.shape} (expect [1, {SEGMENT_LENGTH}, 3, {RESOLUTION}, {RESOLUTION}])")

    print()
    if ok:
        print("RESULT: PASS -- pretrained checkpoint loads, runs generate(), "
              f"and detokenize() cleanly on device={device} with no NaN/Inf.")
    else:
        print("RESULT: FAILURE -- NaN/Inf detected somewhere in the pipeline on "
              f"device={device}. See per-tensor stats above.")


if __name__ == "__main__":
    main()
