"""
Validates src/cube_tracker.py against simulator-projected ground truth
before it is used on any decoded/generated rollout frame (paper: "median
error 1.11-1.27px, real versus VQ-reconstructed frames; 1.4% and 0%
detection-failure rates respectively").

Two checks: validate_on_episode tracks real simulator renders (every 3rd
timestep, no model inference); validate_on_vq_reconstruction tracks frames
after a tokenize/detokenize round trip through the checkpoint (every 5th
timestep, first 5 episodes per batch), both compared to the same
ground-truth pixel projection.
"""
import os
import pickle
import sys

import numpy as np
import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src import simulator_interface as lib
from src import causal_splice as sl
from src.cube_tracker import cube_centroid
from src.model_loading import load_models, pick_device, CONTEXT_LENGTH

OUT_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "outputs")
QUAL_DIR = os.path.join(OUT_DIR, "qualitative")


def validate_on_episode(env, log, seed_label, timesteps):
    errors = []
    failures = []
    for t in timesteps:
        frame = log["model_frame_64"][t]  # (3,64,64) float
        gt_pos = log["obj_pos"][t, 0]
        gt_pix = lib.world_to_pixel(env, gt_pos, render_res=64)  # (row, col)
        det = cube_centroid(frame)
        if det is None:
            failures.append(t)
            continue
        err = float(np.hypot(det[0] - gt_pix[0], det[1] - gt_pix[1]))
        errors.append((t, err, gt_pix, det))
    return errors, failures


def validate_on_vq_reconstruction(env, tokenizer, model, device, exploratory, confirmation):
    errors = []
    failures = 0
    n = 0
    with torch.no_grad():
        for batch_name, episodes in [("exploratory", exploratory), ("confirmation", confirmation)]:
            for seed, log in list(episodes.items())[:5]:
                T = len(log["action"])
                for t in range(2, T, 5):
                    context = torch.from_numpy(log["model_frame_64"][t - 2:t]).unsqueeze(0).to(device)
                    frame = torch.from_numpy(log["model_frame_64"][t]).to(device)
                    clip = torch.cat([context, frame.unsqueeze(0).unsqueeze(0)], dim=1)
                    tokens, _ = tokenizer.tokenize(clip, CONTEXT_LENGTH)
                    recon = tokenizer.detokenize(tokens, CONTEXT_LENGTH).clamp(0, 1)[0]  # (3,3,64,64): ctx,ctx,future
                    recon_frame = recon[2].cpu().numpy()

                    gt_pos = log["obj_pos"][t, 0]
                    gt_pix = lib.world_to_pixel(env, gt_pos, render_res=64)
                    det = cube_centroid(recon_frame)
                    n += 1
                    if det is None:
                        failures += 1
                        continue
                    err = float(np.hypot(det[0] - gt_pix[0], det[1] - gt_pix[1]))
                    errors.append(err)
    return np.array(errors), failures, n


def main():
    with open(os.path.join(OUT_DIR, "pilot_episodes.pkl"), "rb") as f:
        exploratory = pickle.load(f)
    with open(os.path.join(OUT_DIR, "pilot_episodes_confirmation.pkl"), "rb") as f:
        confirmation = pickle.load(f)

    env = lib.build_env(seed=0)

    print("Check 1: real simulator renders")
    all_errors = []
    all_failures = 0
    all_n = 0
    for batch_name, episodes in [("exploratory", exploratory), ("confirmation", confirmation)]:
        for seed, log in episodes.items():
            T = len(log["action"])
            timesteps = list(range(0, T, 3))  # sample every 3rd frame across the whole episode
            errors, failures = validate_on_episode(env, log, f"{batch_name}:{seed}", timesteps)
            all_errors.extend([e[1] for e in errors])
            all_failures += len(failures)
            all_n += len(timesteps)

    all_errors = np.array(all_errors)
    print(f"Validated on {all_n} logged frames across {len(exploratory) + len(confirmation)} episodes "
          f"(every 3rd timestep, real simulator renders, known ground-truth cube position).")
    print(f"Detection failures (no red component found): {all_failures}/{all_n} "
          f"({100*all_failures/all_n:.1f}%)")
    print(f"Pixel error (of {len(all_errors)} successful detections, on 64x64 frames):")
    print(f"  mean={all_errors.mean():.2f}px  median={np.median(all_errors):.2f}px  "
          f"p90={np.percentile(all_errors,90):.2f}px  max={all_errors.max():.2f}px")
    print(f"  fraction with error <= 2px: {(all_errors<=2).mean():.2%}")
    print(f"  fraction with error <= 4px: {(all_errors<=4).mean():.2%}")

    os.makedirs(QUAL_DIR, exist_ok=True)
    np.savez(os.path.join(QUAL_DIR, "tracker_validation_errors.npz"), errors=all_errors,
              n_total=all_n, n_failures=all_failures)

    print("\nCheck 2: VQ-reconstructed frames (tokenize/detokenize round trip)")
    device = pick_device()
    tokenizer, model = load_models(device)
    vq_errors, vq_failures, vq_n = validate_on_vq_reconstruction(env, tokenizer, model, device,
                                                                    exploratory, confirmation)
    print(f"VQ-reconstructed frames: n={vq_n}, failures={vq_failures} ({100*vq_failures/vq_n:.1f}%)")
    print(f"pixel error: mean={vq_errors.mean():.2f} median={np.median(vq_errors):.2f} "
          f"p90={np.percentile(vq_errors,90):.2f} max={vq_errors.max():.2f}")
    print(f"fraction <=2px: {(vq_errors<=2).mean():.2%}, <=4px: {(vq_errors<=4).mean():.2%}")
    np.save(os.path.join(QUAL_DIR, "tracker_validation_vqrecon_errors.npy"), vq_errors)


if __name__ == "__main__":
    main()
