"""
Benchmarks deterministic continuation runtime for the real iVideoGPT checkpoint
on this machine, for 1/2/4/6/8 future frames, using src/causal_splice.py's
generate_frames (the same reimplementation used and validated in
validate_causal_splice.py). Reports wall-clock time and per-frame cost, and
estimates rough compute for a ~30-50 contact-event experiment. Does NOT run
that experiment -- estimate only. Requires the same real demonstration clip
as validate_checkpoint_loading.py.
"""
import json
import os
import sys
import time

import numpy as np
import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from src import causal_splice as sl
from src.model_loading import load_models, pick_device, CONTEXT_LENGTH
from validate_checkpoint_loading import load_real_robosuite_clip

FRAME_COUNTS = [1, 2, 4, 6, 8]


def main():
    device = pick_device()
    print(f"Using device: {device}")
    tokenizer, model = load_models(device)
    imgs, actions = load_real_robosuite_clip(segment_length=12)
    pixel_values = imgs.unsqueeze(0).to(device)
    action_t = actions.unsqueeze(0).to(device)

    with torch.no_grad():
        tokens, _ = tokenizer.tokenize(pixel_values, CONTEXT_LENGTH)
    gen_input = tokens[:, :CONTEXT_LENGTH * (16 * 16 + 1)]

    results = []
    for n_frames in FRAME_COUNTS:
        torch.manual_seed(0)
        t0 = time.time()
        _, _ = sl.generate_frames(
            model, gen_input.clone(), action_t, start_frame=0, end_frame_exclusive=n_frames,
            do_sample=False,
        )
        elapsed = time.time() - t0
        per_frame = elapsed / n_frames
        results.append({"n_frames": n_frames, "wall_clock_s": elapsed, "s_per_frame": per_frame})
        print(f"n_frames={n_frames}: {elapsed:.2f}s total, {per_frame:.2f}s/frame")

    avg_s_per_frame = np.mean([r["s_per_frame"] for r in results])
    print(f"\nAverage: {avg_s_per_frame:.2f} s/frame (do_sample=False, batch=1, device={device})")

    # Rough estimate for an eventual experiment.
    n_events = 40          # ~30-50 contact events
    n_conditions = 3        # e.g. contact-adjacent perturbed, matched-control perturbed, unperturbed baseline
    n_epsilons = 3          # a small epsilon grid actually used in the real experiment (not the full 8-pt calibration sweep)
    frames_per_continuation = 6   # representative continuation horizon
    total_continuations = n_events * n_conditions * n_epsilons
    total_frame_generations = total_continuations * frames_per_continuation
    est_seconds = total_frame_generations * avg_s_per_frame

    print("\nRough compute estimate (batch=1, greedy):")
    print(f"  {n_events} events x {n_conditions} conditions x {n_epsilons} epsilons "
          f"x {frames_per_continuation} continuation frames = {total_frame_generations} frame-generations")
    print(f"  estimated wall-clock: {est_seconds/60:.1f} min ({est_seconds/3600:.2f} hours)")
    print("  NOTE: batching multiple trials together (model.generate already supports a batch dim)"
          " would reduce this substantially since MPS/CPU cost is not linear in batch size for a"
          " single forward pass; this estimate is the conservative batch=1 serial case.")

    out_path = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "outputs", "timing_benchmark.json")
    with open(out_path, "w") as f:
        json.dump({
            "device": str(device), "results": results, "avg_s_per_frame": avg_s_per_frame,
            "estimate": {
                "n_events": n_events, "n_conditions": n_conditions, "n_epsilons": n_epsilons,
                "frames_per_continuation": frames_per_continuation,
                "total_frame_generations": total_frame_generations,
                "estimated_seconds": est_seconds,
            },
        }, f, indent=2)
    print(f"\nSaved to {out_path}")


if __name__ == "__main__":
    main()
