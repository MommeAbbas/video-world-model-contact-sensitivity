"""
Perturbation-instrument calibration sweep (paper Section "Simulator-Grounded
Causal Sensitivity" -- "our calibration sweep shows that the mapping depends
on direction and depth").

For the cube (push_center_multi.py's primary/largest object -- see
src/simulator_interface.py::object_joint_name), at each of 3 depths
(near/middle/far, chosen to span the task's valid placement range along the
camera's actual depth axis) and 3 camera-relative directions (lateral X,
lateral/vertical Y, depth Z -- from actual camera extrinsics, not assumed
world axes), sweep physical displacement epsilon in
{1,2,5,10,20,40,60,80} mm and measure:

  physical: requested vs actual object translation (simulator ground truth)
  image:    projected pixel displacement (analytic, via camera geometry) +
            whole-frame MAE (secondary, coarse signal only)
  token:    dyna-token Hamming distance / fraction changed / which of the
            4x4 grid cells changed

Protocol per trial: restore identical baseline state -> modify ONLY the
target object's translation -> mujoco.mj_forward() -> render -> tokenize.
mj_step() is never called.

Controls included: (A) repeated restore+render+tokenize of the same baseline
state per depth (determinism), (B) actual vs requested displacement
(simulator ground truth), (C) diff of full qpos vectors before/after to
confirm nothing else changed, (D) a single render/preprocess code path used
throughout (src/simulator_interface.py::render_raw + preprocess_to_model_input).
"""
import json
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src import simulator_interface as lib
from src.model_loading import load_models, pick_device

EPSILONS_MM = [1, 2, 5, 10, 20, 40, 60, 80]
OBJ_INDEX = 0  # "cube"
OUT_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "outputs")


def choose_depth_states(env):
    """Pick 3 object x-positions spanning the placement sampler's valid
    x_range=[-0.1, 0.3] (push_center_multi.py:479), scored by actual camera
    depth (dot product with the camera's real depth axis) rather than
    assumed. y=0 (table-center) and z = table surface + z_offset + half-height,
    matching the sampler's own conventions (push_center_multi.py:485-486).
    """
    candidates_x = [-0.08, 0.0, 0.1, 0.2, 0.28]
    z = 0.8 + 0.01 + 0.06  # table_offset_z + z_offset + cube half-size
    cam_pos = env.sim.data.cam_xpos[env.sim.model.camera_name2id(lib.CAMERA_NAME)]
    scored = []
    for x in candidates_x:
        pos = np.array([x, 0.0, z])
        # Euclidean distance to camera -- unambiguous "near/far" regardless of
        # the depth axis's sign convention.
        dist = float(np.linalg.norm(pos - cam_pos))
        scored.append((dist, x, pos))
    scored.sort(key=lambda t: t[0])  # ascending distance: index 0 = nearest camera
    nearest, lo_mid, mid, hi_mid, farthest = scored
    return (
        {"near": nearest[2], "middle": mid[2], "far": farthest[2]},
        {"near": nearest[0], "middle": mid[0], "far": farthest[0]},
    )


def token_grid_diff(base_tokens, pert_tokens):
    b = base_tokens.reshape(4, 4)
    p = pert_tokens.reshape(4, 4)
    diff = (b != p)
    changed_cells = [[int(r), int(c)] for r, c in zip(*np.where(diff))]
    return int(diff.sum()), changed_cells


def run():
    device = pick_device()
    print(f"Using device: {device}")
    tokenizer, model = load_models(device)

    env = lib.build_env(seed=0)
    depth_positions, depth_scores = choose_depth_states(env)
    print("Depth states (world xyz) and raw camera-depth score:")
    for k in depth_positions:
        print(f"  {k}: pos={depth_positions[k]}, depth_score={depth_scores[k]:.4f}")

    axes = lib.camera_axes_world(env)
    directions = {
        "lateral_x": axes["lateral_x"] / np.linalg.norm(axes["lateral_x"]),
        "lateral_y": axes["lateral_y"] / np.linalg.norm(axes["lateral_y"]),
        "depth_z": axes["depth_z"] / np.linalg.norm(axes["depth_z"]),
    }
    print("Camera-relative unit direction vectors (world frame):")
    for k, v in directions.items():
        print(f"  {k}: {v}")

    rows = []
    control_rows = []

    for depth_name, base_pos in depth_positions.items():
        lib.set_object_world_pos(env, base_pos, OBJ_INDEX)
        baseline_snap = lib.snapshot_state(env)
        baseline_qpos_full = baseline_snap["qpos"].copy()

        img_base = lib.render_raw(env)
        frame_base = lib.preprocess_to_model_input(img_base).to(device)
        context_base = frame_base.unsqueeze(0).unsqueeze(0).repeat(1, 2, 1, 1, 1)
        dyna_base = lib.tokenize_single_frame_dyna(tokenizer, context_base, frame_base, device)[0].cpu().numpy()
        pix_base = lib.world_to_pixel(env, base_pos)

        # --- Control A: repeat restore+render+tokenize of the SAME baseline ---
        lib.restore_state(env, baseline_snap)
        img_base2 = lib.render_raw(env)
        frame_base2 = lib.preprocess_to_model_input(img_base2).to(device)
        dyna_base2 = lib.tokenize_single_frame_dyna(tokenizer, context_base, frame_base2, device)[0].cpu().numpy()
        control_rows.append({
            "depth": depth_name,
            "pixel_mae_repeat": float(np.abs(img_base.astype(np.float32) - img_base2.astype(np.float32)).mean()),
            "token_hamming_repeat": int((dyna_base != dyna_base2).sum()),
            "qpos_max_abs_diff_after_restore": float(np.abs(baseline_qpos_full - env.sim.get_state().qpos).max()),
        })

        for dir_name, dir_vec in directions.items():
            for eps_mm in EPSILONS_MM:
                eps_m = eps_mm / 1000.0

                # restore identical baseline
                lib.restore_state(env, baseline_snap)
                qpos_before_perturb = env.sim.get_state().qpos.copy()

                # perturb ONLY the target object's translation
                new_pos = base_pos + eps_m * dir_vec
                lib.set_object_world_pos(env, new_pos, OBJ_INDEX)

                # Control C: confirm nothing else in qpos changed except this object's xyz
                qpos_after_perturb = env.sim.get_state().qpos.copy()
                addr, _ = lib.get_qpos_addr(env, OBJ_INDEX)
                mask = np.ones_like(qpos_after_perturb, dtype=bool)
                mask[addr[0]:addr[0] + 3] = False  # allow the xyz slot to differ
                other_state_changed = float(np.abs(
                    (qpos_after_perturb - qpos_before_perturb)[mask]
                ).max())

                actual_pos = lib.get_object_world_pos(env, OBJ_INDEX)
                actual_disp_mm = float(np.linalg.norm(actual_pos - base_pos) * 1000.0)

                img_pert = lib.render_raw(env)
                frame_pert = lib.preprocess_to_model_input(img_pert).to(device)
                dyna_pert = lib.tokenize_single_frame_dyna(
                    tokenizer, context_base, frame_pert, device
                )[0].cpu().numpy()

                pix_pert = lib.world_to_pixel(env, actual_pos)
                pixel_disp = float(np.linalg.norm(pix_pert.astype(np.float64) - pix_base.astype(np.float64)))
                whole_image_mae = float(np.abs(img_pert.astype(np.float32) - img_base.astype(np.float32)).mean())

                n_changed, changed_cells = token_grid_diff(dyna_base, dyna_pert)

                rows.append({
                    "depth": depth_name, "direction": dir_name,
                    "epsilon_requested_mm": eps_mm,
                    "epsilon_actual_mm": actual_disp_mm,
                    "other_qpos_max_abs_diff": other_state_changed,
                    "pixel_displacement_projected": pixel_disp,
                    "whole_image_mae": whole_image_mae,
                    "n_dyna_tokens_changed": n_changed,
                    "frac_dyna_tokens_changed": n_changed / 16.0,
                    "changed_grid_cells": changed_cells,
                    "base_pixel_rc": pix_base.tolist(),
                    "pert_pixel_rc": pix_pert.tolist(),
                })
                print(f"[{depth_name:6s} | {dir_name:10s} | {eps_mm:3d}mm] "
                      f"actual={actual_disp_mm:.3f}mm pix_disp={pixel_disp:.2f}px "
                      f"tokens_changed={n_changed}/16 mae={whole_image_mae:.4f}")

    os.makedirs(OUT_DIR, exist_ok=True)
    with open(os.path.join(OUT_DIR, "rq2_calibration.json"), "w") as f:
        json.dump({"trials": rows, "controls": control_rows,
                   "depth_positions": {k: v.tolist() for k, v in depth_positions.items()},
                   "directions": {k: v.tolist() for k, v in directions.items()}}, f, indent=2)

    import csv
    csv_path = os.path.join(OUT_DIR, "rq2_calibration.csv")
    fieldnames = [k for k in rows[0].keys() if k != "changed_grid_cells"] + ["changed_grid_cells"]
    with open(csv_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for r in rows:
            r2 = dict(r)
            r2["changed_grid_cells"] = json.dumps(r2["changed_grid_cells"])
            writer.writerow(r2)

    print(f"\nSaved {len(rows)} trials to {csv_path}")
    print("\nControl A (determinism of repeated baseline restore+render+tokenize):")
    for c in control_rows:
        print(f"  {c}")


if __name__ == "__main__":
    run()
