"""
Go/no-go proof for the causal-splice intervention mechanism (paper Section
"Simulator-Grounded Causal Sensitivity", "Validation" paragraph).

Question: can iVideoGPT generate through a complete predicted frame t, have
that frame replaced with a modified, re-tokenized version, have the causal
token prefix reconstructed, and have a FRESH continuation generation call
from that prefix produce future frames that genuinely differ from the
unmodified continuation?

Procedure, using the real pretrained checkpoint:
  1. Generate normally through frame t (greedy decoding) -> prefix_tokens.
  2. Branch A: continue generation unmodified from prefix_tokens, frames t+1..t+k.
  3. Branch B: decode frame t, apply a large, deliberately strong pixel
     modification (paint a bright solid-color block over most of the frame),
     re-tokenize it, splice it into a NEW prefix tensor (prefix_tokens is not
     mutated), then continue generation from that reconstructed prefix,
     frames t+1..t+k.
  4. Both continuations use continue_from_prefix-style reconstruction: fresh
     embeddings are rebuilt purely from (tokens, actions) with no persisted
     KV-cache or Python object crossing the splice boundary (see
     src/causal_splice.py docstring for why the architecture allows this).
  5. Compare A vs B token-for-token and pixel-for-pixel for frames t+1..t+k.
     A difference is a positive result: later frames causally depend on
     frame t's content.

All greedy (do_sample=False) so both branches are deterministic apart from
the intervention itself. Requires the same real demonstration clip as
validate_checkpoint_loading.py.
"""
import os
import sys

import numpy as np
import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from src.model_loading import load_models, pick_device, CONTEXT_LENGTH, SEGMENT_LENGTH
from src import causal_splice as sl
from validate_checkpoint_loading import load_real_robosuite_clip

STOP_FRAME = 1          # 0-indexed future frame to intervene on (3rd frame overall: 2 context + this)
CONTINUE_FRAMES = 2     # how many further frames to generate in each branch (t+1 .. t+CONTINUE_FRAMES)


def strong_modification(pixels):
    """Deliberately large, valid modification: paint a bright solid-color block
    over the central half of the frame. pixels: (B,3,H,W) in [0,1].
    """
    modified = pixels.clone()
    H, W = pixels.shape[-2], pixels.shape[-1]
    h0, h1 = H // 4, 3 * H // 4
    w0, w1 = W // 4, 3 * W // 4
    modified[:, 0, h0:h1, w0:w1] = 1.0  # bright red channel
    modified[:, 1, h0:h1, w0:w1] = 0.0
    modified[:, 2, h0:h1, w0:w1] = 0.0
    return modified


def to_uint8(frame_bchw):
    return (frame_bchw.clamp(0, 1).detach().float().cpu().numpy() * 255).astype(np.uint8)


def main():
    device = pick_device()
    print(f"Using device: {device}")

    tokenizer, model = load_models(device)
    imgs, actions = load_real_robosuite_clip()
    pixel_values = imgs.unsqueeze(0).to(device)   # 1,T,C,H,W
    action_t = actions.unsqueeze(0).to(device)    # 1,T,4
    context_pixel_values = pixel_values[:, :CONTEXT_LENGTH]

    with torch.no_grad():
        tokens, _ = tokenizer.tokenize(pixel_values, CONTEXT_LENGTH)
    gen_input = tokens[:, :CONTEXT_LENGTH * (16 * 16 + 1)]
    print(f"gen_input shape: {gen_input.shape}")
    print(f"prelude_tokens_num={model.prelude_tokens_num}, tokens_num_per_dyna={model.tokens_num_per_dyna}")

    # --- Step 1: normal generation through frame STOP_FRAME ---
    print(f"\n[1] Generating normally through frame {STOP_FRAME} (0-indexed future frame)...")
    prefix_tokens, prefix_embeds = sl.generate_frames(
        model, gen_input.clone(), action_t, start_frame=0, end_frame_exclusive=STOP_FRAME + 1,
        do_sample=False,
    )
    expected_len = model.prelude_tokens_num + 1 + (STOP_FRAME + 1) * (model.tokens_num_per_dyna + 1)
    assert prefix_tokens.shape[1] == expected_len
    print(f"    prefix_tokens shape: {prefix_tokens.shape} (expected {expected_len})")

    # --- Step 2: splice frame STOP_FRAME with a strong modification ---
    print(f"\n[2] Decoding frame {STOP_FRAME}, applying strong modification, re-tokenizing...")
    spliced_tokens, orig_pixels, modified_pixels = sl.splice_frame(
        model, tokenizer, prefix_tokens, context_pixel_values, STOP_FRAME, strong_modification
    )
    start, end = sl.dyna_slice(model, STOP_FRAME)
    orig_dyna = prefix_tokens[:, start:end]
    new_dyna = spliced_tokens[:, start:end]
    n_changed = (orig_dyna != new_dyna).sum().item()
    print(f"    dyna tokens changed at frame {STOP_FRAME}: {n_changed}/{orig_dyna.numel()}")
    assert not torch.equal(orig_dyna, new_dyna), "re-tokenized frame produced IDENTICAL tokens -- modification too weak or splice wrong"
    # sanity: everything before the spliced frame must be untouched
    assert torch.equal(prefix_tokens[:, :start], spliced_tokens[:, :start]), "splice corrupted earlier tokens"
    assert prefix_tokens.shape == spliced_tokens.shape

    # --- Step 3: FRESH continuation from each prefix (A = unmodified, B = spliced) ---
    end_frame = STOP_FRAME + 1 + CONTINUE_FRAMES
    print(f"\n[3] Fresh continuation A (unmodified prefix), frames {STOP_FRAME + 1}..{end_frame - 1}...")
    tokens_A, _ = sl.generate_frames(
        model, prefix_tokens.clone(), action_t, start_frame=STOP_FRAME + 1, end_frame_exclusive=end_frame,
        do_sample=False, inputs_embeds=None,
    )
    print(f"\n[3] Fresh continuation B (spliced prefix), frames {STOP_FRAME + 1}..{end_frame - 1}...")
    tokens_B, _ = sl.generate_frames(
        model, spliced_tokens.clone(), action_t, start_frame=STOP_FRAME + 1, end_frame_exclusive=end_frame,
        do_sample=False, inputs_embeds=None,
    )

    # --- Step 3b (CONTROL): repeat continuation A from the SAME unmodified prefix
    # a second time, independently reconstructed. If greedy decoding + the fresh
    # embedding reconstruction were nondeterministic on this backend (e.g. MPS
    # float reduction order), this control would show spurious differences and
    # the A-vs-B result above would be uninterpretable. ---
    print(f"\n[3b] CONTROL: repeat continuation A from the identical unmodified prefix...")
    tokens_A_repeat, _ = sl.generate_frames(
        model, prefix_tokens.clone(), action_t, start_frame=STOP_FRAME + 1, end_frame_exclusive=end_frame,
        do_sample=False, inputs_embeds=None,
    )
    control_identical = torch.equal(tokens_A, tokens_A_repeat)
    print(f"    control (A == A_repeat): {control_identical}")
    if not control_identical:
        n_diff = (tokens_A != tokens_A_repeat).sum().item()
        print(f"    WARNING: {n_diff} tokens differ between two runs of the SAME unmodified "
              f"prefix -- generation is not deterministic on this backend/setup. The A-vs-B "
              f"result below cannot be attributed to the splice alone.")

    # --- Step 4: compare ---
    print("\n[4] Comparing continuations A vs B token-for-token, frame by frame:")
    any_diff = False
    for i in range(STOP_FRAME + 1, end_frame):
        s, e = sl.dyna_slice(model, i)
        a = tokens_A[:, s:e]
        b = tokens_B[:, s:e]
        n_diff = (a != b).sum().item()
        print(f"    frame {i}: {n_diff}/{a.numel()} dyna tokens differ")
        any_diff = any_diff or n_diff > 0

    # detokenize expects the HeadModelWithAction.generate() convention of
    # dropping the trailing (not-yet-consumed) sdf token.
    with torch.no_grad():
        recon_A = tokenizer.detokenize(tokens_A[:, :-1], CONTEXT_LENGTH).clamp(0, 1)
        recon_B = tokenizer.detokenize(tokens_B[:, :-1], CONTEXT_LENGTH).clamp(0, 1)
    pixel_diffs = []
    for i in range(STOP_FRAME + 1, end_frame):
        # recon_* frame axis is [context frames][future frames], so future
        # frame index i (0-indexed, matching sl.dyna_slice's convention) sits
        # at absolute index CONTEXT_LENGTH + i.
        d = (recon_A[:, CONTEXT_LENGTH + i] - recon_B[:, CONTEXT_LENGTH + i]).abs().mean().item()
        pixel_diffs.append(d)
        print(f"    frame {i}: mean abs pixel diff A vs B = {d:.5f}")

    out_dir = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "outputs")
    os.makedirs(out_dir, exist_ok=True)
    np.save(os.path.join(out_dir, "splice_orig_frame.npy"), to_uint8(orig_pixels))
    np.save(os.path.join(out_dir, "splice_modified_frame.npy"), to_uint8(modified_pixels))
    np.save(os.path.join(out_dir, "splice_recon_A.npy"), to_uint8(recon_A[0]))
    np.save(os.path.join(out_dir, "splice_recon_B.npy"), to_uint8(recon_B[0]))
    print(f"\nSaved frame arrays to {out_dir}")

    print()
    if not control_identical:
        print("RESULT: INCONCLUSIVE -- the determinism control failed (see WARNING above), so the "
              "A-vs-B divergence cannot be attributed to the splice with confidence. STOP and diagnose "
              "backend nondeterminism before proceeding.")
    elif any_diff and max(pixel_diffs) > 1e-4:
        print("RESULT: PASS -- (1) two continuations from an IDENTICAL unmodified "
              "prefix are bit-identical (deterministic control passed), and (2) splicing a modified, "
              "re-tokenized frame t into the causal prefix and starting a fresh continuation call from "
              "there causes later generated frames to differ from the unmodified continuation.")
    else:
        print("RESULT: FAILURE -- later frames did NOT change after the splice. "
              "The architecture may not support this intervention as implemented, or the "
              "modification/splice did not actually alter model input. STOP and diagnose.")


if __name__ == "__main__":
    main()
