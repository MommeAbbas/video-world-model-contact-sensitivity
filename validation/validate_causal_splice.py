"""Validate that a re-tokenized frame splice affects subsequent iVideoGPT
predictions while identical-prefix continuations remain deterministic.

Generates through frame t, then branches: A continues unmodified; B splices
in a strongly modified, re-tokenized version of frame t (a fresh prefix
tensor, not mutating A's) and continues from there. Both continuations
rebuild embeddings from scratch (tokens, actions only, no persisted
KV-cache; see src/causal_splice.py). A determinism control repeats A's
continuation independently. All greedy decoding. Requires the same real
demonstration clip as validate_checkpoint_loading.py.
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

    print(f"\nGenerating normally through frame {STOP_FRAME}...")
    prefix_tokens, prefix_embeds = sl.generate_frames(
        model, gen_input.clone(), action_t, start_frame=0, end_frame_exclusive=STOP_FRAME + 1,
        do_sample=False,
    )
    expected_len = model.prelude_tokens_num + 1 + (STOP_FRAME + 1) * (model.tokens_num_per_dyna + 1)
    assert prefix_tokens.shape[1] == expected_len
    print(f"    prefix_tokens shape: {prefix_tokens.shape} (expected {expected_len})")

    print(f"\nDecoding frame {STOP_FRAME}, applying strong modification, re-tokenizing...")
    spliced_tokens, orig_pixels, modified_pixels = sl.splice_frame(
        model, tokenizer, prefix_tokens, context_pixel_values, STOP_FRAME, strong_modification
    )
    start, end = sl.dyna_slice(model, STOP_FRAME)
    orig_dyna = prefix_tokens[:, start:end]
    new_dyna = spliced_tokens[:, start:end]
    n_changed = (orig_dyna != new_dyna).sum().item()
    print(f"    dyna tokens changed at frame {STOP_FRAME}: {n_changed}/{orig_dyna.numel()}")
    assert not torch.equal(orig_dyna, new_dyna), "re-tokenized frame produced identical tokens: modification too weak or splice wrong"
    # Everything before the spliced frame must be untouched.
    assert torch.equal(prefix_tokens[:, :start], spliced_tokens[:, :start]), "splice corrupted earlier tokens"
    assert prefix_tokens.shape == spliced_tokens.shape

    end_frame = STOP_FRAME + 1 + CONTINUE_FRAMES
    print(f"\nFresh continuation A (unmodified prefix), frames {STOP_FRAME + 1}..{end_frame - 1}...")
    tokens_A, _ = sl.generate_frames(
        model, prefix_tokens.clone(), action_t, start_frame=STOP_FRAME + 1, end_frame_exclusive=end_frame,
        do_sample=False, inputs_embeds=None,
    )
    print(f"\nFresh continuation B (spliced prefix), frames {STOP_FRAME + 1}..{end_frame - 1}...")
    tokens_B, _ = sl.generate_frames(
        model, spliced_tokens.clone(), action_t, start_frame=STOP_FRAME + 1, end_frame_exclusive=end_frame,
        do_sample=False, inputs_embeds=None,
    )

    # Determinism check: repeat A's continuation independently. If greedy
    # decoding weren't deterministic on this backend, this would show
    # spurious differences and the A-vs-B result would be uninterpretable.
    print(f"\nDeterminism check: repeat continuation A from the identical unmodified prefix...")
    tokens_A_repeat, _ = sl.generate_frames(
        model, prefix_tokens.clone(), action_t, start_frame=STOP_FRAME + 1, end_frame_exclusive=end_frame,
        do_sample=False, inputs_embeds=None,
    )
    control_identical = torch.equal(tokens_A, tokens_A_repeat)
    print(f"    control (A == A_repeat): {control_identical}")
    if not control_identical:
        n_diff = (tokens_A != tokens_A_repeat).sum().item()
        print(f"    warning: {n_diff} tokens differ between two runs of the same unmodified "
              f"prefix; generation is not deterministic on this backend/setup, so the A-vs-B "
              f"result below cannot be attributed to the splice alone.")

    print("\nComparing continuations A vs B token-for-token, frame by frame:")
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
        # Future frame i is at CONTEXT_LENGTH + i in the reconstruction.
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
        print("Result: inconclusive. The determinism control failed, so the A-vs-B "
              "divergence cannot be attributed to the splice with confidence.")
    elif any_diff and max(pixel_diffs) > 1e-4:
        print("Result: pass. Repeated continuations from an identical unmodified prefix are "
              "bit-identical, and splicing a modified, re-tokenized frame into the causal prefix "
              "causes later generated frames to differ from the unmodified continuation.")
    else:
        print("Result: fail. Later frames did not change after the splice.")


if __name__ == "__main__":
    main()
