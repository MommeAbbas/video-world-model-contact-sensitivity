"""
Core causal-splice intervention primitives: the fixed context/action/future-
frame window used to anchor every intervention (build_segment), the token
prefix construction that splices a clean simulator-rendered intervention frame
into an otherwise real context (build_prefix_tokens), and the downstream
token-Hamming divergence measure between two continuations (divergence_curve)
-- the D_i measure described in the paper's Method section.

build_segment defines the fixed window layout (segment_length=12, context=2,
matching the checkpoint): segment_start = t_event - 5, frames =
[segment_start, ..., segment_start+11] (12 absolute timesteps), context =
frames[0], frames[1]. It also applies the one-step action shift needed because
episode logs store action[t] as the action APPLIED to reach frame t from frame
t-1 (a "look-back" convention from env.step() bookkeeping), while
HeadModelWithAction's generate()/forward() convention needs action[t] = the
action taken FROM frame t (a "look-ahead" convention).
"""
import torch

from . import causal_splice as sl
from .model_loading import CONTEXT_LENGTH

import numpy as np


def build_segment(log, t_event):
    """Returns (frames_chw_2x3x64x64_context, full_action_seq_12x4, gt_frames_10x3x64x64)
    for the fixed window anchored at t_event, or None if out of range.
    """
    start = t_event - 5
    end = t_event + 6  # inclusive, 12 absolute timesteps: start..end
    if start < 0 or end > len(log["action"]) - 1:
        return None
    frames = log["model_frame_64"][start:end + 1]  # (12, 3, 64, 64)
    # action look-back -> look-ahead shift: model_action[k] = log_action[start+k+1]
    model_actions = np.zeros((12, 4), dtype=np.float32)
    model_actions[:11] = log["action"][start + 1: start + 12]
    context = frames[:CONTEXT_LENGTH]
    gt_future = frames[CONTEXT_LENGTH:]
    return context, model_actions, gt_future


@torch.no_grad()
def build_prefix_tokens(tokenizer, model, context_pixel_values, frame_chw, device):
    """context_pixel_values: (1,2,3,64,64) real ground-truth context.
    frame_chw: the (clean, simulator-rendered) intervention frame.
    Returns a (1, 530) token tensor: [513 context tokens][sdf][16 dyna] --
    exactly the layout sl.generate_frames expects to continue from.
    """
    dyna = sl.retokenize_single_frame(tokenizer, context_pixel_values, frame_chw.unsqueeze(0).to(device),
                                        CONTEXT_LENGTH)
    # tokenizer.tokenize requires future_length>0 -- reuse the already-tokenized
    # context prefix by tokenizing context+frame_chw as a 1-future-frame clip.
    clip = torch.cat([context_pixel_values.to(device), frame_chw.unsqueeze(0).unsqueeze(0).to(device)], dim=1)
    tokens, _ = tokenizer.tokenize(clip, CONTEXT_LENGTH)
    context_tokens = tokens[:, :CONTEXT_LENGTH * (256 + 1) - 1]  # 513
    sdf_token = torch.full((1, 1), model.token_for_sdf, dtype=context_tokens.dtype, device=device)
    # [context(513)][sdf frame0][dyna frame0 (16)][sdf frame1] -- the trailing
    # sdf marks the boundary for the NEXT frame, matching HeadModelWithAction's
    # own loop convention (action_model.py: dyna tokens are always followed by
    # the next frame's sdf token before the sequence is handed onward), which
    # is what sl.generate_frames(start_frame=1, ...) expects.
    prefix = torch.cat([context_tokens, sdf_token, dyna.to(context_tokens.dtype), sdf_token], dim=1)
    assert prefix.shape[1] == model.prelude_tokens_num + 1 + 1 * (model.tokens_num_per_dyna + 1)
    return prefix


def divergence_curve(model, tokens_A, tokens_B, start_frame, end_frame):
    return [(tokens_A[:, s:e] != tokens_B[:, s:e]).sum().item()
            for i in range(start_frame, end_frame) for s, e in [sl.dyna_slice(model, i)]]
