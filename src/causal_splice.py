"""
Reimplementation of the per-frame generation loop used by
ivideogpt.transformer.action_model.HeadModelWithAction.generate() (see that file,
lines 56-121), generalized to support:

  1. stopping generation after an arbitrary intermediate frame t ("prefix" phase), and
  2. resuming generation from an arbitrary reconstructed token prefix ("continuation"
     phase), rebuilding all embeddings fresh from token ids + actions only -- no
     hidden Python/cache state is carried across the stop/resume boundary.

This module does NOT modify ivideogpt/transformer/action_model.py. It calls the
model's public methods (get_input_embeddings, action_linear, llm.generate) --
the exact same primitives HeadModelWithAction.generate() itself uses -- so the
per-frame mechanics are faithfully reproduced rather than approximated.

Architecture facts this relies on (verified by reading action_model.py and
compressive_vq_model.py):

  - HeadModelWithAction.generate() is a hand-written Python loop, NOT a single
    call to transformers' generate(). It calls self.llm.generate() once per
    future frame ("dyna block"), with use_cache=True *scoped to that single call*.
    There is no KV-cache object that persists across frames in Python: each
    iteration recomputes `inputs_embeds` by concatenation
    (action_model.py:113) and calls self.llm.generate(inputs_embeds=...) fresh
    on the whole grown sequence (action_model.py:101-110). This means the
    causal prefix is fully and only represented by the token/embedding
    tensors, not by any persistent cache -- exactly the precondition needed
    for a clean prefix-splice intervention.

  - Action conditioning is injected by an in-place *additive* perturbation of
    one embedding vector per frame (the "sdf" boundary token immediately
    preceding that frame's 16 dyna tokens), not as extra tokens
    (action_model.py:80-81, 174-177). Consequently, reconstructing
    `inputs_embeds` from token ids via get_input_embeddings() alone is NOT
    sufficient after the first frame -- the action-embedding additions for
    every already-generated frame must be replayed. `continue_from_prefix`
    below does this explicitly and deterministically from (tokens, actions)
    alone.

  - Per ivideogpt/vq_model/compressive_vq_model.py `tokenize()`/`detokenize()`,
    each future frame's 16 "dyna" tokens are a VQ code for that frame
    conditioned only on the fixed context frames (cond_features come from the
    context encoder and are simply broadcast-repeated across future_length,
    compressive_vq_model.py:174-191) -- NOT autoregressively conditioned on
    other future frames at the tokenizer level. So a single future frame can
    be tokenized in isolation (context frames + that one frame, future_length=1)
    and will yield the same dyna tokens as if tokenized as part of a longer
    clip. This is what makes "decode frame t -> modify pixels -> re-tokenize
    frame t alone" a valid, non-approximate operation.

Token layout (context_length=2, so prelude_tokens_num=513, tokens_num_per_dyna=16):
  indices [0, 513)   : context tokens (2 context frames x 257 tokens, minus 1
                       dropped leading scf token -- see compressive_vq_model.py:208)
  index   513        : sdf token for frame 0 (also the tensor slot the action
                       for frame 0 is added onto)
  indices [514, 530) : 16 dyna tokens for frame 0
  index   530        : sdf token for frame 1
  indices [531, 547) : 16 dyna tokens for frame 1
  ... (period 17 = tokens_num_per_dyna + 1)

  sdf_pos(i)        = prelude_tokens_num + i * (tokens_num_per_dyna + 1)
  dyna_start(i)     = sdf_pos(i) + 1
  dyna_end(i)       = dyna_start(i) + tokens_num_per_dyna   (exclusive)
  tokens length after frame i complete = prelude_tokens_num + 1 + (i+1)*(tokens_num_per_dyna+1)
"""
import torch


def sdf_pos(model, i):
    return model.prelude_tokens_num + i * (model.tokens_num_per_dyna + 1)


def dyna_slice(model, i):
    start = sdf_pos(model, i) + 1
    end = start + model.tokens_num_per_dyna
    return start, end


@torch.no_grad()
def generate_frames(model, inputs_token, action, start_frame, end_frame_exclusive,
                     do_sample=False, temperature=1.0, top_k=100, pad_token_id=50256,
                     inputs_embeds=None):
    """Generate future frames [start_frame, end_frame_exclusive) (0-indexed among
    predicted frames), appending to inputs_token.

    If inputs_embeds is None, it is rebuilt from scratch from inputs_token via
    model.get_input_embeddings(), with the action-embedding additions for every
    frame boundary < start_frame replayed explicitly. This is the "fresh
    continuation" path: it takes NO hidden state, only (inputs_token, action,
    start_frame) -- proving frames are conditioned purely on the reconstructed
    tensor prefix, not on any Python object carried over from a prior call.

    If inputs_embeds is provided (already-correct running embeddings from a
    live loop), it is used as-is and just extended -- this is the "normal, no
    intervention" path used to build the initial prefix.

    Returns (inputs_token, inputs_embeds) after generating through
    end_frame_exclusive - 1.
    """
    device = inputs_token.device
    action_embeds = model.action_linear(action)

    if inputs_embeds is None:
        inputs_embeds = model.get_input_embeddings(inputs_token)
        for i in range(start_frame):
            pos = sdf_pos(model, i)
            inputs_embeds[:, pos, :] += action_embeds[:, i + model.context - 1, :]
        expected_len = model.prelude_tokens_num + 1 + start_frame * (model.tokens_num_per_dyna + 1)
        assert inputs_token.shape[1] == expected_len, (
            f"inputs_token length {inputs_token.shape[1]} does not match expected prefix "
            f"length {expected_len} for start_frame={start_frame}")

    B = inputs_token.shape[0]
    for i in range(start_frame, end_frame_exclusive):
        pos = sdf_pos(model, i)
        assert pos == inputs_embeds.shape[1] - 1, (
            f"action injection position {pos} does not point at the last embedding "
            f"slot ({inputs_embeds.shape[1] - 1}) -- prefix/embeds are out of sync")
        inputs_embeds[:, pos, :] += action_embeds[:, i + model.context - 1, :]

        predicted_token = model.llm.generate(
            inputs_embeds=inputs_embeds,
            do_sample=do_sample,
            temperature=temperature,
            top_k=top_k,
            pad_token_id=pad_token_id,
            use_cache=True,
            max_new_tokens=model.tokens_num_per_dyna,
            return_dict_in_generate=False,
        )
        predicted_token = torch.cat(
            [predicted_token, (torch.ones(B) * model.token_for_sdf).unsqueeze(1).to(device)], dim=1
        ).to(predicted_token.dtype)

        inputs_embeds = torch.cat([inputs_embeds, model.get_input_embeddings(predicted_token)], dim=1)
        inputs_token = torch.cat([inputs_token, predicted_token], dim=1)

    return inputs_token, inputs_embeds


@torch.no_grad()
def decode_single_frame_pixels(tokenizer, inputs_token, model, frame_idx):
    """Decode just frame_idx (0-indexed among predicted frames) to pixel space,
    using tokenizer.detokenize() with future_length=1 (context tokens + that
    frame's sdf+dyna tokens only). Returns pixel tensor (B, 3, H, W) in [0,1].
    """
    context_tokens = inputs_token[:, :model.prelude_tokens_num]
    start, end = dyna_slice(model, frame_idx)
    frame_tokens = inputs_token[:, sdf_pos(model, frame_idx):end]  # sdf + 16 dyna = 17 tokens
    indices_for_decode = torch.cat([context_tokens, frame_tokens], dim=1)
    frames = tokenizer.detokenize(indices_for_decode, tokenizer.context_length)
    return frames[:, -1]  # (B, 3, H, W), the single future frame


@torch.no_grad()
def retokenize_single_frame(tokenizer, context_pixel_values, modified_frame_pixels, context_length):
    """Re-tokenize a single (possibly modified) future frame conditioned on the
    given context frames, exactly as tokenizer.tokenize() would if it were part
    of a longer clip (see module docstring: dyna tokens only depend on context,
    not on other future frames). Returns the 16 dyna tokens (B, 16).
    """
    clip = torch.cat([context_pixel_values, modified_frame_pixels.unsqueeze(1)], dim=1)
    tokens, _ = tokenizer.tokenize(clip, context_length)
    prelude = context_length * (256 + 1) - 1
    dyna_tokens = tokens[:, prelude + 1: prelude + 1 + 16]
    return dyna_tokens


@torch.no_grad()
def splice_frame(model, tokenizer, inputs_token, context_pixel_values, frame_idx, modify_fn):
    """Replace frame_idx's 16 dyna tokens in inputs_token with tokens obtained by:
      1. decoding frame_idx to pixels,
      2. applying modify_fn(pixels) -> modified pixels,
      3. re-tokenizing the modified frame against the same context.

    Returns a NEW inputs_token tensor (original is not mutated in place).
    """
    orig_pixels = decode_single_frame_pixels(tokenizer, inputs_token, model, frame_idx)
    modified_pixels = modify_fn(orig_pixels).clamp(0.0, 1.0)
    new_dyna_tokens = retokenize_single_frame(
        tokenizer, context_pixel_values, modified_pixels, tokenizer.context_length
    ).to(inputs_token.dtype)

    start, end = dyna_slice(model, frame_idx)
    new_inputs_token = inputs_token.clone()
    new_inputs_token[:, start:end] = new_dyna_tokens
    return new_inputs_token, orig_pixels, modified_pixels
