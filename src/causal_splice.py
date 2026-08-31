"""
Reimplementation of the per-frame generation loop in
ivideogpt.transformer.action_model.HeadModelWithAction.generate() (lines
56-121), generalized to stop after an arbitrary frame t (prefix phase) and
resume from a reconstructed token prefix (continuation phase), rebuilding
embeddings fresh from token ids and actions with no hidden state carried
across the boundary. Does not modify action_model.py; calls the same public
methods (get_input_embeddings, action_linear, llm.generate) the original
loop uses.

Three architecture facts this relies on:

- generate() calls llm.generate() once per frame with use_cache scoped to
  that single call; there is no persistent KV-cache across frames in Python
  (action_model.py:101-113). The causal prefix is therefore fully
  represented by the token/embedding tensors, which is what makes a clean
  prefix splice possible.
- Action conditioning is an in-place additive perturbation of one embedding
  vector per frame (the sdf token before that frame's 16 dyna tokens), not
  extra tokens (action_model.py:80-81, 174-177). Reconstructing
  inputs_embeds from token ids alone is therefore insufficient after the
  first frame; the action additions for every prior frame must be replayed.
- Per compressive_vq_model.py's tokenize()/detokenize(), each frame's 16
  dyna tokens depend only on the fixed context frames, not on other future
  frames (context features are broadcast across future_length,
  compressive_vq_model.py:174-191). A single future frame can therefore be
  tokenized in isolation and yields the same tokens as in a longer clip,
  which is what makes decode-modify-retokenize valid.

Token layout (context_length=2: prelude_tokens_num=513, tokens_num_per_dyna=16):
  [0, 513)   context tokens (2 frames x 257, minus one dropped leading scf token)
  513        sdf token for frame 0 (action for frame 0 adds onto this slot)
  [514, 530) dyna tokens for frame 0
  530        sdf token for frame 1
  [531, 547) dyna tokens for frame 1
  ... (period 17 = tokens_num_per_dyna + 1)

  sdf_pos(i)    = prelude_tokens_num + i * (tokens_num_per_dyna + 1)
  dyna_start(i) = sdf_pos(i) + 1
  dyna_end(i)   = dyna_start(i) + tokens_num_per_dyna
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
    """Generate frames [start_frame, end_frame_exclusive), appending to inputs_token.

    If inputs_embeds is None, it is rebuilt from inputs_token via
    get_input_embeddings(), replaying the action-embedding additions for
    every frame boundary before start_frame (the fresh-continuation path:
    conditioned only on (inputs_token, action, start_frame), no carried
    Python state). If inputs_embeds is provided, it is extended as-is (the
    no-intervention path used to build the initial prefix).
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
            f"slot ({inputs_embeds.shape[1] - 1}), prefix/embeds are out of sync")
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
    """Decode frame_idx to pixel space via detokenize() with future_length=1
    (context tokens plus that frame's sdf+dyna tokens only). Returns (B,3,H,W) in [0,1].
    """
    context_tokens = inputs_token[:, :model.prelude_tokens_num]
    start, end = dyna_slice(model, frame_idx)
    frame_tokens = inputs_token[:, sdf_pos(model, frame_idx):end]  # sdf + 16 dyna = 17 tokens
    indices_for_decode = torch.cat([context_tokens, frame_tokens], dim=1)
    frames = tokenizer.detokenize(indices_for_decode, tokenizer.context_length)
    return frames[:, -1]  # (B, 3, H, W), the single future frame


@torch.no_grad()
def retokenize_single_frame(tokenizer, context_pixel_values, modified_frame_pixels, context_length):
    """Re-tokenize one future frame against the given context, as tokenize()
    would if it were part of a longer clip (see module docstring). Returns
    the 16 dyna tokens as (B,16).
    """
    clip = torch.cat([context_pixel_values, modified_frame_pixels.unsqueeze(1)], dim=1)
    tokens, _ = tokenizer.tokenize(clip, context_length)
    prelude = context_length * (256 + 1) - 1
    dyna_tokens = tokens[:, prelude + 1: prelude + 1 + 16]
    return dyna_tokens


@torch.no_grad()
def splice_frame(model, tokenizer, inputs_token, context_pixel_values, frame_idx, modify_fn):
    """Replace frame_idx's 16 dyna tokens: decode to pixels, apply modify_fn,
    re-tokenize against the same context. Returns a new tensor; inputs_token
    is not mutated.
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
