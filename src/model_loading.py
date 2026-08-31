"""
Loads the pretrained iVideoGPT checkpoint (tokenizer + action-conditioned
transformer) and picks a compute device. This is the shared entry point every
experiment/analysis script in this repository uses to obtain a ready-to-run
model; it does not run any experiment itself.

The checkpoint-loading and generation code paths here call directly into the
unmodified upstream iVideoGPT implementation (third_party/iVideoGPT); see
validation/validate_checkpoint_loading.py for the standalone sanity check that
this loading path and the model's generate()/detokenize() produce finite,
well-shaped output before any perturbation experiment is run.
"""
import os
import sys

import torch
from safetensors.torch import load_file
from transformers import AutoModelForCausalLM, AutoConfig

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(REPO_ROOT, "third_party", "iVideoGPT"))

from ivideogpt.vq_model import CompressiveVQModel
from ivideogpt.transformer import HeadModelWithAction

CKPT = os.path.join(REPO_ROOT, "checkpoints", "ivideogpt-vp2-robosuite-64-act-cond")

CONTEXT_LENGTH = 2
SEGMENT_LENGTH = 12
RESOLUTION = 64
ACTION_DIM = 4


def pick_device():
    if torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def load_models(device):
    tokenizer = CompressiveVQModel.from_pretrained(
        CKPT, subfolder="tokenizer", low_cpu_mem_usage=False).to(device)
    assert tokenizer.context_length == CONTEXT_LENGTH

    config = AutoConfig.from_pretrained(CKPT, subfolder="transformer")
    llm = AutoModelForCausalLM.from_config(config)
    prelude_tokens_num = (256 + 1) * CONTEXT_LENGTH - 1
    tokens_num_per_dyna = 16
    model = HeadModelWithAction(
        llm, action_dim=ACTION_DIM,
        prelude_tokens_num=prelude_tokens_num,
        tokens_num_per_dyna=tokens_num_per_dyna,
        context=CONTEXT_LENGTH,
        segment_length=SEGMENT_LENGTH,
    ).to(device)
    state_dict = load_file(os.path.join(CKPT, "transformer", "model.safetensors"))
    model.load_state_dict(state_dict, strict=True)
    model.eval()

    assert model.llm.config.vocab_size == tokenizer.num_vq_embeddings + tokenizer.num_dyn_embeddings + 2
    return tokenizer, model
