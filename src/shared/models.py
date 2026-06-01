from __future__ import annotations

import logging

import torch
from torch import nn
from transformers import AutoModelForCausalLM, AutoTokenizer

log = logging.getLogger(__name__)


DEFAULT_MODELS = {
    "gemma-4-4b": "google/gemma-4-E4B-it",
    "llama-3.2-3b": "meta-llama/Llama-3.2-3B-Instruct",
    "qwen3-8b": "Qwen/Qwen3-8B",
    "llama-3.1-8b": "meta-llama/Llama-3.1-8B-Instruct",
    "llama-3.1-405b": "meta-llama/Llama-3.1-405B-Instruct",
    "llama-4-maverick": "meta-llama/Llama-4-Maverick-17B-128E-Instruct",
    "pythia-1.4b": "EleutherAI/pythia-1.4b",
    "qwen3-1.7b": "Qwen/Qwen3-1.7B",
    "qwen3-4b": "Qwen/Qwen3-4B",
    "qwen3-14b": "Qwen/Qwen3-14B",
}

MOE_MODELS = {"llama-4-maverick"}
PRIMARY_LINEUP = ["gemma-4-4b", "llama-3.2-3b", "qwen3-8b", "llama-3.1-8b"]


def load_model(model_id: str, *, dtype: torch.dtype = torch.float16, device: str = "cuda"):
    if model_id in DEFAULT_MODELS:
        model_id = DEFAULT_MODELS[model_id]

    log.info("Loading tokenizer: %s", model_id)
    tokenizer = AutoTokenizer.from_pretrained(model_id, trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "left"

    log.info("Loading model: %s (dtype=%s, device=%s)", model_id, dtype, device)
    model = AutoModelForCausalLM.from_pretrained(
        model_id,
        torch_dtype=dtype,
        device_map=device,
        trust_remote_code=True,
        low_cpu_mem_usage=True,
    )
    model.eval()
    model.config.use_cache = False

    n_params = sum(p.numel() for p in model.parameters())
    log.info("Loaded %s (%.2fB params)", model_id, n_params / 1e9)
    return model, tokenizer


LAYER_ROLE_PATTERNS = {
    "attn_qkv": ["query_key_value", "q_proj", "k_proj", "v_proj"],
    "attn_out": ["attention.dense", "o_proj", "self_attn.dense"],
    "mlp_up": ["dense_h_to_4h", "up_proj", "gate_proj", "fc1"],
    "mlp_down": ["dense_4h_to_h", "down_proj", "fc2"],
}


def classify_layer(name: str) -> str:
    for role, patterns in LAYER_ROLE_PATTERNS.items():
        if any(p in name for p in patterns):
            return role
    return "other"


def extract_layer_index(name: str) -> int | None:
    for part in name.split("."):
        if part.isdigit():
            return int(part)
    return None


_NON_LM_LAYER_PATTERNS = (
    "vision_tower",
    "vision_model",
    "vision_encoder",
    "visual.",
    "audio_tower",
    "audio_model",
    "audio_encoder",
    "speech_encoder",
    "multi_modal_projector",
    "mm_projector",
    "image_newline",
    "patch_embedder",
    "image_encoder",
)


def is_vision_layer(name: str) -> bool:
    return any(pat in name for pat in _NON_LM_LAYER_PATTERNS)


def list_lm_linear_layers(model: nn.Module) -> list[tuple[str, nn.Linear]]:
    from shared.ablator import list_linear_layers
    return [(n, m) for n, m in list_linear_layers(model) if not is_vision_layer(n)]


def is_vlm(model: nn.Module) -> bool:
    from shared.ablator import list_linear_layers
    return any(is_vision_layer(n) for n, _ in list_linear_layers(model))
