from __future__ import annotations

import argparse
import logging
import sys
import time
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from shared.ablator import Ablator, is_lm_head
from shared.metrics import safe_logit_kl, safe_top1_disagreement
from shared.models import classify_layer, list_lm_linear_layers, load_model
from shared.prompts import SMOKE_PROMPTS

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(message)s",
    datefmt="%H:%M:%S",
)
log = logging.getLogger("smoke_test")


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--model", default="pythia-1.4b")
    p.add_argument("--bits", type=int, default=4)
    p.add_argument("--device", default="cuda")
    p.add_argument("--target-role", default="mlp_up",
                   choices=["mlp_up", "mlp_down", "attn_qkv", "attn_out"])
    return p.parse_args()


def pick_target_layer(model, role: str) -> str:
    candidates = [
        name for name, _ in list_lm_linear_layers(model)
        if classify_layer(name) == role and not is_lm_head(name)
    ]
    if not candidates:
        raise RuntimeError(f"No layers found for role '{role}'")
    return candidates[len(candidates) // 2]


def main() -> int:
    args = parse_args()
    log.info("Smoke: model=%s bits=%d role=%s", args.model, args.bits, args.target_role)

    model, tokenizer = load_model(args.model, device=args.device)
    ablator = Ablator(model)

    target = pick_target_layer(model, args.target_role)
    log.info("Target: %s", target)

    inputs = tokenizer(SMOKE_PROMPTS, return_tensors="pt", padding=True).to(args.device)
    attention_mask = inputs.get("attention_mask")

    log.info("Baseline forward pass")
    t0 = time.time()
    with torch.inference_mode():
        baseline = model(**inputs).logits.detach().clone()
    log.info("  %.2fs", time.time() - t0)

    log.info("Ablated forward pass (INT%d)", args.bits)
    t0 = time.time()
    with ablator.ablate([target], bits=args.bits):
        with torch.inference_mode():
            ablated = model(**inputs).logits.detach().clone()
    log.info("  %.2fs", time.time() - t0)

    with torch.inference_mode():
        restored = model(**inputs).logits.detach().clone()

    abl_kl, n_valid = safe_logit_kl(baseline, ablated, attention_mask=attention_mask)
    res_kl, _ = safe_logit_kl(baseline, restored, attention_mask=attention_mask)
    abl_top1 = safe_top1_disagreement(baseline, ablated, attention_mask=attention_mask)

    log.info("Ablation KL:    %.6f  (n_valid=%d)", abl_kl, n_valid)
    log.info("Top-1 disagree: %.4f", abl_top1)
    log.info("Restoration KL: %.6e (should be ~0)", res_kl)

    if not (abl_kl == abl_kl):
        log.error("FAIL: ablation KL is NaN, all positions were masked out")
        return 1
    if res_kl > 1e-5:
        log.error("FAIL: weights not restored")
        return 1
    if abl_kl < 1e-6:
        log.error("FAIL: ablation had no effect")
        return 1
    log.info("PASS")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
