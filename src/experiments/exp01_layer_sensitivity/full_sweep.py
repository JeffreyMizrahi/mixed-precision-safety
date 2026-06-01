from __future__ import annotations

import argparse
import csv
import logging
import sys
import time
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from shared.ablator import Ablator, is_lm_head
from shared.metrics import safe_logit_kl, safe_top1_disagreement
from shared.models import classify_layer, extract_layer_index, list_lm_linear_layers, load_model
from shared.prompts import SWEEP_PROMPTS
from shared.utils import progress_bar, redirect_logging_to_tqdm

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(message)s",
    datefmt="%H:%M:%S",
)
log = logging.getLogger("full_sweep")


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--model", default="pythia-1.4b")
    p.add_argument("--bits", type=int, default=4)
    p.add_argument("--device", default="cuda")
    p.add_argument("--dtype", default="float16",
                   help="Model load dtype. Use bfloat16 for Qwen3 to avoid float16 overflow.")
    p.add_argument("--output", default=None)
    p.add_argument("--include-lm-head", action="store_true")
    p.add_argument("--limit", type=int, default=None)
    p.add_argument("--resume", action="store_true")
    return p.parse_args()


def default_output_path(model_name: str, bits: int) -> Path:
    repo_root = Path(__file__).resolve().parents[2]
    safe = model_name.replace("/", "_")
    return repo_root / "src" / "results" / f"exp01_{safe}_int{bits}.csv"


def run_sweep(
    model,
    tokenizer,
    *,
    bits: int,
    device: str,
    out_path: Path,
    include_lm_head: bool = False,
    limit: int | None = None,
) -> list[dict]:
    ablator = Ablator(model)

    targets = list_lm_linear_layers(model)
    if not include_lm_head:
        targets = [(n, m) for n, m in targets if not is_lm_head(n)]
    if limit:
        targets = targets[: limit]
    log.info("Ablating %d Linear layers", len(targets))

    inputs = tokenizer(SWEEP_PROMPTS, return_tensors="pt", padding=True).to(device)
    attention_mask = inputs.get("attention_mask")

    log.info("Computing FP16 baseline")
    t0 = time.time()
    with torch.inference_mode():
        baseline = model(**inputs).logits.detach().clone()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    log.info("  %.2fs", time.time() - t0)

    rows = []
    sweep_start = time.time()
    nan_count = 0

    pbar = progress_bar(targets, desc=f"sweep int{bits}", total=len(targets))
    with redirect_logging_to_tqdm():
        for i, (name, module) in enumerate(pbar):
            layer_t0 = time.time()
            try:
                with ablator.ablate([name], bits=bits):
                    with torch.inference_mode():
                        ablated = model(**inputs).logits.detach()
                    kl, n_valid = safe_logit_kl(baseline, ablated, attention_mask=attention_mask)
                    top1 = safe_top1_disagreement(baseline, ablated, attention_mask=attention_mask)
                    del ablated
            except Exception as e:
                log.warning("Layer %s failed: %s", name, e)
                kl, top1, n_valid = float("nan"), float("nan"), 0

            if not (kl == kl):
                nan_count += 1

            elapsed = time.time() - layer_t0
            rows.append({
                "layer_idx_in_block": extract_layer_index(name),
                "layer_name": name,
                "role": classify_layer(name),
                "out_features": module.out_features,
                "in_features": module.in_features,
                "n_params": module.weight.numel(),
                "kl_div": kl,
                "top1_disagree": top1,
                "n_valid_positions": n_valid,
                "elapsed_s": elapsed,
            })

            pbar.set_postfix(kl=f"{kl:.4f}", top1=f"{top1:.3f}", nans=nan_count)

            if (i + 1) % 25 == 0:
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()
                done = i + 1
                rate = done / (time.time() - sweep_start)
                eta = (len(targets) - done) / rate
                log.info("  [%d/%d] %s  kl=%.4f  top1=%.3f  (%.2fs, eta %.0fs)",
                         done, len(targets), name, kl, top1, elapsed, eta)

    fieldnames = list(rows[0].keys())
    with out_path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)

    log.info("Wrote %d rows to %s (%d NaN rows)", len(rows), out_path, nan_count)
    log.info("Total: %.1fs", time.time() - sweep_start)

    by_role: dict[str, list[float]] = {}
    for r in rows:
        if r["kl_div"] == r["kl_div"]:
            by_role.setdefault(r["role"], []).append(r["kl_div"])
    log.info("KL by role (mean / max):")
    for role, kls in sorted(by_role.items()):
        if kls:
            log.info("  %-10s  mean=%.4f  max=%.4f  (n=%d)",
                     role, sum(kls) / len(kls), max(kls), len(kls))
    return rows


def main() -> int:
    args = parse_args()
    out_path = Path(args.output) if args.output else default_output_path(args.model, args.bits)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    if args.resume and out_path.exists():
        log.info("Output %s already exists, skipping (--resume)", out_path)
        return 0

    log.info("Sweep: model=%s bits=%d -> %s", args.model, args.bits, out_path)

    model, tokenizer = load_model(args.model, device=args.device,
                                  dtype=getattr(torch, args.dtype))
    run_sweep(
        model, tokenizer,
        bits=args.bits,
        device=args.device,
        out_path=out_path,
        include_lm_head=args.include_lm_head,
        limit=args.limit,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
