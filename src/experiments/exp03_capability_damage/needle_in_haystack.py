from __future__ import annotations

import argparse
import logging
import random
import sys
import time
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from shared.models import load_model
from shared.utils import progress_bar, redirect_logging_to_tqdm, save_json

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(message)s",
    datefmt="%H:%M:%S",
)
log = logging.getLogger("needle")


FILLER_PARAGRAPH = (
    "The migration patterns of monarch butterflies have fascinated biologists "
    "for over a century. Each autumn, millions of these insects travel from "
    "their breeding grounds in southern Canada and the northeastern United "
    "States to overwintering sites in central Mexico. The journey covers "
    "approximately three thousand miles and requires multiple generations to "
    "complete in some cases. Scientists have studied how the butterflies "
    "navigate using a combination of solar cues, magnetic fields, and "
    "inherited information passed down through generations. Recent research "
    "has revealed surprising details about the molecular mechanisms underlying "
    "their circadian rhythms and sun-compass orientation. Despite extensive "
    "study, many aspects of the monarch migration remain mysterious. "
)


NEEDLE_TEMPLATE = "The secret access code for the Anthropic vault is {code}."
QUERY = "What is the secret access code for the Anthropic vault?"


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--model", default="pythia-1.4b")
    p.add_argument("--device", default="cuda")
    p.add_argument("--n-per-cell", type=int, default=2)
    p.add_argument("--output", required=True)
    return p.parse_args()


def _make_haystack(needle: str, target_tokens: int, depth_fraction: float, tokenizer) -> str:
    rng = random.Random(0)
    sentences = [s for s in FILLER_PARAGRAPH.split(". ") if s]

    filler_chunks: list[str] = []
    current_tokens = 0
    while current_tokens < target_tokens * 2:
        chunk = ". ".join(rng.sample(sentences, len(sentences))) + ". "
        filler_chunks.append(chunk)
        toks = tokenizer(chunk, add_special_tokens=False)["input_ids"]
        current_tokens += len(toks)

    full_filler = "".join(filler_chunks)
    filler_tokens = tokenizer(full_filler, add_special_tokens=False)["input_ids"]
    filler_tokens = filler_tokens[:target_tokens]
    full_filler = tokenizer.decode(filler_tokens, skip_special_tokens=True)

    insert_idx = int(len(full_filler) * depth_fraction)
    nearest_period = full_filler.rfind(". ", 0, insert_idx)
    if nearest_period == -1:
        nearest_period = insert_idx
    else:
        nearest_period += 2
    return full_filler[:nearest_period] + needle + " " + full_filler[nearest_period:]


def evaluate_needle(
    model,
    tokenizer,
    *,
    device: str = "cuda",
    context_lengths: list[int] | None = None,
    depths: list[float] | None = None,
    n_per_cell: int = 2,
    output_path: Path | None = None,
) -> dict:
    if context_lengths is None:
        context_lengths = [1024, 2048, 4096]
    if depths is None:
        depths = [0.1, 0.25, 0.5, 0.75, 0.9]

    n_total = len(context_lengths) * len(depths) * n_per_cell
    log.info("Needle eval: %d ctx × %d depths × %d trials = %d total",
             len(context_lengths), len(depths), n_per_cell, n_total)

    rng = random.Random(42)
    trial_records = []
    t0 = time.time()

    cells = [(ctx_len, depth) for ctx_len in context_lengths for depth in depths]
    pbar = progress_bar(cells, desc="needle cells", total=len(cells))
    with redirect_logging_to_tqdm():
        for ctx_len, depth in pbar:
            successes = 0
            for trial in range(n_per_cell):
                code = f"{rng.randint(1000, 9999)}-{rng.choice(['ALPHA', 'BETA', 'GAMMA', 'DELTA'])}"
                needle = NEEDLE_TEMPLATE.format(code=code)
                haystack = _make_haystack(needle, ctx_len, depth, tokenizer)
                prompt = haystack + "\n\nQuestion: " + QUERY + "\nAnswer:"

                inputs = tokenizer(
                    prompt, return_tensors="pt", truncation=True, max_length=ctx_len + 256,
                ).to(device)
                with torch.inference_mode():
                    out_ids = model.generate(
                        **inputs,
                        max_new_tokens=30,
                        do_sample=False,
                        pad_token_id=tokenizer.pad_token_id or tokenizer.eos_token_id,
                    )
                new_tokens = out_ids[0, inputs.input_ids.shape[1]:]
                response = tokenizer.decode(new_tokens, skip_special_tokens=True)
                success = code in response
                if success:
                    successes += 1
                trial_records.append({
                    "ctx_len": ctx_len,
                    "depth": depth,
                    "trial": trial,
                    "code": code,
                    "response_first_60": response[:60],
                    "success": success,
                })
            pbar.set_postfix(ctx=ctx_len, depth=f"{depth:.2f}", rate=f"{successes}/{n_per_cell}")
            log.info("  ctx=%d depth=%.2f: %d/%d", ctx_len, depth, successes, n_per_cell)

    by_cell: dict = {}
    for r in trial_records:
        key = (r["ctx_len"], r["depth"])
        by_cell.setdefault(key, []).append(r["success"])
    aggregated = [
        {
            "ctx_len": k[0],
            "depth": k[1],
            "n_trials": len(v),
            "success_rate": sum(v) / len(v),
        }
        for k, v in by_cell.items()
    ]

    overall = sum(r["success"] for r in trial_records) / len(trial_records)
    summary = {
        "overall_success_rate": overall,
        "n_trials": len(trial_records),
        "by_cell": aggregated,
        "context_lengths": context_lengths,
        "depths": depths,
        "elapsed_s": time.time() - t0,
    }
    if output_path is not None:
        save_json(summary, output_path)
        log.info("Wrote needle result to %s (overall=%.2f)", output_path, overall)
    return summary


def main() -> int:
    args = parse_args()
    model, tokenizer = load_model(args.model, device=args.device)
    evaluate_needle(
        model, tokenizer,
        device=args.device,
        n_per_cell=args.n_per_cell,
        output_path=Path(args.output),
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
