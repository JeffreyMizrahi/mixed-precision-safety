from __future__ import annotations

import argparse
import json
import logging
import os
import sys
import time
from pathlib import Path

os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")

REPO_ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(REPO_ROOT / "src"))

import torch

from shared.models import load_model
from shared.quant_variants import VARIANTS, quantized_variant
from shared.utils import free_cuda_memory, save_json, setup_logging
from experiments.exp04_real_benchmarks.tasks import resolve

log = logging.getLogger("exp04.lm_eval")


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--model", required=True,
                   help="Short name (e.g. llama-3.2-3b) or HF id")
    p.add_argument("--variant", required=True,
                   choices=list(VARIANTS) + ["all"],
                   help="Quant variant. 'all' loops over fp16, int4_naive, int4_protected.")
    p.add_argument("--sensitivity-csv", type=Path, default=None,
                   help="Exp 1 INT4 CSV. Required for int4_protected.")
    p.add_argument("--bits", type=int, default=4)
    p.add_argument("--protect-top-pct", type=float, default=0.10)
    p.add_argument("--suites", nargs="+", default=["standard"],
                   help="Suite names from tasks.SUITES, or raw lm-eval task ids.")
    p.add_argument("--limit", type=int, default=None,
                   help="Items per subtask (smoke testing). None = full.")
    p.add_argument("--batch-size", default="auto",
                   help="'auto' or an integer.")
    p.add_argument("--num-fewshot", type=int, default=None,
                   help="Override few-shot count. Default uses task default.")
    p.add_argument("--device", default="cuda")
    p.add_argument("--dtype", default="float16",
                   help="Model load dtype. Use bfloat16 for Qwen3 — float16 "
                        "overflows on some inputs and yields all-NaN logits.")
    p.add_argument("--add-bos-token", choices=["auto", "true", "false"], default="auto",
                   help="Prepend BOS for loglikelihood scoring. 'auto' = True for "
                        "Gemma (required; without it Gemma scores at chance on "
                        "multiple-choice tasks), False otherwise.")
    p.add_argument("--output", type=Path, default=None,
                   help="Single-variant output JSON path.")
    p.add_argument("--output-dir", type=Path, default=None,
                   help="With --variant all, write {variant}.json into this dir.")
    p.add_argument("--log-file", type=Path, default=None)
    p.add_argument("--cache-dir", type=Path, default=None,
                   help="lm-eval request cache dir (speeds up reruns).")
    p.add_argument("--skip-existing", action="store_true",
                   help="Skip any variant whose output JSON already exists.")
    return p.parse_args()


def _evaluate(lm, tasks, *, limit, num_fewshot, cache_dir):
    from lm_eval import simple_evaluate

    kw = dict(
        model=lm,
        tasks=tasks,
        limit=limit,
        log_samples=False,
    )
    if num_fewshot is not None:
        kw["num_fewshot"] = num_fewshot
    if cache_dir is not None:
        kw["use_cache"] = str(cache_dir)
    results = simple_evaluate(**kw)
    if "config" in results and "model" in results["config"]:
        results["config"]["model"] = type(lm).__name__
    return results


def run_variant(
    model_short: str,
    model, tokenizer,
    variant: str,
    *,
    suites: list[str],
    sensitivity_csv: Path | None,
    bits: int,
    protect_top_pct: float,
    limit: int | None,
    batch_size,
    num_fewshot: int | None,
    output_path: Path,
    cache_dir: Path | None,
    add_bos_token: bool = False,
) -> dict:
    from lm_eval.models.huggingface import HFLM

    tasks = resolve(suites)
    log.info("Variant=%s | tasks=%s | limit=%s", variant, tasks, limit)

    t0 = time.time()
    with quantized_variant(
        model, variant,
        sensitivity_csv=sensitivity_csv,
        bits=bits,
        protect_top_pct=protect_top_pct,
    ):
        bs = batch_size if batch_size == "auto" else int(batch_size)
        lm = HFLM(pretrained=model, tokenizer=tokenizer, batch_size=bs,
                  add_bos_token=add_bos_token)
        raw = _evaluate(lm, tasks,
                        limit=limit, num_fewshot=num_fewshot,
                        cache_dir=cache_dir)

    elapsed = time.time() - t0

    summary = {
        "model": model_short,
        "variant": variant,
        "bits": bits,
        "protect_top_pct": protect_top_pct if variant == "int4_protected" else None,
        "sensitivity_csv": str(sensitivity_csv) if sensitivity_csv else None,
        "suites": suites,
        "tasks": tasks,
        "limit": limit,
        "elapsed_s": elapsed,
        "results": raw.get("results", {}),
        "group_subtasks": raw.get("group_subtasks", {}),
        "versions": raw.get("versions", {}),
        "n_shot": raw.get("n-shot", raw.get("n_shot", {})),
        "config": raw.get("config", {}),
    }
    output_path.parent.mkdir(parents=True, exist_ok=True)
    save_json(summary, output_path)
    log.info("Wrote %s (%.1f min)", output_path, elapsed / 60)
    return summary


def main() -> int:
    args = parse_args()
    setup_logging(args.log_file)

    if args.variant == "int4_protected" and args.sensitivity_csv is None:
        log.error("--sensitivity-csv is required for int4_protected")
        return 2
    if args.variant == "all" and args.sensitivity_csv is None:
        log.error("--variant all needs --sensitivity-csv (for int4_protected pass)")
        return 2

    if args.variant == "all":
        if args.output_dir is None:
            log.error("--variant all requires --output-dir")
            return 2
        args.output_dir.mkdir(parents=True, exist_ok=True)
        variants_to_run = list(VARIANTS)
        outputs = {v: args.output_dir / f"lm_eval_{v}.json" for v in variants_to_run}
    else:
        if args.output is None:
            log.error("Single variant needs --output")
            return 2
        variants_to_run = [args.variant]
        outputs = {args.variant: args.output}

    log.info("Loading model %s", args.model)
    model, tokenizer = load_model(
        args.model,
        dtype=getattr(torch, args.dtype),
        device=args.device,
    )

    if args.add_bos_token == "auto":
        add_bos = "gemma" in args.model.lower()
    else:
        add_bos = args.add_bos_token == "true"
    log.info("add_bos_token=%s for model %s", add_bos, args.model)

    try:
        for v in variants_to_run:
            if args.skip_existing and outputs[v].exists():
                log.info("Skipping %s — %s already exists", v, outputs[v])
                continue
            try:
                run_variant(
                    args.model, model, tokenizer, v,
                    suites=args.suites,
                    sensitivity_csv=args.sensitivity_csv,
                    bits=args.bits,
                    protect_top_pct=args.protect_top_pct,
                    limit=args.limit,
                    batch_size=args.batch_size,
                    num_fewshot=args.num_fewshot,
                    output_path=outputs[v],
                    cache_dir=args.cache_dir,
                    add_bos_token=add_bos,
                )
            except Exception:
                log.exception("Variant %s failed; continuing", v)
                outputs[v].parent.mkdir(parents=True, exist_ok=True)
                with open(outputs[v].with_suffix(".error.json"), "w") as f:
                    import traceback
                    json.dump({"variant": v, "traceback": traceback.format_exc()}, f, indent=2)
    finally:
        del model, tokenizer
        free_cuda_memory()

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
