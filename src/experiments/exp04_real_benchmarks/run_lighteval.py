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

from shared.models import load_model, DEFAULT_MODELS
from shared.quant_variants import VARIANTS, quantized_variant
from shared.utils import free_cuda_memory, save_json, setup_logging

log = logging.getLogger("exp04.lighteval")


DEFAULT_LIGHTEVAL_TASKS = ",".join([
    "leaderboard|mmlu|0|0",
    "leaderboard|hellaswag|0|0",
    "leaderboard|arc:challenge|0|0",
    "leaderboard|winogrande|0|0",
    "leaderboard|gsm8k|0|0",
    "leaderboard|truthfulqa:mc|0|0",
])


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--model", required=True)
    p.add_argument("--variant", required=True,
                   choices=list(VARIANTS) + ["all"])
    p.add_argument("--sensitivity-csv", type=Path, default=None)
    p.add_argument("--bits", type=int, default=4)
    p.add_argument("--protect-top-pct", type=float, default=0.10)
    p.add_argument("--tasks", default=DEFAULT_LIGHTEVAL_TASKS,
                   help="Comma-separated lighteval task specs.")
    p.add_argument("--limit", type=int, default=None)
    p.add_argument("--batch-size", type=int, default=4)
    p.add_argument("--device", default="cuda")
    p.add_argument("--dtype", default="float16")
    p.add_argument("--output", type=Path, default=None)
    p.add_argument("--output-dir", type=Path, default=None)
    p.add_argument("--log-file", type=Path, default=None)
    p.add_argument("--skip-existing", action="store_true",
                   help="Skip any variant whose output JSON already exists.")
    return p.parse_args()


def _build_pipeline(model, tokenizer, *, hf_id: str, batch_size: int,
                    tasks: str, limit: int | None):
    from lighteval.pipeline import Pipeline, PipelineParameters, ParallelismManager
    from lighteval.models.transformers.transformers_model import (
        TransformersModel, TransformersModelConfig,
    )

    model_config = TransformersModelConfig(
        pretrained=hf_id,
        dtype="float16",
        batch_size=batch_size,
    )
    lm = TransformersModel(config=model_config)
    lm.model = model
    lm.tokenizer = tokenizer
    lm._tokenizer = tokenizer

    params = PipelineParameters(
        launcher_type=ParallelismManager.NONE,
        max_samples=limit,
    )
    pipe = Pipeline(
        tasks=tasks,
        pipeline_parameters=params,
        model=lm,
    )
    return pipe


def run_variant(
    model_short: str,
    hf_id: str,
    model, tokenizer,
    variant: str,
    *,
    tasks: str,
    sensitivity_csv: Path | None,
    bits: int,
    protect_top_pct: float,
    limit: int | None,
    batch_size: int,
    output_path: Path,
) -> dict:
    log.info("Variant=%s | tasks=%s | limit=%s", variant, tasks, limit)
    t0 = time.time()

    with quantized_variant(
        model, variant,
        sensitivity_csv=sensitivity_csv,
        bits=bits,
        protect_top_pct=protect_top_pct,
    ):
        pipe = _build_pipeline(
            model, tokenizer,
            hf_id=hf_id,
            batch_size=batch_size,
            tasks=tasks,
            limit=limit,
        )
        pipe.evaluate()
        results = pipe.get_results()

    elapsed = time.time() - t0

    def _safe(o):
        try:
            json.dumps(o)
            return o
        except TypeError:
            return str(o)

    clean = json.loads(json.dumps(results, default=_safe))

    summary = {
        "model": model_short,
        "hf_id": hf_id,
        "variant": variant,
        "bits": bits,
        "protect_top_pct": protect_top_pct if variant == "int4_protected" else None,
        "sensitivity_csv": str(sensitivity_csv) if sensitivity_csv else None,
        "tasks": tasks,
        "limit": limit,
        "elapsed_s": elapsed,
        "results": clean,
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

    if args.variant == "all":
        if args.output_dir is None:
            log.error("--variant all needs --output-dir")
            return 2
        args.output_dir.mkdir(parents=True, exist_ok=True)
        variants_to_run = list(VARIANTS)
        outputs = {v: args.output_dir / f"lighteval_{v}.json" for v in variants_to_run}
    else:
        if args.output is None:
            log.error("Single variant needs --output")
            return 2
        variants_to_run = [args.variant]
        outputs = {args.variant: args.output}

    hf_id = DEFAULT_MODELS.get(args.model, args.model)

    log.info("Loading model %s (%s)", args.model, hf_id)
    model, tokenizer = load_model(
        args.model,
        dtype=getattr(torch, args.dtype),
        device=args.device,
    )

    try:
        for v in variants_to_run:
            if args.skip_existing and outputs[v].exists():
                log.info("Skipping %s — %s already exists", v, outputs[v])
                continue
            try:
                run_variant(
                    args.model, hf_id, model, tokenizer, v,
                    tasks=args.tasks,
                    sensitivity_csv=args.sensitivity_csv,
                    bits=args.bits,
                    protect_top_pct=args.protect_top_pct,
                    limit=args.limit,
                    batch_size=args.batch_size,
                    output_path=outputs[v],
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
