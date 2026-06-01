from __future__ import annotations

import os
os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")

import argparse
import logging
import sys
import time
from pathlib import Path

import torch
import yaml

REPO_ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(REPO_ROOT / "src"))

from shared.preflight import run_preflight
from shared.utils import (
    checkpoint_exists,
    cuda_memory_summary,
    free_cuda_memory,
    progress_bar,
    redirect_logging_to_tqdm,
    safe_model_slug,
    save_json,
    setup_logging,
    with_retry_once,
)


log = logging.getLogger("overnight")


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--config", required=True, type=Path)
    p.add_argument("--resume", action="store_true",
                   help="Skip stages whose checkpoint files already exist")
    p.add_argument("--dry-run", action="store_true",
                   help="Print what would run without doing anything")
    p.add_argument("--smoke-test", action="store_true",
                   help="Full smoke-test (~5min/model) before the real run. "
                        "Recommended for first runs on a new machine.")
    p.add_argument("--skip-preflight", action="store_true",
                   help="Skip pre-flight checks (NOT RECOMMENDED for overnight)")
    return p.parse_args()


def run_one_model(model_short_name: str, model_dir: Path, cfg: dict, args) -> None:
    from shared.models import load_model
    from src.experiments.exp01_layer_sensitivity.full_sweep import run_sweep
    from experiments.exp02_feature_regression.extract_features import run_extraction
    from experiments.exp03_capability_damage.protected_quant import protected_quantization
    from experiments.exp03_capability_damage.calibration import evaluate_calibration
    from experiments.exp03_capability_damage.refusal import evaluate_refusal
    from experiments.exp03_capability_damage.needle_in_haystack import evaluate_needle

    @with_retry_once(f"load:{model_short_name}")
    def _load():
        return load_model(
            model_short_name,
            dtype=getattr(torch, cfg.get("dtype", "float16")),
            device=cfg.get("device", "cuda"),
        )
    loaded = _load()
    if loaded is None:
        log.error("Could not load %s — skipping all stages for this model", model_short_name)
        return
    model, tokenizer = loaded
    del loaded
    log.info("After load: %s", cuda_memory_summary())

    try:
        precisions = cfg.get("precisions", [4, 3, 2])
        sweep_csvs: dict[int, Path] = {}
        for bits in precisions:
            out_path = model_dir / f"exp01_{safe_model_slug(model_short_name)}_int{bits}.csv"
            if args.resume and checkpoint_exists(out_path):
                log.info("Skipping exp01 int%d (checkpoint exists)", bits)
                sweep_csvs[bits] = out_path
                continue

            @with_retry_once(f"exp01:{model_short_name}:int{bits}")
            def _exp1():
                return run_sweep(
                    model, tokenizer,
                    bits=bits,
                    device=cfg.get("device", "cuda"),
                    out_path=out_path,
                )
            result = _exp1()
            if result is not None:
                sweep_csvs[bits] = out_path

        feat_path = model_dir / f"exp02_features_{safe_model_slug(model_short_name)}.csv"
        if args.resume and checkpoint_exists(feat_path):
            log.info("Skipping exp02 features (checkpoint exists)")
        else:
            @with_retry_once(f"exp02a:{model_short_name}")
            def _exp2a():
                return run_extraction(model, out_path=feat_path)
            _exp2a()

        int4_csv = sweep_csvs.get(4)
        if int4_csv is None or not int4_csv.exists():
            log.warning("Skipping Exp 3 for %s: no INT4 ablation results available",
                        model_short_name)
            return

        exp3_dir = model_dir / "exp03"
        exp3_dir.mkdir(parents=True, exist_ok=True)
        device = cfg.get("device", "cuda")

        probes_fp16 = [
            ("calibration_fp16", lambda: evaluate_calibration(
                model, tokenizer, device=device,
                n_per_subject=cfg.get("calibration_n_per_subject", 80),
                output_path=exp3_dir / "calibration_fp16.json",
            )),
            ("refusal_fp16", lambda: evaluate_refusal(
                model, tokenizer, device=device,
                max_new_tokens=cfg.get("refusal_max_new_tokens", 100),
                output_path=exp3_dir / "refusal_fp16.json",
            )),
            ("needle_fp16", lambda: evaluate_needle(
                model, tokenizer, device=device,
                context_lengths=cfg.get("needle_context_lengths", [1024, 2048, 4096]),
                depths=cfg.get("needle_depths", [0.1, 0.25, 0.5, 0.75, 0.9]),
                n_per_cell=cfg.get("needle_n_per_cell", 2),
                output_path=exp3_dir / "needle_fp16.json",
            )),
        ]
        for probe_name, runner in probes_fp16:
            ckpt = exp3_dir / f"{probe_name}.json"
            if args.resume and checkpoint_exists(ckpt):
                log.info("Skipping %s (checkpoint exists)", probe_name)
                continue
            _wrap = with_retry_once(f"exp3:{model_short_name}:{probe_name}")(runner)
            _wrap()

        log.info("Building protected-quant variant for %s", model_short_name)
        for probe_name, base_runner in [
            ("calibration_protected_int4", lambda: evaluate_calibration(
                model, tokenizer, device=device,
                n_per_subject=cfg.get("calibration_n_per_subject", 80),
                output_path=exp3_dir / "calibration_protected_int4.json",
            )),
            ("refusal_protected_int4", lambda: evaluate_refusal(
                model, tokenizer, device=device,
                max_new_tokens=cfg.get("refusal_max_new_tokens", 100),
                output_path=exp3_dir / "refusal_protected_int4.json",
            )),
            ("needle_protected_int4", lambda: evaluate_needle(
                model, tokenizer, device=device,
                context_lengths=cfg.get("needle_context_lengths", [1024, 2048, 4096]),
                depths=cfg.get("needle_depths", [0.1, 0.25, 0.5, 0.75, 0.9]),
                n_per_cell=cfg.get("needle_n_per_cell", 2),
                output_path=exp3_dir / "needle_protected_int4.json",
            )),
        ]:
            ckpt = exp3_dir / f"{probe_name}.json"
            if args.resume and checkpoint_exists(ckpt):
                log.info("Skipping %s (checkpoint exists)", probe_name)
                continue

            def _quant_probe(runner=base_runner):
                with protected_quantization(
                    model, int4_csv,
                    bits=4,
                    protect_top_pct=cfg.get("protect_top_pct", 0.10),
                ):
                    return runner()

            _wrap = with_retry_once(f"exp3:{model_short_name}:{probe_name}")(_quant_probe)
            _wrap()

    finally:
        log.info("Freeing model %s", model_short_name)
        del model, tokenizer
        free_cuda_memory()
        log.info("After free: %s", cuda_memory_summary())


def run_cross_model(results_root: Path, cfg: dict, args) -> None:
    from experiments.exp02_feature_regression.cross_model_regression import run_regression

    for bits in cfg.get("precisions", [4, 3, 2]):
        output_path = results_root / f"exp02_regression_int{bits}.json"
        if args.resume and checkpoint_exists(output_path):
            log.info("Skipping cross-model regression int%d (checkpoint exists)", bits)
            continue

        @with_retry_once(f"exp2b:int{bits}")
        def _exp2b():
            return run_regression(results_root, bits, output_path)
        _exp2b()


def main() -> int:
    args = parse_args()

    with open(args.config) as f:
        cfg = yaml.safe_load(f)

    results_root = Path(cfg["results_root"])
    results_root.mkdir(parents=True, exist_ok=True)
    log_path = results_root / f"run_{time.strftime('%Y%m%d_%H%M')}.log"
    setup_logging(log_path)
    log.info("Overnight pipeline run started. Config: %s | Results: %s | Resume=%s | Dry run=%s | Smoke test=%s",
             args.config, results_root, args.resume, args.dry_run, args.smoke_test)

    model_names = [m["name"] for m in cfg["models"]]

    if not args.skip_preflight and not args.dry_run:
        log.info("Running pre-flight checks...")
        pf = run_preflight(
            model_names=model_names,
            results_root=results_root,
            smoke_test=args.smoke_test,
            min_disk_gb=cfg.get("min_disk_gb", 5.0),
        )
        log.info("\n" + pf.report())
        save_json({
            "cuda_ok": pf.cuda_ok,
            "vram_total_gb": pf.vram_total_gb,
            "vram_free_gb": pf.vram_free_gb,
            "disk_free_gb": pf.disk_free_gb,
            "mmlu_subjects_ok": pf.mmlu_subjects_ok,
            "mmlu_subjects_failed": pf.mmlu_subjects_failed,
            "models_ok": pf.models_ok,
            "models_failed": pf.models_failed,
            "smoke_test_passed": pf.smoke_test_passed,
            "fatal_issues": pf.fatal_issues,
        }, results_root / "preflight.json")

        if pf.fatal_issues:
            log.error("Pre-flight reported fatal issues. Aborting.")
            log.error("Fix issues above and re-run, or pass --skip-preflight to override.")
            return 1
        if args.smoke_test:
            failed = [n for n, ok in pf.smoke_test_passed.items() if not ok]
            if failed:
                log.error("Smoke test failed for: %s. Aborting.", failed)
                return 1
        log.info("Pre-flight complete.")

    if not torch.cuda.is_available():
        log.warning("CUDA not available — this run will be extremely slow")
    else:
        log.info("CUDA device: %s", torch.cuda.get_device_name(0))
        log.info("Initial %s", cuda_memory_summary())

    if args.dry_run:
        log.info("Dry run mode: would run %d models", len(model_names))
        return 0

    overall_t0 = time.time()

    model_pbar = progress_bar(cfg["models"], desc="models", total=len(cfg["models"]))
    with redirect_logging_to_tqdm():
        for model_cfg in model_pbar:
            model_name = model_cfg["name"]
            model_dir = results_root / safe_model_slug(model_name)
            model_dir.mkdir(parents=True, exist_ok=True)

            model_pbar.set_postfix(model=model_name)

            log.info("Model: %s | Output: %s", model_name, model_dir)

            run_one_model(model_name, model_dir, cfg, args)
            free_cuda_memory()
            log.info("Between-model %s", cuda_memory_summary())

    log.info("Stage: Cross-model regression (Exp 2b)")
    run_cross_model(results_root, cfg, args)

    elapsed = time.time() - overall_t0
    log.info("Run complete | total elapsed: %.1f hours (%.0f minutes)",
             elapsed / 3600, elapsed / 60)

    save_json(
        {
            "completed_at": time.strftime("%Y-%m-%d %H:%M:%S"),
            "elapsed_hours": elapsed / 3600,
            "models": model_names,
            "log_path": str(log_path),
        },
        results_root / "run_summary.json",
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
