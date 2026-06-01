from __future__ import annotations

import logging
import time
from dataclasses import dataclass, field
from pathlib import Path

import torch

log = logging.getLogger(__name__)


@dataclass
class PreflightResult:
    cuda_ok: bool = False
    vram_total_gb: float = 0.0
    vram_free_gb: float = 0.0
    disk_free_gb: float = 0.0
    disk_ok: bool = False
    mmlu_ok: bool = False
    mmlu_subjects_ok: list[str] = field(default_factory=list)
    mmlu_subjects_failed: list[str] = field(default_factory=list)
    models_ok: list[str] = field(default_factory=list)
    models_failed: dict[str, str] = field(default_factory=dict)
    smoke_test_passed: dict[str, bool] = field(default_factory=dict)

    @property
    def fatal_issues(self) -> list[str]:
        issues = []
        if not self.cuda_ok:
            issues.append("CUDA unavailable")
        if not self.disk_ok:
            issues.append(f"Disk space too low: {self.disk_free_gb:.1f} GB free")
        if self.models_failed:
            issues.append(f"Models failed to load: {list(self.models_failed.keys())}")
        if not self.mmlu_ok:
            issues.append("MMLU not loadable — Exp 3 calibration will fail")
        return issues

    def report(self) -> str:
        lines = []
        lines.append("Pre-flight report")
        lines.append(f"CUDA available: {self.cuda_ok}")
        lines.append(f"VRAM total: {self.vram_total_gb:.1f} GB | free: {self.vram_free_gb:.1f} GB")
        lines.append(f"Disk free at results_root: {self.disk_free_gb:.1f} GB")
        lines.append(f"MMLU loadable: {self.mmlu_ok}")
        lines.append(f"  Subjects OK: {self.mmlu_subjects_ok}")
        lines.append(f"  Subjects failed: {self.mmlu_subjects_failed}")
        lines.append(f"Models OK: {self.models_ok}")
        if self.models_failed:
            lines.append("Models FAILED:")
            for name, err in self.models_failed.items():
                lines.append(f"  {name}: {err}")
        if self.smoke_test_passed:
            lines.append("Smoke test results:")
            for name, passed in self.smoke_test_passed.items():
                lines.append(f"  {name}: {'PASS' if passed else 'FAIL'}")
        if self.fatal_issues:
            lines.append("Fatal issues — run will be aborted:")
            for iss in self.fatal_issues:
                lines.append(f"  - {iss}")
        else:
            lines.append("All checks passed.")
        return "\n".join(lines)


DEFAULT_MMLU_SUBJECTS = [
    "high_school_us_history",
    "high_school_biology",
    "elementary_mathematics",
    "computer_security",
    "professional_law",
]


def run_preflight(
    model_names: list[str],
    results_root: Path,
    mmlu_subjects: list[str] | None = None,
    smoke_test: bool = False,
    min_disk_gb: float = 5.0,
) -> PreflightResult:
    from shared.utils import disk_free_gb

    result = PreflightResult()

    if torch.cuda.is_available():
        result.cuda_ok = True
        free, total = torch.cuda.mem_get_info()
        result.vram_total_gb = total / 1024**3
        result.vram_free_gb = free / 1024**3
        log.info("CUDA OK: %s | %.1f/%.1f GB free",
                 torch.cuda.get_device_name(0),
                 result.vram_free_gb, result.vram_total_gb)
    else:
        log.error("CUDA not available")

    results_root = Path(results_root)
    results_root.mkdir(parents=True, exist_ok=True)
    result.disk_free_gb = disk_free_gb(results_root)
    result.disk_ok = result.disk_free_gb >= min_disk_gb
    log.info("Disk: %.1f GB free at %s (need %.1f GB)",
             result.disk_free_gb, results_root, min_disk_gb)

    if mmlu_subjects is None:
        mmlu_subjects = DEFAULT_MMLU_SUBJECTS
    try:
        from datasets import load_dataset
        for subj in mmlu_subjects:
            try:
                ds = load_dataset("cais/mmlu", subj, split="test")
                _ = ds[0]
                result.mmlu_subjects_ok.append(subj)
            except Exception as e:
                log.warning("MMLU subject %s failed: %s", subj, e)
                result.mmlu_subjects_failed.append(subj)
        result.mmlu_ok = len(result.mmlu_subjects_ok) > 0
    except ImportError:
        log.error("`datasets` package not installed — Exp 3 calibration will fail")
        result.mmlu_ok = False

    from transformers import AutoTokenizer
    from shared.models import DEFAULT_MODELS
    for name in model_names:
        resolved = DEFAULT_MODELS.get(name, name)
        try:
            tok = AutoTokenizer.from_pretrained(resolved, trust_remote_code=True)
            _ = tok("hello", return_tensors="pt")
            result.models_ok.append(name)
            log.info("Model name OK: %s -> %s (vocab size %d)", name, resolved, tok.vocab_size)
        except Exception as e:
            err_msg = f"{type(e).__name__}: {str(e)[:200]}"
            result.models_failed[name] = err_msg
            log.error("Model name FAILED: %s -> %s — %s", name, resolved, err_msg)

    if smoke_test and result.cuda_ok and not result.models_failed:
        result.smoke_test_passed = _run_full_smoke_test(
            [n for n in model_names if n in result.models_ok]
        )

    return result


def _run_full_smoke_test(model_names: list[str]) -> dict[str, bool]:
    from shared.utils import free_cuda_memory, progress_bar, redirect_logging_to_tqdm

    results: dict[str, bool] = {}
    pbar = progress_bar(model_names, desc="smoke test", total=len(model_names))
    with redirect_logging_to_tqdm():
        for name in pbar:
            log.info("Smoke test: %s", name)
            try:
                passed = _smoke_test_one_model(name)
                results[name] = passed
            except Exception as e:
                log.error("  smoke test crashed: %s", e)
                results[name] = False
            free_cuda_memory()
            pbar.set_postfix(
                passed=sum(results.values()),
                failed=sum(1 for v in results.values() if not v),
            )
            log.info("Memory after %s: %s",
                     "PASS" if results.get(name) else "FAIL",
                     _cuda_summary())

    return results


def _smoke_test_one_model(name: str) -> bool:
    from shared.ablator import Ablator
    from shared.metrics import safe_logit_kl
    from shared.models import list_lm_linear_layers, load_model

    t0 = time.time()
    try:
        model, tokenizer = load_model(name)
    except Exception as e:
        log.error("  load failed: %s", e)
        return False

    layers = list_lm_linear_layers(model)
    log.info("  loaded in %.1fs (%d LM linear layers)", time.time() - t0, len(layers))

    inp = tokenizer("The capital of France is", return_tensors="pt").to(model.device)
    with torch.inference_mode():
        base = model(**inp).logits.detach().clone()

    ablator = Ablator(model)
    target_name = layers[len(layers) // 2][0]
    with ablator.ablate([target_name], bits=4):
        with torch.inference_mode():
            abl = model(**inp).logits.detach()
        kl, n_valid = safe_logit_kl(base, abl, attention_mask=inp.get("attention_mask"))

    if not (kl == kl):
        log.error("  KL is NaN for %s — safe_logit_kl masking didn't catch it", target_name)
        return False
    if kl <= 0 or n_valid == 0:
        log.error("  KL=%s n_valid=%d unexpected", kl, n_valid)
        return False
    log.info("  ablation OK: KL=%.5f on %s", kl, target_name)

    with torch.inference_mode():
        gen = model.generate(
            **inp, max_new_tokens=10, do_sample=False,
            pad_token_id=tokenizer.pad_token_id or tokenizer.eos_token_id,
        )
    resp = tokenizer.decode(gen[0], skip_special_tokens=True)
    log.info("  generation OK: %r", resp[:80])
    log.info("  PASSED in %.1fs", time.time() - t0)
    return True


def _cuda_summary() -> str:
    if not torch.cuda.is_available():
        return "(no CUDA)"
    free, total = torch.cuda.mem_get_info()
    used_gb = (total - free) / 1024**3
    total_gb = total / 1024**3
    return f"VRAM used: {used_gb:.1f}/{total_gb:.1f} GB"
