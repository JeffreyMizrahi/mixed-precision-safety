from __future__ import annotations

import argparse
import logging
import sys
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from datasets import load_dataset

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from shared.metrics import expected_calibration_error
from shared.models import load_model
from shared.utils import progress_bar, redirect_logging_to_tqdm, save_json

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(message)s",
    datefmt="%H:%M:%S",
)
log = logging.getLogger("calibration")


_CHOICES = ["A", "B", "C", "D"]
DEFAULT_MMLU_SUBJECTS = [
    "high_school_us_history",
    "high_school_biology",
    "elementary_mathematics",
    "computer_security",
    "professional_law",
]


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--model", default="pythia-1.4b")
    p.add_argument("--device", default="cuda")
    p.add_argument("--n-per-subject", type=int, default=80)
    p.add_argument("--output", required=True)
    return p.parse_args()


def _build_prompt(item: dict) -> str:
    q = item["question"].strip()
    a, b, c, d = item["choices"]
    return (
        f"Question: {q}\n"
        f"A. {a}\n"
        f"B. {b}\n"
        f"C. {c}\n"
        f"D. {d}\n"
        f"Answer:"
    )


def _letter_token_ids(tokenizer) -> list[int]:
    ids: list[int] = []
    issues = []
    for c in _CHOICES:
        spaced = tokenizer(" " + c, add_special_tokens=False)["input_ids"]
        plain = tokenizer(c, add_special_tokens=False)["input_ids"]
        if len(spaced) == 1:
            ids.append(spaced[0])
        elif len(plain) == 1:
            ids.append(plain[0])
        else:
            ids.append(spaced[-1])
            issues.append(f"{c!r}: ' {c}'={spaced}, '{c}'={plain}")
    if issues:
        log.warning("Letter tokenization issues (calibration may be approximate): %s",
                    "; ".join(issues))
    if len(set(ids)) < 4:
        log.error("Letter token IDs collide: %s. Calibration numbers will be wrong.", ids)
    return ids


def _parse_mmlu_answer(answer) -> int:
    if isinstance(answer, int):
        return answer
    if isinstance(answer, str):
        a = answer.strip().upper()
        if a in _CHOICES:
            return _CHOICES.index(a)
        try:
            return int(a)
        except ValueError:
            raise ValueError(f"Unparseable MMLU answer: {answer!r}")
    raise TypeError(f"MMLU answer has unexpected type {type(answer)}: {answer!r}")


def evaluate_calibration(
    model,
    tokenizer,
    *,
    device: str = "cuda",
    n_per_subject: int = 80,
    subjects: list[str] | None = None,
    output_path: Path | None = None,
) -> dict:
    if subjects is None:
        subjects = DEFAULT_MMLU_SUBJECTS
    log.info("Loading MMLU subjects: %s", subjects)
    items: list[dict] = []
    for subj in subjects:
        try:
            ds = load_dataset("cais/mmlu", subj, split="test")
            for x in ds.shuffle(seed=0).select(range(min(n_per_subject, len(ds)))):
                items.append(x)
        except Exception as e:
            log.warning("Could not load MMLU subject %s: %s", subj, e)

    if not items:
        result = {"error": "no MMLU data available"}
        if output_path is not None:
            save_json(result, output_path)
        return result

    log.info("Calibration eval on %d items", len(items))
    letter_ids = _letter_token_ids(tokenizer)
    log.info("Letter token IDs: A=%d B=%d C=%d D=%d", *letter_ids)

    confidences = []
    correct_flags = []
    t0 = time.time()

    pbar = progress_bar(items, desc="calibration", total=len(items))
    with redirect_logging_to_tqdm():
        for i, item in enumerate(pbar):
            prompt = _build_prompt(item)
            inputs = tokenizer(prompt, return_tensors="pt", truncation=True, max_length=1024).to(device)
            with torch.inference_mode():
                out = model(**inputs)
            last_logits = out.logits[0, -1, :].float()
            letter_logits = last_logits[letter_ids]
            probs = F.softmax(letter_logits, dim=-1)
            pred_idx = int(probs.argmax().item())
            conf = float(probs.max().item())
            try:
                gold_idx = _parse_mmlu_answer(item["answer"])
            except (ValueError, TypeError) as e:
                log.warning("Skipping item with bad answer: %s", e)
                continue
            confidences.append(conf)
            correct_flags.append(int(pred_idx == gold_idx))
            running_acc = sum(correct_flags) / len(correct_flags)
            pbar.set_postfix(acc=f"{running_acc:.3f}", conf=f"{conf:.2f}")
            if (i + 1) % 100 == 0:
                log.info("  calibration progress: %d/%d (acc=%.3f)", i + 1, len(items), running_acc)

    confidences_t = torch.tensor(confidences)
    correct_t = torch.tensor(correct_flags)

    accuracy = float(correct_t.float().mean())
    ece = expected_calibration_error(confidences_t, correct_t, n_bins=15)
    mean_conf = float(confidences_t.mean())

    result = {
        "n_items": len(confidences),
        "accuracy": accuracy,
        "mean_confidence": mean_conf,
        "ece": ece,
        "overconfidence": mean_conf - accuracy,
        "subjects": subjects,
        "n_per_subject_requested": n_per_subject,
        "elapsed_s": time.time() - t0,
    }
    if output_path is not None:
        save_json(result, output_path)
        log.info("Wrote calibration result to %s", output_path)
    return result


def main() -> int:
    args = parse_args()
    model, tokenizer = load_model(args.model, device=args.device)
    evaluate_calibration(
        model, tokenizer,
        device=args.device,
        n_per_subject=args.n_per_subject,
        output_path=Path(args.output),
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
