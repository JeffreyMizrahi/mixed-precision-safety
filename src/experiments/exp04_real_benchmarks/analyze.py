from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

PRIMARY_METRIC = {
    "mmlu": "acc,none",
    "hellaswag": "acc_norm,none",
    "arc_challenge": "acc_norm,none",
    "winogrande": "acc,none",
    "gsm8k": "exact_match,strict-match",
    "truthfulqa_mc1": "acc,none",
    "truthfulqa_mc2": "acc,none",
    "bbq": "acc,none",
    "toxigen": "acc,none",
}
TASK_ORDER = list(PRIMARY_METRIC.keys())
VARIANTS = ["fp16", "int4_naive", "int4_protected"]
MODELS = ["llama-3_2-3b", "llama-3_1-8b", "gemma-4-4b", "qwen3-8b"]


def load_variant(results_root: Path, model: str, variant: str) -> dict | None:
    p = results_root / model / "exp04" / f"lm_eval_{variant}.json"
    if not p.exists():
        return None
    return json.load(open(p))


def get_score(payload: dict, task: str) -> float | None:
    if payload is None:
        return None
    res = payload.get("results", {}).get(task)
    if not res:
        return None
    return res.get(PRIMARY_METRIC[task])


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--results-root", type=Path, default=Path("src/results"))
    ap.add_argument("--csv", type=Path, default=None, help="Optional CSV dump path")
    args = ap.parse_args()

    rows_for_csv = []

    for model in MODELS:
        payloads = {v: load_variant(args.results_root, model, v) for v in VARIANTS}
        if all(p is None for p in payloads.values()):
            print(f"\n### {model}: NO DATA\n")
            continue

        print(f"\n{model}")
        print(f"{'task':<18}{'FP16':>9}{'INT4-naive':>13}{'INT4-prot':>12}"
              f"{'Δnaive':>10}{'Δprot':>10}{'recover%':>10}")

        for task in TASK_ORDER:
            fp = get_score(payloads["fp16"], task)
            nv = get_score(payloads["int4_naive"], task)
            pr = get_score(payloads["int4_protected"], task)
            if fp is None:
                continue
            d_nv = (nv - fp) if nv is not None else None
            d_pr = (pr - fp) if pr is not None else None
            recover = None
            if d_nv is not None and d_pr is not None and d_nv < -0.005:
                recover = (d_pr - d_nv) / (-d_nv) * 100

            def f(x, pct=False):
                if x is None:
                    return f"{'—':>0}"
                return f"{x*100:+.1f}" if pct else f"{x*100:.1f}"

            print(f"{task:<18}"
                  f"{f(fp):>9}"
                  f"{f(nv):>13}"
                  f"{f(pr):>12}"
                  f"{f(d_nv, True):>10}"
                  f"{f(d_pr, True):>10}"
                  f"{(f'{recover:.0f}%' if recover is not None else '—'):>10}")

            rows_for_csv.append({
                "model": model, "task": task,
                "fp16": fp, "int4_naive": nv, "int4_protected": pr,
                "delta_naive": d_nv, "delta_protected": d_pr,
                "recovery_pct": recover,
            })

    print("\nCross-model average degradation (percentage points vs FP16)")
    print(f"{'task':<18}{'avg Δnaive':>14}{'avg Δprot':>14}{'avg recover%':>14}")
    for task in TASK_ORDER:
        dn = [r["delta_naive"] for r in rows_for_csv if r["task"] == task and r["delta_naive"] is not None]
        dp = [r["delta_protected"] for r in rows_for_csv if r["task"] == task and r["delta_protected"] is not None]
        rc = [r["recovery_pct"] for r in rows_for_csv if r["task"] == task and r["recovery_pct"] is not None]
        if not dn:
            continue
        avg_dn = sum(dn) / len(dn) * 100
        avg_dp = sum(dp) / len(dp) * 100 if dp else None
        avg_rc = sum(rc) / len(rc) if rc else None
        print(f"{task:<18}{avg_dn:>+13.1f} "
              f"{(f'{avg_dp:+.1f}' if avg_dp is not None else '—'):>13} "
              f"{(f'{avg_rc:.0f}%' if avg_rc is not None else '—'):>13}")

    if args.csv:
        with open(args.csv, "w", newline="") as fh:
            w = csv.DictWriter(fh, fieldnames=list(rows_for_csv[0].keys()))
            w.writeheader()
            w.writerows(rows_for_csv)
        print(f"\nWrote {args.csv}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
