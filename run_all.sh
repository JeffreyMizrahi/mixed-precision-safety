#!/bin/bash
set -u
export HF_HOME=/ephemeral/hf_cache
source /home/ubuntu/mps/.venv/bin/activate
cd /home/ubuntu/mps/src

LLAMA_3_2_CSV=results/exp01_meta-llama_Llama-3.2-3B_int4.csv
LLAMA_3_1_CSV=results/exp01_meta-llama_Llama-3.1-8B_int4.csv
QWEN_CSV=results/exp01_Qwen_Qwen3-8B_int4.csv
GEMMA_CSV=results/exp01_google_gemma-4-E4B_int4.csv

run_lm_eval() {
    local model="$1" csv="$2" dtype="$3" outdir="$4"
    mkdir -p "$outdir"
    echo "Running $model exp04 in $dtype..."
    python -m experiments.exp04_real_benchmarks.run_lm_eval \
        --model "$model" --variant all --dtype "$dtype" \
        --sensitivity-csv "$csv" \
        --suites standard safety \
        --output-dir "$outdir" --skip-existing
    echo "$model exp04 done (exit=$?)."
}

run_exp03() {
    local model="$1" csv="$2" dtype="$3" outdir="$4"
    mkdir -p "$outdir"
    if [ -f "$outdir/calibration_fp16.json" ] && [ -f "$outdir/refusal_fp16.json" ] && [ -f "$outdir/needle_fp16.json" ] \
       && [ -f "$outdir/calibration_protected_int4.json" ] && [ -f "$outdir/refusal_protected_int4.json" ] && [ -f "$outdir/needle_protected_int4.json" ]; then
        echo "$model exp03 already complete. Skipping."
        return 0
    fi
    echo "Running $model exp03 capability probes in $dtype..."
    EXP03_MODEL="$model" EXP03_CSV="$csv" EXP03_DTYPE="$dtype" EXP03_OUT="$outdir" python - <<'PY'
import os
os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")
import sys; sys.path.insert(0, os.getcwd())
from pathlib import Path
import torch
from shared.models import load_model
from shared.utils import setup_logging
from experiments.exp03_capability_damage.protected_quant import protected_quantization
from experiments.exp03_capability_damage.calibration import evaluate_calibration
from experiments.exp03_capability_damage.refusal import evaluate_refusal
from experiments.exp03_capability_damage.needle_in_haystack import evaluate_needle

setup_logging(None)
MODEL = os.environ["EXP03_MODEL"]
CSV = os.environ["EXP03_CSV"]
DTYPE = getattr(torch, os.environ["EXP03_DTYPE"])
OUT = Path(os.environ["EXP03_OUT"]); OUT.mkdir(parents=True, exist_ok=True)
CTX, DEPTHS = [1024, 2048, 4096], [0.1, 0.25, 0.5, 0.75, 0.9]

model, tok = load_model(MODEL, dtype=DTYPE, device="cuda")

evaluate_calibration(model, tok, device="cuda", n_per_subject=80, output_path=OUT/"calibration_fp16.json")
evaluate_refusal(model, tok, device="cuda", max_new_tokens=100, output_path=OUT/"refusal_fp16.json")
evaluate_needle(model, tok, device="cuda", context_lengths=CTX, depths=DEPTHS, n_per_cell=2, output_path=OUT/"needle_fp16.json")

with protected_quantization(model, CSV, bits=4, protect_top_pct=0.10):
    evaluate_calibration(model, tok, device="cuda", n_per_subject=80, output_path=OUT/"calibration_protected_int4.json")
    evaluate_refusal(model, tok, device="cuda", max_new_tokens=100, output_path=OUT/"refusal_protected_int4.json")
    evaluate_needle(model, tok, device="cuda", context_lengths=CTX, depths=DEPTHS, n_per_cell=2, output_path=OUT/"needle_protected_int4.json")

print("Probes complete.")
PY
    echo "$model exp03 done (exit=$?)."
}

run_lm_eval llama-3.2-3b "$LLAMA_3_2_CSV" float16 results/llama-3_2-3b/exp04
run_exp03   llama-3.2-3b "$LLAMA_3_2_CSV" float16 results/llama-3_2-3b/exp03

run_lm_eval llama-3.1-8b "$LLAMA_3_1_CSV" float16 results/llama-3_1-8b/exp04
run_exp03   llama-3.1-8b "$LLAMA_3_1_CSV" float16 results/llama-3_1-8b/exp03

QWEN_FINITE=$(python -c "import pandas as pd; print(int(pd.read_csv('$QWEN_CSV').kl_div.notna().sum()))" 2>/dev/null || echo 0)
if [ "$QWEN_FINITE" -eq 0 ]; then
    echo "Qwen sensitivity CSV missing or all-NaN (fp16 NaN bug). Regenerating exp01 int4 in bf16..."
    python -m experiments.exp01_layer_sensitivity.full_sweep \
        --model qwen3-8b --bits 4 --dtype bfloat16 --output "$QWEN_CSV"
fi
run_lm_eval qwen3-8b "$QWEN_CSV" bfloat16 results/qwen3-8b/exp04
run_exp03   qwen3-8b "$QWEN_CSV" bfloat16 results/qwen3-8b/exp03

GEMMA_OUT=results/gemma-4-4b/exp04
mkdir -p "$GEMMA_OUT"
echo "Running gemma fp16 baseline in bf16 (gate test for the dtype hypothesis)..."
python -m experiments.exp04_real_benchmarks.run_lm_eval \
    --model gemma-4-4b --variant fp16 --dtype bfloat16 \
    --suites standard safety \
    --output "$GEMMA_OUT/lm_eval_fp16.json" --skip-existing

GEMMA_GATE=$(python - <<'PY'
import json
try:
    r = json.load(open("results/gemma-4-4b/exp04/lm_eval_fp16.json"))["results"]["arc_challenge"]
    v = r.get("acc_norm,none") or r.get("acc,none") or 0.0
    print(f"{v:.4f} {1 if v > 0.40 else 0}")
except Exception as e:
    print(f"NA 0  # {e}")
PY
)
echo "Gemma fp16 arc_challenge: $GEMMA_GATE (chance is 0.25, gate threshold is 0.40)"

if [ "$(echo "$GEMMA_GATE" | awk '{print $2}')" = "1" ]; then
    echo "bf16 recovered gemma. Regenerating exp01 in bf16 so the protected layer-ranking isn't built on a corrupted fp16 baseline..."
    python -m experiments.exp01_layer_sensitivity.full_sweep \
        --model gemma-4-4b --bits 4 --dtype bfloat16 --output "$GEMMA_CSV"
    GEMMA_FINITE=$(python -c "import pandas as pd; print(int(pd.read_csv('$GEMMA_CSV').kl_div.notna().sum()))")
    echo "Gemma exp01 finite kl_div rows: $GEMMA_FINITE"
    if [ "$GEMMA_FINITE" -gt 0 ]; then
        echo "Running gemma exp04 naive and protected in bf16..."
        python -m experiments.exp04_real_benchmarks.run_lm_eval \
            --model gemma-4-4b --variant all --dtype bfloat16 \
            --sensitivity-csv "$GEMMA_CSV" \
            --suites standard safety \
            --output-dir "$GEMMA_OUT" --skip-existing
        echo "Gemma exp04 done."
    else
        echo "Gemma exp01 came back all-NaN even in bf16. Skipping naive and protected."
    fi
else
    echo "bf16 did not lift gemma off chance. Dtype is not the cause."
    echo "Gemma is google/gemma-4-E4B-it (multimodal Gemma-3n / MatFormer). The fix is somewhere in the loglikelihood path, not the dtype. Skipping exp01 regen, naive, and protected."
fi

run_exp03 gemma-4-4b "$GEMMA_CSV" float16 results/gemma-4-4b/exp03

echo "Regenerating exp04 summary..."
python -m experiments.exp04_real_benchmarks.analyze --results-root results --csv results/exp04_summary.csv

echo "All runs done."
