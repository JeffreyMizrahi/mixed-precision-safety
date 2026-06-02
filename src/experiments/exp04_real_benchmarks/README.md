# Experiment 04: Real benchmarks

This experiment runs the full benchmark sweep that the rest of the project is judged
against. It evaluates each model at three precisions, fp16, int4-naive, and
int4-protected, across standard and safety suites, and aggregates the results into a single
cross-model recovery table. It is what lets us say how much of the naive-int4 accuracy drop
the protected variant recovers, task by task.

## The three variants

For each model the harness runs the same evaluation three times, switching the model's
weights with the `quantized_variant` context manager from `src/shared/quant_variants.py`:

- fp16: no quantization, the baseline.
- int4-naive: every LM Linear layer fake-quantized to int4 (`lm_head` excluded).
- int4-protected: the top 10 percent highest-KL layers from the exp01 sweep held at fp16,
  the rest quantized to int4. This requires the exp01 sensitivity CSV.

## The suites

Defined in `tasks.py`:

- standard: MMLU, HellaSwag, ARC-Challenge, WinoGrande, GSM8K.
- safety: TruthfulQA-MC1, TruthfulQA-MC2, BBQ, ToxiGen.
- reasoning: BBH, Minerva Math, HumanEval (available, not part of the committed runs).
- perplexity: WikiText.

## How to run it

The primary harness is lm-evaluation-harness via `run_lm_eval.py`. Run all three variants
for one model:

```bash
cd src
python -m experiments.exp04_real_benchmarks.run_lm_eval \
    --model llama-3.2-3b --variant all \
    --sensitivity-csv ../combined/exp01_meta-llama_Llama-3.2-3B_int4.csv \
    --suites standard safety \
    --output-dir ../combined/llama-3_2-3b/exp04
```

Notes on the flags. `--variant all` loops over the three precisions and needs
`--sensitivity-csv` for the protected pass. `--skip-existing` skips any variant whose JSON
already exists, which makes the run resumable. `--dtype bfloat16` is required for Qwen3 and
Gemma. `--add-bos-token auto` prepends a BOS token for Gemma, which Gemma scoring is
sensitive to, and leaves it off otherwise. `--limit N` caps items per subtask for smoke
testing.

There is also a secondary lighteval path in `run_lighteval.py` for cross-checking against a
second harness. lm-eval is the primary source of the committed numbers.

## Aggregating

`analyze.py` reads the per-model variant JSONs, picks the primary metric for each task
(for example `acc_norm` for HellaSwag and ARC, `exact_match,strict-match` for GSM8K),
computes the naive and protected deltas from fp16, and computes a recovery percentage. It
prints a per-model table and a cross-model average, and optionally writes the summary CSV:

```bash
python -m experiments.exp04_real_benchmarks.analyze \
    --results-root ../combined --csv ../combined/exp04_summary.csv
```

Recovery percentage is the share of the naive drop that protection claws back: 100 percent
means protection fully restored fp16, 0 percent means no better than naive, and negative
means protected is worse than naive on that task. The recovery is only computed when the
naive drop exceeds half a point, to avoid dividing by near-zero deltas (the ToxiGen
percentages are the exception, and the top-level README explains why those should be read
as absolute point regressions rather than percentages).

## Output

Per model under `combined/<model>/exp04/`: `lm_eval_fp16.json`, `lm_eval_int4_naive.json`,
`lm_eval_int4_protected.json`, and the run log. The aggregated cross-model table is
`combined/exp04_summary.csv`.

## A note on Gemma

The Gemma sweep is gated in `run_all.sh` and only the fp16 baseline is committed. Gemma's
multi-token continuation loglikelihood scoring comes back at chance on this architecture
(ARC, HellaSwag, WinoGrande, TruthfulQA), while its single-token MMLU accuracy and its
generative GSM8K work fine. The top-level README covers this in detail.