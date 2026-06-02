# Experiment 01: Layer-wise sensitivity mapping

This experiment measures how much each individual Linear layer matters under
quantization. It quantizes one LM Linear layer at a time to int-N while holding every
other layer at fp16, then measures how far the model's output distribution moves. The
result is a per-layer sensitivity score that the rest of the pipeline uses to decide which
layers are worth protecting.

## What it does

For a fixed precision (int4, int3, or int2), the sweep runs a fixed 16-prompt battery
through the fp16 model once to get a baseline set of logits. Then, for each LM Linear
layer in turn, it fake-quantizes just that layer, runs the same prompts, and compares the
new logits against the baseline. Two quantities are recorded per layer:

- `kl_div`: the mean KL divergence between the baseline and ablated logit distributions,
  computed by `safe_logit_kl` in `src/shared/metrics.py`, which masks out any token
  position where either distribution went non-finite.
- `top1_disagree`: the fraction of token positions where the top-1 predicted token changed.

Quantization is per-output-channel symmetric fake-quant from `src/shared/ablator.py`. The
`lm_head` is excluded by default, and for multimodal models the vision and audio tower
layers are filtered out so only the text backbone is touched. After each layer is
measured, the original weights are restored, so layers are always tested in isolation.

The sweep is one forward pass per layer, which is fast even for the larger models (the 8B
models have a few hundred LM Linear layers).

## How to run it

A smoke test first, to confirm the ablator and the metric work and that weights restore
cleanly:

```bash
cd src
python -m experiments.exp01_layer_sensitivity.smoke_test --model llama-3.2-3b
```

Then a full sweep for one model and precision:

```bash
python -m experiments.exp01_layer_sensitivity.full_sweep \
    --model llama-3.2-3b --bits 4 \
    --output ../combined/exp01_meta-llama_Llama-3.2-3B_int4.csv
```

Use `--dtype bfloat16` for Qwen3 and Gemma, since fp16 produces all-NaN logits on those.
Useful flags: `--bits {2,3,4}`, `--limit N` to ablate only the first N layers for a quick
check, `--include-lm-head` to also sweep the output head, and `--resume` to skip if the
output CSV already exists.

To produce sweeps for all models at once, the orchestrators at the repo root
(`overnight_run.py` and `run_all.sh`) call this module for each model and precision.

## Output

One CSV per model per precision, written to the path given by `--output` (the canonical
copies live in `combined/`). Columns: `layer_idx_in_block`, `layer_name`, `role`,
`out_features`, `in_features`, `n_params`, `kl_div`, `top1_disagree`, `n_valid_positions`,
`elapsed_s`. One row per layer. This CSV is the direct input to the protected-quant variant
in exp03 and exp04, and to the feature regression in exp02.

In the committed results the KL distribution is dominated by the `mlp_down` layers, with
the earliest and latest blocks the most sensitive, which is the expected
quantization-sensitivity pattern.