# Mixed-Precision Safety

Layer-wise sensitivity attribution for INT-quantized large language models, and the
safety- and capability-specific damage that post-training quantization causes even when
MMLU-class aggregate benchmarks look fine.

The core claim of this repo, supported by the data under [`combined/`](combined/), is
simple. If you hold the small set of most-sensitive layers at full precision and quantize
the rest to int4, aggregate accuracy recovers most of the way back to fp16. But a handful
of specific behaviors (calibration, refusal propensity, ToxiGen classification) shift in
ways that aggregate benchmarks never report. "Protection recovered the benchmark" and
"protection preserved the behavior" turn out to be different statements.

## Contents

1. [Quick start](#quick-start)
2. [Running the full pipeline](#running-the-full-pipeline)
3. [Repository layout](#repository-layout)
4. [Methodology](#methodology)
5. [Models](#models)
6. [Results](#results)
7. [The MoE experiment (scoped, not yet run)](#the-moe-experiment-scoped-not-yet-run)
8. [What went wrong and other limitations](#what-went-wrong-and-other-limitations)
9. [License](#license)

## Quick start

### Install

Any single CUDA GPU works. The 3B and 4B models fit in about 16 GB. The 8B models need
24 GB at fp16, or you can load them in bf16. This was developed on RTX 3090 and A6000
class cards with CUDA and Python 3.10.

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
huggingface-cli login          # the Llama and Gemma repos are gated
```

Every experiment module is run from the `src/` directory as a package. The `-m` form
matters, because it puts `src/` on the path so the `shared.*` imports resolve.

```bash
cd src
```

### Smoke test (about a minute)

This confirms the ablator, the masked-KL metric, and a forward-plus-generate round trip
all work on the default model before you commit to a multi-hour sweep.

```bash
python -m experiments.exp01_layer_sensitivity.smoke_test --model llama-3.2-3b
```

It quantizes one mid-stack layer to int4, checks that the KL is finite and non-zero, then
verifies the original weights are restored (restoration KL near zero). A `PASS` means the
ablation machinery is sound.

### One real sensitivity sweep (about 30 minutes on a 3B model)

```bash
python -m experiments.exp01_layer_sensitivity.full_sweep \
    --model llama-3.2-3b --bits 4 \
    --output ../combined/exp01_meta-llama_Llama-3.2-3B_int4.csv
```

This ablates every LM Linear layer to int4 one at a time (196 layers for the 3B) and
writes a CSV with one row per layer: `layer_name`, `role`, `kl_div`, `top1_disagree`, and
the shape and parameter columns. That CSV is the input the protected-quant variant needs
in order to know which layers to keep at full precision.

## Running the full pipeline

There are two entry points by design. One drives the cross-model benchmark story, and one
runs the original per-model sweep.

| script | what it does | when to use |
|---|---|---|
| [`run_all.sh`](run_all.sh) | exp04 (lm-eval) plus exp03 (capability probes) for all four models, with per-model dtypes and a gate for Gemma | the headline cross-model results |
| [`overnight_run.py`](overnight_run.py) | the original exp01 then exp02 then exp03 sequence per model, driven by [`configs/overnight.yaml`](configs/overnight.yaml) | regenerating the sensitivity sweeps and per-layer features |

`run_all.sh` is idempotent. Every `run_lm_eval` call passes `--skip-existing`, and the
exp03 stage checks for all six probe JSONs before re-running a model. You can kill it and
restart it without losing completed work.

### Locally

```bash
# one model, all three variants (fp16, int4-naive, int4-protected), standard plus safety suites
cd src
python -m experiments.exp04_real_benchmarks.run_lm_eval \
    --model llama-3.2-3b --variant all \
    --sensitivity-csv ../combined/exp01_meta-llama_Llama-3.2-3B_int4.csv \
    --suites standard safety \
    --output-dir ../combined/llama-3_2-3b/exp04
# roughly 2 to 3 hours per variant on an A6000
```

`--variant all` loops over `fp16`, `int4_naive`, and `int4_protected`. The protected pass
reads the sensitivity CSV, keeps the top 10 percent highest-KL layers at fp16, and
quantizes the rest. `--suites standard safety` resolves to MMLU, HellaSwag, ARC-Challenge,
WinoGrande, and GSM8K (standard), plus TruthfulQA-MC1/MC2, BBQ, and ToxiGen (safety).

### On Thundercompute

```bash
export HF_TOKEN=hf_xxxxxxxxxxxxxxxxxxxxxx
bash deploy_to_thunder.sh 0            # 0 is your instance id; this rsyncs the project up
tnr connect 0
bash /home/ubuntu/mps/remote_setup.sh  # venv, deps, HF auth, GPU check, then launches run_all.sh detached
```

When `remote_setup.sh` prints "Safe to disconnect," exit with `Ctrl-D` and monitor from
anywhere.

```bash
tail -f /home/ubuntu/mps/run_all.log
tnr scp 0:/home/ubuntu/mps/src/results/ ./src/   # pull results when done
```

### Regenerate the aggregated table and charts

After any fresh run, rebuild the cross-model summary.

```bash
cd src && python -m experiments.exp04_real_benchmarks.analyze \
    --results-root ../combined --csv ../combined/exp04_summary.csv
```

Then open [`charts.ipynb`](charts.ipynb) and run all cells. It reads only from `combined/`
and renders inline on GitHub, covering the recovery bars, the exp03 probe panels, and the
exp01 per-role sensitivity heatmaps.

## Repository layout

```
run_all.sh              Full pipeline: exp04 plus exp03, all four models, gated for Gemma
overnight_run.py        Legacy orchestrator: exp01 then exp02 then exp03 per model
configs/overnight.yaml  Model list and probe sizes for overnight_run.py
deploy_to_thunder.sh    Local: rsync project to a Thundercompute instance
remote_setup.sh         On instance: venv, deps, HF auth, then launches run_all.sh detached
charts.ipynb            Result charts (renders inline on GitHub)

combined/               Canonical merged results tree
  exp01_*.csv             Top-level sensitivity sweeps (per model, int2/int3/int4)
  exp04_summary.csv       Aggregated cross-model recovery table
  <model>/
    exp01_*.csv             Per-model sensitivity sweep copies
    exp02_features_*.csv    Per-layer weight features (input to the cross-model regression)
    exp03/                  6 JSONs: calibration, refusal, needle, each at fp16 and protected_int4
    exp04/                  lm-eval results (fp16, int4_naive, int4_protected) plus the log

src/
  shared/                 ablator, models, metrics, prompts, features, preflight, utils
  experiments/
    exp01_layer_sensitivity/   Per-layer INT-N ablation sweep
    exp01_5_moe_sensitivity/   Planning notes for the MoE variant (scoped, not yet run)
    exp02_feature_regression/  Predict sensitivity from weight statistics
    exp03_capability_damage/   Calibration, refusal, and needle probes, plus the protected_quant context
    exp04_real_benchmarks/     lm-eval-harness (primary) and lighteval (secondary) wrappers
```

## Methodology

### Layer sensitivity (exp01)

For a fixed precision, exp01 holds every LM Linear layer at fp16 except one, which it
quantizes in isolation, then measures the KL divergence between the resulting logits and
the fp16 baseline logits over a fixed 16-prompt battery. The masked metric
(`safe_logit_kl` in [`src/shared/metrics.py`](src/shared/metrics.py)) drops any token
position where either distribution went non-finite and reports the mean KL over the
surviving positions. That per-layer KL is the layer's sensitivity score.

Quantization is per-output-channel symmetric fake-quant (`fake_quantize` in
[`src/shared/ablator.py`](src/shared/ablator.py)). The scale is max absolute weight per
output row divided by `qmax`, and the weight is then rounded and clamped into the signed
int range (for int4 that is -8 to 7) and dequantized. The `lm_head` is excluded by
default. For multimodal models, the vision and audio tower layers are filtered out by
`list_lm_linear_layers`, so only the text backbone is touched.

The sweep is one forward pass per layer. The layer counts and the resulting protect-set
sizes are below.

| model | LM Linear layers | top 10 percent protected |
|---|---:|---:|
| llama-3.2-3b | 196 | 19 |
| llama-3.1-8b | 224 | 22 |
| qwen3-8b | 252 | 25 |
| gemma-4-4b | 592 | 59 |

Sweeps exist for int2, int3, and int4 (`combined/exp01_*_int{2,3,4}.csv`). Everything
downstream uses int4. The KL distribution is dominated by the `mlp_down` layers, with the
earliest and latest blocks the most sensitive. That is the textbook
quantization-sensitivity pattern, which is a good sign that the metric is measuring
something real.

### Protected-quant variant

There are two consistent implementations. `quantized_variant` in
[`src/shared/quant_variants.py`](src/shared/quant_variants.py) is used by exp04, and the
`protected_quantization` context manager in
[`src/experiments/exp03_capability_damage/protected_quant.py`](src/experiments/exp03_capability_damage/protected_quant.py)
is used by exp03. Both do the same thing.

1. Load the exp01 int4 sensitivity CSV, drop NaN-KL rows, and sort by `kl_div` descending.
2. Take the top `protect_top_pct` (default 10 percent) as the protect set, kept at fp16.
3. Fake-quantize every other LM Linear layer to int4, with `lm_head` excluded.
4. On context exit, restore the original weights.

The exp04 harness and the exp03 probes both wrap this context to get the protected
variant. Everything outside the context manager runs at true fp16.

### The four experiments

| | what it does | output |
|---|---|---|
| exp01 | One-at-a-time INT-N ablation, per-layer KL sensitivity | `exp01_<model>_int{2,3,4}.csv` |
| exp02 | Predict per-layer sensitivity from fp16 weight statistics (kurtosis, outlier ratio, effective rank, condition proxy, role) using leave-one-model-out Ridge and gradient boosting, so the sweep can be skipped on a new model | `exp02_features_<model>.csv`, plus a regression JSON when run |
| exp03 | Three capability-specific probes, fp16 versus protected-int4 | `exp03/*.json` |
| exp04 | Full lm-evaluation-harness sweep at fp16, int4-naive, and int4-protected | `exp04/lm_eval_*.json` |

The exp03 probe designs:

- Calibration. 80 MMLU items each from 5 subjects (high-school US history, high-school
  biology, elementary mathematics, computer security, professional law), for 400 items
  total. It is scored by the probability of the answer letter (A, B, C, or D) at the
  position right after `Answer:`, and reports accuracy, mean confidence, ECE (15 bins),
  and overconfidence. This is single-token scoring, which is relevant to the Gemma
  discussion later.
- Refusal. 20 canonical "harmful instruction" prompts, generated at `max_new_tokens=100`
  through the chat template, classified as refused by a fixed keyword list ("i can't",
  "i'm sorry", and so on). It is a heuristic, and it is labeled as such in the output
  (`"detector": "keyword_heuristic"`).
- Needle in a haystack. A salient fact sentence inserted at depths 0.1, 0.25, 0.5, 0.75,
  and 0.9 into filler at context lengths 1024, 2048, and 4096, with 2 trials per cell for
  30 trials total. Success means the exact code string appears in the generation.

## Models

All loaded via `AutoModelForCausalLM` with `trust_remote_code=True`. Defaults live in
[`src/shared/models.py`](src/shared/models.py).

| short name | HF id | dtype | notes |
|---|---|---|---|
| `llama-3.2-3b` | `meta-llama/Llama-3.2-3B-Instruct` | fp16 | default |
| `llama-3.1-8b` | `meta-llama/Llama-3.1-8B-Instruct` | fp16 | default |
| `qwen3-8b` | `Qwen/Qwen3-8B` | bf16 | fp16 yields all-NaN logits, so it must load in bf16 |
| `gemma-4-4b` | `google/gemma-4-E4B-it` | bf16 | Gemma 4 E4B, about 8B params, multimodal text/image/video/audio "gemma4" architecture. See the limitations section. |

## Results

### exp04, protection recovery on aggregate benchmarks

Recovery percentage is the share of the int4-naive accuracy drop (versus fp16) that
protected-int4 claws back. A value of 100 percent means protection fully restored fp16,
0 percent means no better than naive, and a negative value means protected is worse than
naive on that task. The full numbers are in
[`combined/exp04_summary.csv`](combined/exp04_summary.csv). The three models below have a
clean eval pipeline. Gemma's multi-token scoring path is broken, which is covered in the
limitations section.

| task | llama-3.2-3b | llama-3.1-8b | qwen3-8b | clean-model avg |
|---|---:|---:|---:|---:|
| mmlu | 63% | 49% | 38% | 50% |
| hellaswag | 56% | 64% | 49% | 56% |
| arc_challenge | 51% | 13% | 43% | 36% |
| winogrande | 46% | 49% | 22% | 39% |
| gsm8k | 60% | 62% | 54% | 59% |
| truthfulqa_mc1 | 48% | 52% | 68% | 56% |
| truthfulqa_mc2 | 85% | 54% | 87% | 75% |
| bbq | 65% | 32% | 43% | 47% |
| toxigen | +102% | -512% | -164% | noisy, see below |

A few things worth pulling out of the table:

GSM8K is the most quant-fragile task. Naive int4 drops it by roughly 28 to 40 points from
fp16, but protection recovers about 60 percent of that loss, consistently across all three
models. Mathematical reasoning concentrates in a small number of high-KL layers, and
protecting them buys back most of the capability. This is the cleanest positive result in
the repo.

TruthfulQA-MC2 recovers the most, about 75 percent on average. The model keeps most of
what it knew at fp16 once a handful of sensitive layers are protected.

ARC-Challenge is the most variable, at 51 percent on the 3B but only 13 percent on the 8B.
The global top-10-percent KL ranking does not capture the right layers for that capability
on Llama-3.1.

### exp03, capability-specific probes (fp16 versus protected-int4)

| model | cal acc fp16 | cal acc prot | ECE fp16 | ECE prot | refused fp16 | refused prot | needle fp16 | needle prot |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| llama-3.2-3b | 0.582 | 0.555 | 0.158 | 0.200 | 16/20 | 20/20 | 30/30 | 30/30 |
| llama-3.1-8b | 0.645 | 0.642 | 0.107 | 0.143 | 18/20 | 19/20 | 30/30 | 30/30 |
| qwen3-8b | 0.748 | 0.668 | 0.153 | 0.188 | 4/20 | 7/20 | 30/30 | 30/30 |
| gemma-4-4b | 0.343 | 0.255 | 0.431 | 0.481 | 20/20 | 20/20 | 30/30 | 26/30 |

This is the part the aggregate benchmarks miss.

ECE worsens by 3 to 4 points on every clean model (0.158 to 0.200, 0.107 to 0.143, and
0.153 to 0.188), while accuracy holds within a few points. The protected-int4 models are
more confidently wrong than their fp16 counterparts. This is the most reproducible exp03
signal, with the same direction and similar magnitude across three different models, and
it is exactly the kind of shift that accuracy-based benchmarks cannot see.

Refusal propensity drifts upward under protection. Llama-3.2 goes from 16 to 20 out of 20,
and Qwen goes from 4 to 7 out of 20. This should be read as directional rather than
definitive, because it is a 20-prompt keyword detector, so each prompt is worth 5 points
and the Llama-3.1 result (plus one) and the Gemma result (already at the ceiling) sit
within the noise. The honest claim is that the sign is consistently upward across the
models that are not at a ceiling, not that protection reliably raises refusal by some
fixed amount.

Long-context retrieval is unaffected on every clean model. This is a genuinely easy probe,
with a highly salient needle and only 30 trials, so it should be read as "no gross
long-context damage" rather than as evidence that retrieval is robust in general.

## The MoE experiment (scoped, not yet run)

The natural next model class is mixture-of-experts, and there is a fifth experiment scoped
for it in [`src/experiments/exp01_5_moe_sensitivity/`](src/experiments/exp01_5_moe_sensitivity/),
targeting Llama 4 Maverick (17B active, 128 experts per block). It is designed but not yet
executed, for one reason: I do not currently have the compute to run it. Maverick is
roughly 400B total parameters, about 800 GB in fp16, which needs something like 8x H100
SXM with NVSwitch for a clean fp16 baseline, or 4x H100 if you give up the fp16 baseline
and load int4 only. At on-demand H100 rates a single full sweep lands somewhere around 80
to 150 dollars. That is out of reach for me at the moment, but it is an experiment I would
very much like to finish, and the design is ready to go the day the hardware is.

The reason it needs its own experiment rather than a flag on exp01 is that the dense
ablator does not transfer to MoE. "Ablate one Linear layer" does not have a single
meaning when a block contains 128 experts, so the plan defines three distinct ablation
modes:

- Single-expert. Quantize one expert and leave the other 127 at fp16. This tests how
  sensitive the model is to the specialization of an individual expert.
- Router. Quantize only the gating network. This tests routing fidelity under
  quantization, which has no analog in a dense model.
- Whole-MoE-block. Quantize all 128 experts in one block at once. This is the closest
  analog to dense-model layer ablation, and it is what makes the MoE results comparable to
  the dense findings already in this repo.

The open work is a MoE-aware `Ablator` subclass, the three ablation primitives above,
8-GPU Maverick loading, the three-mode sweeps, and a comparison back against the dense
Tier 1 findings. The interesting hypothesis to test is whether sensitivity concentrates in
the router and a few experts (which would make MoE models cheaper to protect than dense
ones of comparable quality) or whether it spreads across many experts (which would make
them harder).

## What went wrong and other limitations

This is the section that matters for taking the numbers at face value. The code is sound
and the results reproduce. The caveats here are mostly about interpretation, plus one
model whose eval pipeline never produced trustworthy multi-token scores.

### 1. The Gemma failure is real, but it is not "at chance across the board"

`gemma-4-4b` resolves to `google/gemma-4-E4B-it`, which is a genuine recent Gemma 4 E4B
release: about 8B parameters on the multimodal "gemma4" architecture covering text, image,
video, and audio. The model loads and generates fine. The committed fp16 baseline shows a
split:

- Working: MMLU accuracy 0.408, which is well above the 0.25 chance floor (international
  law is 0.66 and world history is 0.65), and GSM8K exact match 0.690.
- At chance: ARC-Challenge accuracy 0.222 (acc_norm 0.249), HellaSwag acc_norm 0.350,
  WinoGrande 0.486, and TruthfulQA-MC1 0.277.

The thing that separates the working tasks from the broken ones is not the task family. It
is single-token versus multi-token continuation scoring. MMLU in lm-eval scores a single
answer-letter continuation, and it works. GSM8K is generative, and it works.
ARC-Challenge, HellaSwag, WinoGrande, and TruthfulQA all score the loglikelihood of
multi-token answer texts, and those are exactly the ones that come back at chance. A
forward path that produces correct logits only at the final position would yield precisely
this signature: single-token scoring fine, generation fine, and multi-token continuation
loglikelihood unusable. That points at the intermediate-position logits for this
freshly-released architecture under the run-time versions of `transformers` and `lm-eval`,
not at the model being incapable.

Dtype was correctly ruled out. Re-running the fp16 baseline in bf16 moved almost nothing
(ARC 0.250 to 0.249, MMLU 0.411 to 0.408). So the earlier framing of "multiple-choice at
chance across the board" was directionally right but imprecise. The sharper statement is
that MMLU and GSM8K on Gemma are usable, and it is the multi-token continuation
loglikelihood path that is broken.

Two follow-ups before any Gemma conclusion is locked in:

- The committed Gemma run predates the repo's own BOS fix. `run_lm_eval.py` now sets
  `add_bos_token=True` for Gemma, which is the right instinct because Gemma scoring is
  BOS-sensitive, but the `combined/gemma-4-4b/exp04/lm_eval.log` has no `add_bos_token`
  line, so these numbers were generated by an earlier version of the script and never
  refreshed. They should be regenerated on the current code. As a caveat, since MMLU works
  even without BOS, BOS is unlikely to be the root cause of the selective pattern, but the
  numbers should be refreshed regardless.
- The Gemma gate in `run_all.sh` gates on `arc_challenge > 0.40`, and ARC is itself a
  multi-token task, which is to say the broken path. So the gate conflates "the model is
  broken" with "multi-token scoring is broken for this architecture," and a fully capable
  Gemma (MMLU 0.41, GSM8K 0.69) fails it for the wrong reason. A better gate would use a
  single-token or generative metric such as MMLU accuracy or GSM8K. As shipped, the gate
  fails, so `combined/gemma-4-4b/exp04/` contains only the fp16 baseline, with no naive and
  no protected run.

### 2. Gemma's exp03 calibration measures the working path, so it is usable

The calibration probe scores the single answer-letter probability at the last position,
which is the same single-token mechanism that gives Gemma its usable MMLU accuracy. So
Gemma's calibration numbers (accuracy 0.343 to 0.255, ECE 0.431 to 0.481, both showing
real degradation under protected-int4) are a legitimate measurement and can be used,
subject to the usual small-subset caveat of 400 items across five harder subjects. Gemma's
refusal and needle probes are generation-based and are likewise fine. To put it plainly,
on Gemma the only untrustworthy results are the multi-token lm-eval tasks. Calibration,
refusal, needle, MMLU accuracy, and GSM8K are all fine.

### 3. ToxiGen recovery percentages are an artifact, but the absolute regression is real

On llama-3.1-8b and qwen3-8b, protected-int4 is worse on ToxiGen than naive int4, by 5.2
and 7.0 points from fp16 respectively, against tiny naive drops of 0.9 and 2.7 points. The
headline -512 percent and -164 percent are artifacts of dividing by a near-zero naive
delta, so the percentage should be ignored and the absolute point regression reported
instead. The sign and magnitude are real, but there is no clean mechanistic story yet. The
top-10-percent KL ranking is driven by aggregate-distribution KL, which need not align with
whatever governs ToxiGen-style classification.

### 4. Probe statistical power

Refusal (20 prompts, keyword detector) and needle (30 trials, salient needle) are
screening probes, not measurements. The refusal keyword detector should be replaced with a
real classifier of the Llama-Guard class and scaled to a few hundred prompts before the
refusal shift is treated as a headline result. Calibration (400 items) and the exp04 suites
(full test splits) are the load-bearing numbers, and ECE is the one exp03 signal robust
enough to lead with.

### 5. exp02 cross-model regression not run on the final data

Per-layer features exist for the two Llamas and Gemma
(`combined/<model>/exp02_features_*.csv`), but the leave-one-model-out regression has not
been run on the post-fix data, and the qwen3-8b features were never produced. Its int4
sweep exists at the top level but was never copied into `qwen3-8b/` or paired with a
features CSV. Until that is run, "predict sensitivity from weight statistics" is plumbing
rather than a result.

### 6. Minor notes

`run_lighteval.py` hardcodes `dtype="float16"` in `TransformersModelConfig` regardless of
`--dtype`. This is harmless today, because the already-loaded model object is swapped in
afterward (`lm.model = model`), so the load-time dtype wins, but it is a latent
inconsistency worth tidying. lighteval is the secondary harness, and lm-eval is primary.

One sanity check worth recording: llama-3.1-8b and qwen3-8b have an identical
protected-versus-fp16 GSM8K delta of -0.155420773313116. This looks like a copy-paste bug
but is not. It is exactly 205 out of 1319 for both, an integer-count collision on a
1319-item test set, and the fp16, naive, and protected scores themselves differ between the
two models. It is benign.

### What checks out

For the record, since these are the things that would invalidate the work if they were
broken: the masked-KL metric handles non-finite logits correctly, the ablator restores
weights exactly on context exit (restoration KL is near zero in the smoke test), the two
protected-quant implementations agree, the exp01 sweeps have zero NaN rows and a physically
sensible sensitivity distribution, and every number in the exp04 summary CSV and the exp03
tables above matches the underlying raw JSONs.

## License

MIT. See [LICENSE](LICENSE).