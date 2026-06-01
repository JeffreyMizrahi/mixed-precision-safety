# Mixed-Precision Safety

Layer-wise sensitivity attribution for quantized language models, and what it says about which capabilities get quietly degraded by post-training quantization.

## Experiments

1. **Layer sensitivity sweep** (`src/experiments/exp01_layer_sensitivity/`) — ablate every Linear layer to INT-N, measure KL vs the full-precision baseline.
2. **Feature regression** (`src/experiments/exp02_feature_regression/`) — predict per-layer sensitivity from FP16 weight statistics, across models.
3. **Capability-specific damage** (`src/experiments/exp03_capability_damage/`) — does protected-quant silently break refusal, calibration, or long-context retrieval, even when MMLU is preserved? Three probes: calibration, refusal, needle-in-haystack.
4. **Real benchmarks** (`src/experiments/exp04_real_benchmarks/`) — lm-eval-harness sweep (MMLU, HellaSwag, ARC, WinoGrande, GSM8K, TruthfulQA, BBQ, ToxiGen) across FP16, int4-naive, and int4-protected.

## Models

Single-GPU lineup (A6000-class):

- `llama-3.2-3b` (meta-llama/Llama-3.2-3B-Instruct), fp16
- `llama-3.1-8b` (meta-llama/Llama-3.1-8B-Instruct), fp16
- `qwen3-8b` (Qwen/Qwen3-8B), bf16 — fp16 yields all-NaN logits
- `gemma-4-4b` (google/gemma-4-E4B-it, a multimodal Gemma-3n / MatFormer), bf16 with a gate (see Known issues)

Also defined but not in the headline lineup: `llama-3.1-405b`, `llama-4-maverick` (MoE), `pythia-1.4b` (smoke), `qwen3-1.7b`, `qwen3-4b`, `qwen3-14b`.

## Workflow

Deploys the repo to a Thundercompute instance, sets up venv + HF auth, and launches `run_all.sh` (all four models, exp04 + exp03) detached.

Locally, from the repo root:
```bash
export HF_TOKEN=hf_xxxxxxxxxxxxxxxxxxxxxx
bash deploy_to_thunder.sh 0          # substitute 0 with your instance id
```

On the instance:
```bash
tnr connect 0
bash /home/ubuntu/mps/remote_setup.sh
```

`remote_setup.sh` defaults to launching `run_all.sh` detached. When it prints "Safe to disconnect," exit with Ctrl+D. Monitor with:
```bash
tail -f /home/ubuntu/mps/run_all.log
```

Pull results back when done:
```bash
tnr scp 0:/home/ubuntu/mps/src/results/ ./src/
```

## Repository layout

```
deploy_to_thunder.sh    Local: rsync project to the instance via tnr scp
remote_setup.sh         On instance: venv, deps, HF auth, launches run_all.sh
run_all.sh              Full pipeline: exp04 lm-eval + exp03 probes, all 4 models
overnight_run.py        Original orchestrator (exp01 + exp02 + exp03 per model)
configs/overnight.yaml  Model list + probe sizes for overnight_run.py
combined/               Canonical merged results tree (per model: exp01-04)
src/
  shared/               Building blocks: ablator, models, metrics, prompts, utils
  experiments/
    exp01_layer_sensitivity/   Per-layer INT-N ablation sweep
    exp02_feature_regression/  Sensitivity prediction from weights
    exp03_capability_damage/   Calibration / refusal / needle probes
    exp04_real_benchmarks/     lm-eval-harness sweep
```

## Per-model dtype gotchas

- **qwen3-8b**: fp16 overflows on some inputs and produces all-NaN logits. Must run in bf16 for both the exp01 sensitivity sweep (otherwise the CSV is all-NaN and protected silently collapses to naive) and exp04. `run_all.sh` self-heals by regenerating the sensitivity CSV in bf16 if it is missing or all-NaN.
- **gemma-4-4b**: multiple-choice eval scores at chance (arc_challenge acc_norm = 0.25, winogrande = 0.49) while generation works (gsm8k = 0.69). bf16 does not fix it — the bug lives in the `AutoModelForCausalLM` text-backbone load or the HFLM loglikelihood path for this Gemma-3n / MatFormer arch. `run_all.sh` runs gemma fp16 in bf16 first and only spends exp01 + naive + protected hours if `arc_challenge > 0.40`; otherwise it stops and reports.

## License

MIT.
