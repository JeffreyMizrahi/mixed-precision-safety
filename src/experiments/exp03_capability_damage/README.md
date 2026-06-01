# Experiment 03 — Capability-Specific Damage

**Status: blocked on Experiments 1 and 2.**

Can quantization silently degrade specific safety-relevant capabilities
(refusal, calibration, long-context, multi-step reasoning) while leaving
aggregate benchmarks like MMLU intact?

## Method

1. Construct a "stealthy" quantized model: aggressive whole-model INT4
   with high-sensitivity layers held at INT8 (heuristic from Exp 1+2).
2. Evaluate against capability-specific probes:
   - HarmBench / AdvBench (refusal robustness)
   - ECE on MMLU (calibration)
   - Needle-in-a-haystack (long-context)
   - GSM8K, BBH (multi-step reasoning)
   - SycophancyEval
3. Compare delta-MMLU vs delta-capability metrics.
4. Layer attribution for any degraded capability.

## Status

- [ ] Whole-model quantization pipeline
- [ ] HarmBench harness
- [ ] Calibration eval
- [ ] Long-context eval
- [ ] Layer attribution for degraded capabilities
