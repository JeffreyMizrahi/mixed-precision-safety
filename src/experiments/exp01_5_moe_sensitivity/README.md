# Experiment 1.5 — MoE Sensitivity (Llama 4 Maverick)

**Status: scoped, blocked on hardware (8x H100 SXM required).**

Dense ablator doesn't transfer to MoE. Maverick has 128 experts x 17B
params per block; "ablate one Linear" doesn't have a single meaning.

## Three ablation modes

- **Single-expert** — quantize one expert, leave 127 in FP16.
  Tests expert specialization sensitivity.
- **Router** — quantize the gating network only.
  Tests routing fidelity under quantization.
- **Whole-MoE-block** — quantize all 128 experts in one block.
  Closest analog to dense-model layer ablation.

## Hardware

400B total params, 800GB FP16. Requires 8x H100 SXM with NVSwitch,
or 4x H100 with INT4-only loading (no FP16 baseline). ~$80-150 per
sweep at on-demand H100 rates.

## Status

- [ ] MoE-aware Ablator subclass
- [ ] Single-expert / router / whole-block primitives
- [ ] Maverick model loading (8-GPU)
- [ ] Three-mode sweeps
- [ ] Comparison vs Tier 1 dense findings
