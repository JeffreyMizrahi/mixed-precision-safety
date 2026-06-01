# Experiment 01 — Layer-wise Sensitivity Mapping

Ablate every Linear layer in the model independently to INT-N, measure
KL divergence and top-1 disagreement vs the FP16 baseline, output a CSV.

## Running

```bash
# Smoke test first
python -m experiments.exp01_layer_sensitivity.smoke_test

# Tier 1 sweeps (run all 4 small models at INT4/INT3/INT2)
bash scripts/run_tier1_sweep.sh

# Or one model at a time
python -m experiments.exp01_layer_sensitivity.full_sweep --model qwen3-8b --bits 4
```

## Status

- [x] Ablator implementation
- [x] Smoke test
- [x] Full-sweep script
- [ ] Tier 1 sweep results (4 models x 3 precisions)
- [ ] Analysis notebook with depth profiles + role aggregates
- [ ] Cross-model role-consistency analysis
