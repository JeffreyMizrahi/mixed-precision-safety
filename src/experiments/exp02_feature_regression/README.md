# Experiment 02 — Predicting Sensitivity from Weights

**Status: blocked on Experiment 1 results.**

Can we predict a layer's quantization sensitivity from cheap statistics
of its FP16 weights alone, without running the ablation?

## Method

1. Extract per-layer features: spectral (singular values, condition
   number), distributional (variance, kurtosis, max-abs), structural
   (depth, role, dimensions), and quantization-error proxies (per-row
   scale variance, MSE between FP16 and quantized).
2. Train regression models (XGBoost, MLP) on Experiment 1's KL targets.
3. Test generalization: cross-model, cross-scale, cross-precision.

## Status

- [ ] Feature extraction code
- [ ] Regression baselines
- [ ] Cross-model evaluation
- [ ] Feature importance analysis
