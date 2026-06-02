# Experiment 02: Predicting sensitivity from weights

This experiment asks whether a layer's quantization sensitivity can be predicted from
cheap statistics of its fp16 weights alone, without running the exp01 ablation sweep. If a
small regression on weight features can predict the per-layer KL well enough, the
expensive sweep can be skipped on a new model.

## What it does

There are two stages.

Stage one, feature extraction (`extract_features.py`), computes a fixed set of per-layer
features straight from the fp16 weights. For each LM Linear layer it records:

- Structural features: normalized depth in the stack, input and output dimensions,
  parameter count, and the layer role (attention QKV, attention output, MLP up, MLP down,
  or other).
- Distributional features: weight kurtosis, max absolute weight, mean absolute weight, and
  an outlier ratio (max over mean absolute weight).
- Spectral features: an effective rank from the entropy of the singular-value spectrum, and
  a condition-number proxy (the ratio of the largest singular value to the value at the 5th
  percentile).

Stage two, regression (`cross_model_regression.py`), joins those features to the exp01
`kl_div` targets, takes the target as log10 of KL, and fits two models on it: a Ridge
linear regression and a gradient boosting regressor (sklearn's
`GradientBoostingRegressor`, 200 trees, depth 3). The role column is one-hot encoded and
concatenated with the numeric features.

Generalization is measured two ways. Leave-one-model-out trains on all models but one and
tests on the held-out model, reporting R-squared for both regressors plus the gradient
boosting feature importances. Within-model uses an 80/20 split per model and reports the
gradient boosting R-squared.

## How to run it

Extract features for a model:

```bash
cd src
python -m experiments.exp02_feature_regression.extract_features \
    --model llama-3.2-3b \
    --output ../combined/llama-3_2-3b/exp02_features_llama-3_2-3b.csv
```

Then run the cross-model regression over all feature and sensitivity CSVs in a directory:

```bash
python -m experiments.exp02_feature_regression.cross_model_regression \
    --results-dir ../combined --bits 4 \
    --output ../combined/exp02_regression_int4.json
```

The regression pairs each `exp01_<slug>_int4.csv` with its matching
`exp02_features_<slug>.csv` and skips any model that is missing one of the two.

## Output

`extract_features.py` writes one CSV per model with one row per layer and the feature
columns above. `cross_model_regression.py` writes a JSON summary with the per-held-out-model
R-squared values, the top features by importance, and the within-model R-squared values.

## Current status

Feature CSVs exist for the two Llamas and Gemma. The cross-model regression has not been
run on the final post-fix data, and qwen3-8b features were never produced, so its int4
sweep at the top level cannot be included until the features are extracted. Until the
regression is run end to end, treat the stage-one features as plumbing rather than a result.