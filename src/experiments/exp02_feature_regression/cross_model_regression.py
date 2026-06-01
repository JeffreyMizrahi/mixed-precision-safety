from __future__ import annotations

import argparse
import json
import logging
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.linear_model import Ridge
from sklearn.metrics import r2_score
from sklearn.preprocessing import OneHotEncoder

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(message)s",
    datefmt="%H:%M:%S",
)
log = logging.getLogger("regression")


NUMERIC_FEATURES = [
    "layer_index_normalized",
    "n_in",
    "n_out",
    "n_params",
    "weight_kurtosis",
    "weight_max_abs",
    "weight_mean_abs",
    "weight_outlier_ratio",
    "effective_rank",
    "condition_proxy",
]


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--results-dir", default=None)
    p.add_argument("--bits", type=int, default=4)
    p.add_argument("--output", default=None)
    return p.parse_args()


def default_results_dir() -> Path:
    return Path(__file__).resolve().parents[2] / "src" / "results"


def run_regression(
    results_dir: Path,
    bits: int,
    output_path: Path,
) -> dict:
    results_dir = Path(results_dir)
    abl_pattern = f"exp01_*_int{bits}.csv"
    log.info("Looking for %s in %s", abl_pattern, results_dir)

    frames = []
    for abl_csv in sorted(results_dir.glob(abl_pattern)):
        stem = abl_csv.stem
        model_slug = stem[len("exp01_"):-len(f"_int{bits}")]
        feat_csv = results_dir / f"exp02_features_{model_slug}.csv"
        if not feat_csv.exists():
            log.warning("Skipping %s, no matching features file at %s",
                        model_slug, feat_csv.name)
            continue
        abl = pd.read_csv(abl_csv)
        feat = pd.read_csv(feat_csv)
        merged = feat.merge(
            abl[["layer_name", "kl_div", "top1_disagree"]],
            on="layer_name",
            how="inner",
        )
        merged["model"] = model_slug
        merged = merged.dropna(subset=["kl_div"])
        merged["log_kl"] = np.log10(merged["kl_div"].clip(lower=1e-9))
        frames.append(merged)
        log.info("  %s: %d valid rows", model_slug, len(merged))

    if not frames:
        log.error("No usable model data found in %s", results_dir)
        return {"error": "no data"}

    df = pd.concat(frames, ignore_index=True)
    log.info("Combined: %d rows across %d models", len(df), df["model"].nunique())

    role_enc = OneHotEncoder(sparse_output=False, handle_unknown="ignore")
    role_onehot = role_enc.fit_transform(df[["role"]])
    role_cols = [f"role_{c}" for c in role_enc.categories_[0]]

    X_numeric = df[NUMERIC_FEATURES].fillna(0.0).to_numpy()
    X = np.concatenate([X_numeric, role_onehot], axis=1)
    feature_names = NUMERIC_FEATURES + role_cols
    y = df["log_kl"].to_numpy()
    models_array = df["model"].to_numpy()

    summary: dict = {
        "bits": bits,
        "n_total_rows": int(len(df)),
        "models": sorted(df["model"].unique().tolist()),
        "feature_names": feature_names,
        "leave_one_out": {},
        "within_model": {},
    }

    for held_out in sorted(df["model"].unique()):
        train_mask = models_array != held_out
        test_mask = ~train_mask
        if train_mask.sum() == 0 or test_mask.sum() == 0:
            continue

        ridge = Ridge(alpha=1.0).fit(X[train_mask], y[train_mask])
        gbr = GradientBoostingRegressor(
            n_estimators=200, max_depth=3, random_state=0
        ).fit(X[train_mask], y[train_mask])

        ridge_r2 = r2_score(y[test_mask], ridge.predict(X[test_mask]))
        gbr_r2 = r2_score(y[test_mask], gbr.predict(X[test_mask]))

        feat_imp = dict(zip(feature_names, gbr.feature_importances_.tolist()))
        summary["leave_one_out"][held_out] = {
            "n_train": int(train_mask.sum()),
            "n_test": int(test_mask.sum()),
            "ridge_r2": float(ridge_r2),
            "gbr_r2": float(gbr_r2),
            "top_features_by_importance": sorted(
                feat_imp.items(), key=lambda x: -x[1]
            )[:8],
        }
        log.info("  LOO held=%s: ridge R2=%.3f, gbr R2=%.3f",
                 held_out, ridge_r2, gbr_r2)

    rng = np.random.default_rng(0)
    for m in sorted(df["model"].unique()):
        mask = models_array == m
        idx = np.where(mask)[0]
        rng.shuffle(idx)
        split = int(0.8 * len(idx))
        train_i, test_i = idx[:split], idx[split:]
        if len(test_i) == 0:
            continue
        gbr = GradientBoostingRegressor(
            n_estimators=200, max_depth=3, random_state=0
        ).fit(X[train_i], y[train_i])
        r2 = r2_score(y[test_i], gbr.predict(X[test_i]))
        summary["within_model"][m] = {"gbr_r2_80_20": float(r2)}

    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w") as f:
        json.dump(summary, f, indent=2)
    log.info("Wrote regression summary to %s", output_path)
    return summary


def main() -> int:
    args = parse_args()
    results_dir = Path(args.results_dir) if args.results_dir else default_results_dir()
    output = Path(args.output) if args.output else results_dir / f"exp02_regression_int{args.bits}.json"
    run_regression(results_dir, args.bits, output)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
