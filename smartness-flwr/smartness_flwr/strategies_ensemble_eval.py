import argparse
import json
import os
import pickle
import numpy as np
import pandas as pd
from sklearn.compose import TransformedTargetRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import train_test_split
from sklearn.linear_model import Ridge

MODEL_DIR = "client_models"

def evaluate(x_data, y_data, y_col, test_size: float = 0.30, val_frac_of_test: float = 0.5, eps: float = 1e-8):
    # 1) create test + leftover (we don't retrain models here)
    X_rem, X_test, y_rem, y_test = train_test_split(
        x_data, y_data[y_col].to_numpy(), test_size=test_size, random_state=42
    )
    # 2) split leftover into val (for weighting) and ignore rest (or used elsewhere)
    if val_frac_of_test > 0:
        X_val, X_unused, y_val, y_unused = train_test_split(
            X_rem, y_rem, test_size=1.0 - val_frac_of_test, random_state=42
        )
    else:
        # fallback: use part of test as val (not ideal)
        X_val, y_val = X_test, y_test

    # Load models
    model_files = sorted([os.path.join(MODEL_DIR, f) for f in os.listdir(MODEL_DIR) if f.endswith(".pkl")])
    models = [pickle.load(open(f, "rb")) for f in model_files]
    print("Loaded", len(models), "models")

    # Per-model checks & metrics (on VAL and TEST)
    val_maes = []
    test_maes = []
    print("\nPer-model metrics (VAL -> TEST) and prediction ranges:")
    for fname, m in zip(model_files, models):
        # predictions on val (for weighting)
        pred_val = m.predict(X_val)
        mae_val = mean_absolute_error(y_val, pred_val)
        val_maes.append(mae_val)

        # predictions on test
        pred_test = m.predict(X_test)
        mae_test = mean_absolute_error(y_test, pred_test)
        test_maes.append(mae_test)

        print(f"{os.path.basename(fname)} | val MAE: {mae_val:.4f} | test MAE: {mae_test:.4f} "
              f"| pred_test range: ({pred_test.min():.3f}, {pred_test.max():.3f}) "
              f"| y_test range: ({y_test.min():.3f}, {y_test.max():.3f})")

    mean_true = np.mean(np.abs(y_test))
    range_true = y_test.max() - y_test.min()

    def print_nmae(mae):
        # both mean-based and range-based NMAE
        return f"mean-NMAE: {mae / mean_true:.2%}, range-NMAE: {mae / range_true:.2%}"

    print("\nSingle-model NMAE (on TEST):")
    for mae in test_maes:
        print(" ", print_nmae(mae))

    # --------------------------
    # 3) Ensemble: simple average
    # --------------------------
    preds_mean = np.zeros_like(y_test, dtype=float)
    for m in models:
        preds_mean += m.predict(X_test)
    preds_mean /= max(1, len(models))
    mae_mean = mean_absolute_error(y_test, preds_mean)
    print("\nSimple average ensemble -> MAE:", mae_mean, "|", print_nmae(mae_mean))

    # --------------------------
    # 4) Ensemble: median
    # --------------------------
    preds_per_model = np.vstack([m.predict(X_test) for m in models])  # shape (n_models, n_samples)
    preds_median = np.median(preds_per_model, axis=0)
    mae_med = mean_absolute_error(y_test, preds_median)
    print("Median ensemble -> MAE:", mae_med, "|", print_nmae(mae_med))

    # --------------------------
    # 5) Ensemble: weighted by val MAE (inverse)
    # --------------------------
    val_maes = np.array(val_maes)
    inv = 1.0 / (val_maes + eps)
    weights = inv / inv.sum()
    print("Weights (inverse-val-MAE):", weights)

    preds_weighted = np.zeros_like(y_test, dtype=float)
    for w, m in zip(weights, models):
        preds_weighted += w * m.predict(X_test)

    mae_w = mean_absolute_error(y_test, preds_weighted)
    print("Weighted ensemble -> MAE:", mae_w, "|", print_nmae(mae_w))

    # --------------------------
    # 6) Optional: stacking (meta-learner) using VAL
    # --------------------------
    # Build meta-features on val and test
    val_preds = np.vstack([m.predict(X_val) for m in models]).T  # (n_val, n_models)
    test_preds = np.vstack([m.predict(X_test) for m in models]).T  # (n_test, n_models)

    meta = Ridge(alpha=1.0)
    meta.fit(val_preds, y_val)
    meta_pred = meta.predict(test_preds)
    mae_meta = mean_absolute_error(y_test, meta_pred)
    print("Stacking (Ridge) -> MAE:", mae_meta, "|", print_nmae(mae_meta))

    # Return a dictionary of results
    return {
        "per_model_val_maes": val_maes.tolist(),
        "per_model_test_maes": test_maes,
        "mean_ensemble_mae": mae_mean,
        "median_ensemble_mae": mae_med,
        "weighted_ensemble_mae": mae_w,
        "stacking_mae": mae_meta,
        "mean_true": mean_true,
        "range_true": range_true,
    }


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", required=True, help="Path to client's CSV (features)")
    parser.add_argument("--target_data", required=True, help="Path to client's CSV (target)")
    parser.add_argument("--target_col", required=True, help="Path to client's CSV (target)")
    args = parser.parse_args()

    x_dataset = pd.read_csv(args.data, low_memory=True).apply(pd.to_numeric, errors='coerce').fillna(0)
    y_dataset = pd.read_csv(args.target_data, low_memory=True).apply(pd.to_numeric, errors='coerce').fillna(0)

    pretty_json_string = json.dumps(evaluate(x_dataset, y_dataset, y_col=args.target_col), indent=4)
    print(pretty_json_string)
