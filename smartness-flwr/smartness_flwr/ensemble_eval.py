import argparse
import os
import pickle
import numpy as np
import pandas as pd
from sklearn.compose import TransformedTargetRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import train_test_split

MODEL_DIR = "client_models"

def safe_predict(model, X, fallback_mean_std=None):
    """
    Return predictions in the ORIGINAL target scale.

    The function tries these cases in order:
    1. If model is a TransformedTargetRegressor -> it already returns inverse-transformed preds.
    2. If model has a `.predict()` that returns original scale, nothing to do (we do a heuristic check below).
    3. If model has attributes from a custom normalizer (y_mean_, y_std_) -> de-normalize preds.
    4. If model has an attached scaler object (y_scaler_ or target_scaler_) -> call inverse_transform.
    5. If fallback_mean_std provided -> apply inverse using (pred * std + mean).
    6. Otherwise return raw preds and warn.

    Returns:
      preds_original_scale (1D numpy array)
    """
    # 1) If it's a sklearn TransformedTargetRegressor it already inverse-transforms on predict()
    if isinstance(model, TransformedTargetRegressor):
        return model.predict(X)

    # 2) Try to predict
    preds = model.predict(X)

    # If preds look roughly on same scale as typical y (heuristic)
    # We can't know the true y here; caller should check ranges afterwards.
    # But we try further if available.

    # 3) Custom normalizer attributes (TargetNormalizer style)
    if hasattr(model, "y_mean_") and hasattr(model, "y_std_"):
        try:
            mean = float(model.y_mean_)
            std = float(model.y_std_) if float(model.y_std_) != 0 else 1.0
            return preds * std + mean
        except Exception:
            pass

    # 4) Attached scaler objects
    # common attribute names people use: y_scaler_, target_scaler_, scaler_y_, y_transformer_
    for attr in ("y_scaler_", "target_scaler_", "scaler_y_", "y_transformer_", "transformer_"):
        scaler = getattr(model, attr, None)
        if scaler is not None:
            # If transformer_ is a StandardScaler inside a TransformedTargetRegressor,
            # sklearn would already inverse-transform on predict. But if transformer_ exists standalone:
            try:
                # sklearn scalers expect 2D arrays
                inv = scaler.inverse_transform(preds.reshape(-1, 1)).ravel()
                return inv
            except Exception:
                # some transformers are function transformers or different API
                pass

    # 5) If fallback provided (mean,std)
    if fallback_mean_std is not None:
        mean, std = fallback_mean_std
        if std == 0 or std is None:
            std = 1.0
        return preds * std + mean

    # 6) Last resort: return raw preds (caller should compare ranges with y_test)
    return preds

def evaluate(x_data, y_data, y_col):
    model_files = sorted([os.path.join(MODEL_DIR, f) for f in os.listdir(MODEL_DIR) if f.endswith(".pkl")])
    models = [pickle.load(open(f, "rb")) for f in model_files]
    print("Loaded", len(models), "models")

    _, x_test, _, y_test = train_test_split(x_data, y_data[y_col].to_numpy(),
                                              test_size=0.30, random_state=42)

    # Average predictions
    preds = np.zeros_like(y_test, dtype=float)
    for m in models:
        safed_preds = safe_predict(m, x_test)
        print("Pred Min:", safed_preds.min(), "Pred Max:", safed_preds.max())
        print("Y Min:", y_test.min(), "Y Max:", y_test.max())

        mae = mean_absolute_error(y_test, safed_preds)
        nmae = mae / (y_test.max() - y_test.min()) * 100
        print("MAE:", mae, "NMAE%:", nmae)

        preds += safed_preds

    preds /= max(1, len(models))

    mae = mean_absolute_error(y_test, preds)
    mean_true = np.mean(np.abs(y_test))
    nmae = (mae / float(mean_true))
    range_true = y_test.max() - y_test.min()

    print("MAE:", mae)
    print(f"NMAE:{nmae:.2%}")
    print(f"mean-NMAE: {mae / mean_true:.2%}, range-NMAE: {mae / range_true:.2%}")
    print(f"Pred Min: {preds.min()} | Pred Max: {preds.max()} | Y Min: {y_test.min()} | Y Max: {y_test.max()}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", required=True, help="Path to client's CSV (features)")
    parser.add_argument("--target_data", required=True, help="Path to client's CSV (target)")
    parser.add_argument("--target_col", required=True, help="Path to client's CSV (target)")
    args = parser.parse_args()

    x_dataset = pd.read_csv(args.data, low_memory=True).apply(pd.to_numeric, errors='coerce').fillna(0)
    y_dataset = pd.read_csv(args.target_data, low_memory=True).apply(pd.to_numeric, errors='coerce').fillna(0)

    evaluate(x_dataset, y_dataset, y_col=args.target_col)
