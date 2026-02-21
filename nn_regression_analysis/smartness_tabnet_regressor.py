import argparse

import numpy as np
import torch
import pandas as pd
from pytorch_tabnet.tab_model import TabNetRegressor
from sklearn.metrics import mean_absolute_error
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler


def run(x_dataset, y_dataset, y_col, model_name):
    y_dataset_aux = y_dataset[y_col].to_numpy()
    y_log = np.log1p(y_dataset_aux).reshape(-1, 1)

    X_train, X_test, y_train, y_test = train_test_split(
        x_dataset, y_log, test_size=0.2, random_state=42
    )

    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train).astype(np.float32)
    X_test = scaler.transform(X_test).astype(np.float32)

    # Make sure targets are float32 as well
    y_train = y_train.astype(np.float32)
    y_test = y_test.astype(np.float32)

    # ---------------------------
    # 5. Create TabNet model
    # ---------------------------
    tabnet = TabNetRegressor(
        n_d=32, n_a=32,
        n_steps=7,
        gamma=1.5,
        lambda_sparse=1e-5,
        optimizer_fn=torch.optim.AdamW,
        optimizer_params=dict(lr=5e-4, weight_decay=1e-5),
        mask_type="sparsemax",
    )

    # ---------------------------
    # 6. Fit the model (y is log-transformed!)
    # ---------------------------
    tabnet.fit(
        X_train, y_train,
        eval_set=[(X_test, y_test)],
        eval_name=['valid'],
        eval_metric=["mae"],
        max_epochs=200,
        patience=30,
        batch_size=128,
        virtual_batch_size=64,
        num_workers=0,
        drop_last=False
    )

    # === After training: diagnostics ===  # contains 'loss', 'valid_0_mae' (depends on version)
    try:
        import matplotlib.pyplot as plt

        # Plot loss
        plt.figure(figsize=(8, 5))
        plt.plot(tabnet.history.loss, label="train_loss")

        # If validation MAE exists
        if "valid_0" in tabnet.history.metrics and "mae" in tabnet.history.metrics["valid_0"]:
            plt.plot(tabnet.history.metrics["valid_0"]["mae"], label="valid_mae")

        plt.legend()
        plt.title("Training Curves")
        plt.xlabel("Epoch")
        plt.ylabel("Loss / MAE")
        plt.show()

    except Exception as e:
        print("Plot skipped:", e)

    # ---------------------------
    # 7. Predict (still in log-scale)
    # ---------------------------
    pred_log = tabnet.predict(X_test).ravel()

    # ---------------------------
    # 8. Inverse log-transform the predictions
    #    expm1(x) = exp(x) - 1 (inverse of log1p)
    # ---------------------------
    pred = np.expm1(pred_log)

    # Also inverse transform the true labels for correct evaluation
    y_test_raw = np.expm1(y_test).ravel()

    # ---------------------------
    # 9. Evaluate in original scale
    # ---------------------------
    mae = mean_absolute_error(y_test_raw, pred)
    print(f"MAE (original scale): {mae:.4f}")

    mean_true = np.mean(np.abs(y_test_raw))
    print(f"NMAE (original scale): {mae / mean_true:.2%}")

    # Feature importances
    try:
        fi = tabnet.feature_importances_
        print("Top 20 feature importances (index,importance):")
        top_idx = np.argsort(fi)[::-1][:20]
        for i in top_idx:
            print(i, fi[i])
    except Exception:
        pass

    # ---------------------------
    # 10. Save model
    # ---------------------------
    tabnet.save_model(f"tabnet_models/{model_name}_tabnet_regressor_log")
    print("Model saved!")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", required=True, help="Path to client's CSV (features)")
    parser.add_argument("--target_data", required=True, help="Path to client's CSV (target)")
    parser.add_argument("--target_col", required=True, help="Path to client's CSV (target)")
    parser.add_argument("--model_name", required=True, help="Model name to save model on disk")
    args = parser.parse_args()

    x_dataset = pd.read_csv(args.data, low_memory=True).apply(pd.to_numeric, errors='coerce').fillna(0)
    y_dataset = pd.read_csv(args.target_data, low_memory=True).apply(pd.to_numeric, errors='coerce').fillna(0)

    run(x_dataset, y_dataset, y_col=args.target_col, model_name=args.model_name)
