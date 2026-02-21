# patched_client.py
import argparse
import tempfile
import os
import logging
import numpy as np
import pandas as pd
import xgboost as xgb
from flwr.common import Status, Parameters, FitRes, EvaluateRes, Code
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_absolute_error, mean_squared_error
import flwr as fl

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")

XGB_PARAMS = {
    "objective": "reg:squarederror",
    "eta": 0.05,
    "max_depth": 6,
    "eval_metric": "rmse",
    "tree_method": "hist",
    "seed": 42,
}

NUM_CLIENTS = 2
LOCAL_NUM_BOOST_ROUND = 10
LOCAL_EARLY_STOP = 3


def nmae(y_pred, y_test):
    mae = mean_absolute_error(y_test, y_pred)
    mean_true = np.mean(np.abs(y_test))
    return mae / mean_true


# ---------- Partition data deterministically ----------
def load_partition(dataset: pd.DataFrame, target_col: str):
    # dataset must contain features + target_col and no timestamp join is done here
    y = dataset[[target_col]].copy()
    X = dataset.drop(columns=[target_col])

    X_train, X_val, y_train, y_val = train_test_split(
        X, y, test_size=0.2, random_state=42
    )

    scaler = StandardScaler().fit(X_train)
    X_train = scaler.transform(X_train)
    X_val = scaler.transform(X_val)

    return xgb.DMatrix(X_train, label=y_train.values.ravel()), xgb.DMatrix(X_val, label=y_val.values.ravel())


# ---------- Flower Client ----------
class XgbClient(fl.client.Client):
    def __init__(self, cid, dataset: pd.DataFrame, target_col: str):
        self.cid = cid

        # dataset must contain features + target_col and no timestamp join is done here
        y = dataset[[target_col]].copy()
        X = dataset.drop(columns=[target_col])

        X_train, X_val, y_train, y_val = train_test_split(
            X, y, test_size=0.2, random_state=42
        )

        scaler = StandardScaler().fit(X_train)
        X_train = scaler.transform(X_train)
        X_val = scaler.transform(X_val)

        self.X_train = X_train
        self.y_train = y_train
        self.X_test = X_val
        self.y_test = y_val

    def get_parameters(self, ins):
        return fl.common.GetParametersRes(
            status=Status(code=Code.OK, message="OK"),
            parameters=Parameters(tensors=[], tensor_type="bytes"),
        )

    def fit(self, ins):
        # 1. Load Global Model (if present)
        global_model_bytes = ins.parameters.tensors[0] if ins.parameters.tensors else b""

        # 2. Setup Data
        dtrain = xgb.DMatrix(self.X_train, label=self.y_train)

        # 3. XGBoost Parameters
        # Standard Regression Params
        params = {
            "booster": "gbtree",
            "num_parallel_tree": 5,

            # Core Regression Settings
            "objective": "reg:squarederror",
            "eval_metric": ['rmse', 'mae'],

            # Tree Structure (Prevent Overfitting)
            "max_depth": 5,  # Depth of 5 is a safe middle ground
            "min_child_weight": 2,  # Conservative splitting

            # Stochastic (Add variety for Bagging strategy)
            "subsample": 0.5,  # Use 50% of rows per tree
            "colsample_bytree": 0.9,  # Use 90% of features per tree

            # Speed & Regularization
            "eta": 0.02, # Learning rate
            "alpha": 0.1,  # L1 regularization (Lasso) - helps ignore noise
            "lambda": 1.0,  # L2 regularization (Ridge) - standard default

            # Hardware
            "nthread": -1,  # Use all CPU cores
        }

        evals = [(dtrain, "train")]

        print(f"\n[Client] Starting training...")

        # 4. Train (Bagging logic: train new trees)
        bst = None
        if global_model_bytes:
            bst = xgb.Booster(params=params)
            bst.load_model(bytearray(global_model_bytes))
            # Continue training / add trees
            bst = xgb.train(
                params,
                dtrain,
                num_boost_round=3,
                xgb_model=bst,
                evals=evals,  # <--- Added
                verbose_eval=True
            )
        else:
            # First round
            bst = xgb.train(
                params,
                dtrain,
                num_boost_round=3,
                evals=evals,       # <--- Added
                verbose_eval=True  # <--- Prints: [0] train-rmse:123.45
            )

        # 5. Save and Return
        local_model_bytes = bst.save_raw("json")
        # !!! FIX: Cast bytearray to bytes explicitly !!!
        local_model_bytes_safe = bytes(local_model_bytes)

        return FitRes(
            status=Status(code=Code.OK, message="OK"),
            parameters=Parameters(tensors=[local_model_bytes_safe], tensor_type="bytes"),
            num_examples=len(self.X_train),
            metrics={},
        )

    def evaluate(self, ins):
        # 1. Load Global Model
        global_model_bytes = ins.parameters.tensors[0]
        if not global_model_bytes:
            return EvaluateRes(
                status=Status(code=Code.OK, message="OK"),
                loss=0.0, num_examples=len(self.X_test), metrics={"mse": 0.0}
            )

        # 2. Predict
        bst = xgb.Booster()
        bst.load_model(bytearray(global_model_bytes))
        dtest = xgb.DMatrix(self.X_test, label=self.y_test)

        preds = bst.predict(dtest)
        mse = mean_squared_error(self.y_test, preds)
        mae = mean_absolute_error(self.y_test, preds)
        nmae_val = nmae(preds, self.y_test)

        # PRINT THE METRIC TO CONSOLE
        print(f"[Client] Evaluation MSE: {mse:.4f} | MAE: {mae:.4f} | NMAE: {nmae_val:.2%}")

        return EvaluateRes(
            status=Status(code=Code.OK, message="OK"),
            loss=float(mse),
            num_examples=len(self.X_test),
            metrics={"mse": float(mse), "mae": float(mae), "nmae": float(nmae_val)},
        )


# ---------- Run client ----------
if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--cid", type=int, required=True)
    parser.add_argument("--data", required=True, help="Path to client's features CSV")
    parser.add_argument("--target_data", required=True, help="Path to client's target CSV")
    parser.add_argument("--target_col", required=True, help="Target column name")
    args = parser.parse_args()

    # load CSVs and merge
    x_dataset = pd.read_csv(args.data, low_memory=True).apply(pd.to_numeric, errors="coerce").fillna(0)
    y_dataset = pd.read_csv(args.target_data, low_memory=True).apply(pd.to_numeric, errors="coerce").fillna(0)
    y_dataset = y_dataset[["timestamp", args.target_col]].copy()

    df_merged = pd.merge(x_dataset, y_dataset, on="timestamp", how="inner")
    logging.info("Client %s merged dataset rows: %s", args.cid, len(df_merged))

    # PASS THE MERGED DF (was y_dataset in your original code — that was the bug)
    print(f"Starting Client {args.cid}...")
    fl.client.start_client(
        server_address="127.0.0.1:8080",
        client=XgbClient(cid=args.cid, dataset=df_merged, target_col=args.target_col),
    )
