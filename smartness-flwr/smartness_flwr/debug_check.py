# debug_check.py
import os, pickle, numpy as np, pandas as pd
import argparse
from sklearn.metrics import mean_absolute_error
from sklearn.model_selection import train_test_split

MODEL_DIR = "client_models"

def load_model(path):
    with open(path, "rb") as f:
        return pickle.load(f)

def debug_check(X, y, y_col):
    _, x_test, _, y_test = train_test_split(X, y[y_col].to_numpy(),
                                            test_size=0.30, random_state=42)

    for path in sorted(os.listdir(MODEL_DIR)):
        if not path.endswith(".pkl"):
            continue
        mpath = os.path.join(MODEL_DIR, path)
        model = load_model(mpath)
        preds = model.predict(x_test)  # should be on original scale if transformer inverse works
        mae = mean_absolute_error(y_test, preds)
        denom = y_test.max() - y_test.min()
        nmae = mae / denom * 100
        print(path, "MAE:", mae, "NMAE%:", round(nmae, 2))


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", required=True, help="Path to client's CSV (features)")
    parser.add_argument("--target_data", required=True, help="Path to client's CSV (target)")
    parser.add_argument("--target_col", required=True, help="Path to client's CSV (target)")
    args = parser.parse_args()

    x_dataset = pd.read_csv(args.data, low_memory=True).apply(pd.to_numeric, errors='coerce').fillna(0)
    y_dataset = pd.read_csv(args.target_data, low_memory=True).apply(pd.to_numeric, errors='coerce').fillna(0)

    debug_check(x_dataset, y_dataset, args.target_col)