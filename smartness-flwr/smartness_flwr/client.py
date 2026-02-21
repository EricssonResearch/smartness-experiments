import flwr as fl
import pandas as pd
import numpy as np
import argparse
import pickle

from flwr.common import Scalar, NDArrays
from sklearn.base import BaseEstimator, RegressorMixin
from sklearn.compose import TransformedTargetRegressor
from sklearn.metrics import mean_absolute_error
from sklearn.model_selection import train_test_split
from sklearn.pipeline import make_pipeline
from sklearn.ensemble import RandomForestRegressor
from sklearn.preprocessing import StandardScaler
from sklearn.utils.validation import check_is_fitted


class SklearnNumPyClient(fl.client.NumPyClient):
    def __init__(self, cid: str, X_train, y_train):
        self.cid = cid
        self.X = X_train
        self.y = y_train

        base = RandomForestRegressor(
            n_estimators=120,
            random_state=42,
            n_jobs=-1,
        )
        self.model = TransformedTargetRegressor(regressor=base,
                                                transformer=StandardScaler())

    def get_parameters(self, config: dict[str, Scalar]) -> NDArrays:
        return []

    def fit(self, config: dict[str, Scalar], parameters: NDArrays):
        self.model.fit(X=self.X, y=self.y)

        b = pickle.dumps(self.model)
        arr = np.frombuffer(b, dtype=np.uint8)

        metrics = {"train_examples": len(self.y)}
        return [arr], len(self.y), metrics

    def evaluate(
            self, parameters: NDArrays, config: dict[str, Scalar]
    ) -> tuple[float, int, dict[str, Scalar]]:
        # If model not fitted, fit it now (safe fallback)
        try:
            check_is_fitted(self.model)
        except Exception:
            # model not fitted -> fit on local data
            # Optionally, you can fit on a small local holdout instead of entire X,y
            self.model.fit(self.X, self.y)

        preds = self.model.predict(self.X)
        mae = mean_absolute_error(self.y, preds)
        mean_true = np.mean(np.abs(self.y))
        nmae = (mae / float(mean_true))

        print("MAE:", mae)
        print(f"NMAE:{nmae:.2%}")

        return float(nmae), len(self.y), {"nmae": float(nmae)}


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--cid", required=True, help="Client id")
    parser.add_argument("--data", required=True, help="Path to client's CSV (features)")
    parser.add_argument("--target_data", required=True, help="Path to client's CSV (target)")
    parser.add_argument("--target_col", required=True, help="Path to client's CSV (target)")
    parser.add_argument("--server_address", default="localhost:8080", help="Flower server address")
    args = parser.parse_args()

    x_data = pd.read_csv(args.data, low_memory=True).apply(pd.to_numeric, errors='coerce').fillna(0)
    y_data = pd.read_csv(args.target_data, low_memory=True).apply(pd.to_numeric, errors='coerce').fillna(0)
    x_train, _, y_train, _ = train_test_split(x_data, y_data[args.target_col].to_numpy(),
                                              test_size=0.30, random_state=42)

    client = SklearnNumPyClient(args.cid, x_train, y_train)
    fl.client.start_numpy_client(server_address=args.server_address, client=client)
