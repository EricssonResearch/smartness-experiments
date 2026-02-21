import argparse

import flwr as fl
import numpy as np
import pandas as pd
import xgboost as xgb
from flwr.common import Parameters
from flwr.server.strategy import FedXgbBagging
from sklearn.metrics import mean_absolute_error, mean_squared_error
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler


class EvaluatedXgbBagging(FedXgbBagging):
    def aggregate_fit(self, server_round, results, failures):
        # 1. Run the standard aggregation (get the trees)
        agg_parameters, agg_metrics = super().aggregate_fit(server_round, results, failures)

        # 2. Manually trigger your evaluation function here
        if self.evaluate_fn:
            print(f"--- FORCING EVALUATION FOR ROUND {server_round} ---")

            # Note: We pass agg_parameters. If it's None, we might need to use
            # the strategy's internal global model depending on your specific XGB setup.
            # Usually, agg_parameters contains the serialized trees here.

            try:
                eval_res = self.evaluate_fn(server_round, agg_parameters, {})
                if eval_res:
                    loss, metrics = eval_res
                    print(f"Round {server_round} Centralized Accuracy: {metrics}")
            except Exception as e:
                print(f"Evaluation crashed: {e}")

        return agg_parameters, agg_metrics


def nmae(y_pred, y_test):
    mae = mean_absolute_error(y_test, y_pred)
    mean_true = np.mean(np.abs(y_test))
    return mae / mean_true


def get_evaluate_fn(X_test, y_test):
    """
    Return an evaluation function for server-side evaluation.
    """
    print("DEBUG: Factory created.")

    def evaluate(server_round: int, parameters: Parameters, config: dict):
        print(f"DEBUG: Entering evaluation for round {server_round}")

        # If there are no parameters yet (round 0), skip
        if not parameters.tensors:
            print("DEBUG: No parameters received!")
            return None

        # 1. Reconstruct the Global Model from bytes
        # The strategy sends parameters as a list of byte objects.
        # For XGBoost bagging, usually the model is in the first tensor.
        model_bytes = parameters.tensors[0]

        bst = xgb.Booster()
        bst.load_model(bytearray(model_bytes))

        # 2. Run prediction on the Server's Test Set
        dtest = xgb.DMatrix(X_test, label=y_test)
        preds = bst.predict(dtest)

        # 3. Calculate Metric
        mse = mean_squared_error(y_test, preds)
        mae = mean_absolute_error(y_test, preds)
        nmae_val = nmae(preds, y_test)


        print(f"--- SERVER GLOBAL EVALUATION (Round {server_round}) MSE: {mse:.4f} | MAE: {mae:.4f} | NMAE: {nmae_val:.2%} ---")

        return float(mse), {"mse": float(mse), "mae": float(mae), "nmae": float(nmae_val)}

    return evaluate

# Define metric aggregation function
def eval_metrics_aggregation(metrics):
    """
    Aggregates metrics (MSE) received from clients.
    """
    total_num = 0
    total_mse = 0
    for num_examples, metric_dict in metrics:
        total_num += num_examples
        total_mse += num_examples * metric_dict["mse"]

    return {"mse": total_mse / total_num} if total_num > 0 else {"mse": 0.0}


def main(dataset: pd.DataFrame, target_col: str):
    # dataset must contain features + target_col and no timestamp join is done here
    y = dataset[[target_col]].copy()
    X = dataset.drop(columns=[target_col])

    _, X_val, _, y_val = train_test_split(
        X, y, test_size=0.2, random_state=42
    )

    scaler = StandardScaler().fit(X_val)
    X_val = scaler.transform(X_val)

    # Define the strategy
    strategy = EvaluatedXgbBagging(
        fraction_fit=1.0,  # Sample 100% of currently connected clients
        fraction_evaluate=1.0,  # Sample 100% of currently connected clients
        min_fit_clients=2,  # Wait for 2 clients to connect
        min_evaluate_clients=2,  # Wait for 2 clients to connect
        min_available_clients=2,  # Wait for 2 clients to connect
        evaluate_metrics_aggregation_fn=eval_metrics_aggregation,
        evaluate_fn=get_evaluate_fn(X_test=X_val, y_test=y_val),
    )

    # Start the server
    print("Server starting... waiting for 2 clients to connect.")
    fl.server.start_server(
        server_address="0.0.0.0:8080",  # Listen on all interfaces, port 8080
        config=fl.server.ServerConfig(num_rounds=50),
        strategy=strategy,
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", required=True, help="Path to client's features CSV")
    parser.add_argument("--target_data", required=True, help="Path to client's target CSV")
    parser.add_argument("--target_col", required=True, help="Target column name")
    parser.add_argument("--model_name", required=True, help="Model name to save")
    args = parser.parse_args()

    # load CSVs and merge
    x_dataset = pd.read_csv(args.data, low_memory=True).apply(pd.to_numeric, errors="coerce").fillna(0)
    y_dataset = pd.read_csv(args.target_data, low_memory=True).apply(pd.to_numeric, errors="coerce").fillna(0)
    y_dataset = y_dataset[["timestamp", args.target_col]].copy()

    df_merged = pd.merge(x_dataset, y_dataset, on="timestamp", how="inner")

    main(dataset=df_merged, target_col=args.target_col)