import pickle

import numpy as np
import flwr as fl
from flwr.common import FitRes, Parameters, Scalar, parameters_to_ndarrays
from flwr.server.strategy import FedAvg
import os

MODEL_DIR = "client_models"
os.makedirs(MODEL_DIR, exist_ok=True)


class CollectModelsStrategy(FedAvg):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)

    def aggregate_fit(
            self,
            rnd,
            results,
            failures,
    ):
        model_count = 0

        if results is None:
            return None, {}

        for client_proxy, fit_res in results:
            if fit_res is None or fit_res.parameters is None:
                continue

            try:
                ndarrays = parameters_to_ndarrays(fit_res.parameters)
            except Exception as e:
                print(f"[Server] Failed to convert parameters from client {client_proxy.cid}: {e}")
                continue

            if len(ndarrays) == 0:
                continue

            arr = ndarrays[0]
            if not isinstance(arr, np.ndarray) or arr.dtype != np.uint8:
                print(
                    f"[Server] Unexpected parameter type from client {client_proxy.cid}: dtype={getattr(arr, 'dtype', None)}")
                continue

            try:
                raw_bytes = arr.tobytes()
                model = pickle.loads(raw_bytes)
            except Exception as e:
                print(f"[Server] Failed to unpickle model from client {client_proxy.cid}: {e}")
                continue

            fname = os.path.join(MODEL_DIR, f"model_round{rnd}_client{client_proxy.cid}.pkl")
            with open(fname, "wb") as f:
                pickle.dump(model, f)

            print(f"[Server] Saved client {client_proxy.cid} model to {fname}")
            model_count += 1

        print(f"[Server] Round {rnd} collected {model_count} models")

        return None, {}

def start_server(num_rounds: int = 3, server_address: str = "0.0.0.0:8080"):
    strategy = CollectModelsStrategy()

    server_config = fl.server.ServerConfig(num_rounds=num_rounds)
    fl.server.start_server(server_address=server_address, config=server_config, strategy=strategy)

if __name__ == "__main__":
    start_server(num_rounds=3, server_address="0.0.0.0:8080")