import flwr as fl
import numpy as np
import torch
import sys
import os
import argparse
from typing import Dict, Optional, Tuple, List
from flwr.server.client_manager import ClientManager
from flwr.server.client_proxy import ClientProxy
from flwr.server.strategy import FedAvg
from flwr.common import Parameters, FitRes, Scalar, EvaluateRes

# Importa as classes e funções do seu arquivo utils
from utils import WideAndDeepModel, get_weights, set_weights, initialize_weights, preprocess_data, get_weights, \
    set_weights

CLIENTS_REQUIRED = 2
NUM_ROUNDS = 20


# --- Aggregation Functions (Identical to previous fixes) ---
def aggregate_metrics(metrics: List[Tuple[int, Dict[str, Scalar]]]):
    """Aggregates the NMAE metric across all participating clients."""
    weighted_nmaes = [num_examples * m["nmae"] for num_examples, m in metrics]
    examples = [num_examples for num_examples, _ in metrics]
    aggregated_nmae = sum(weighted_nmaes) / sum(examples)
    return {"NMAE_aggregated": aggregated_nmae}


def aggregate_train_loss_metrics(metrics: List[Tuple[int, Dict[str, Scalar]]]):
    """Aggregates the client's final training loss (L1 Log Scale)."""
    weighted_losses = [num_examples * m["client_train_loss_log"] for num_examples, m in metrics]
    examples = [num_examples for num_examples, _ in metrics]
    aggregated_loss = sum(weighted_losses) / sum(examples)
    return {"TrainLoss_aggregated_log": aggregated_loss}

# --- Nova Classe de Estratégia para Salvar o Modelo ---
class SaveModelStrategy(FedAvg):
    """
    Subclasse de FedAvg que adiciona a funcionalidade de salvar o modelo
    após a última rodada de agregação.
    """

    def __init__(self, in_features, model_path, **kwargs):
        super().__init__(**kwargs)
        self.in_features = in_features
        self.model_path = model_path

    def aggregate_evaluate(
            self,
            rnd: int,
            results: List[Tuple[ClientProxy, fl.common.EvaluateRes]], # NOTE: We still use EvaluateRes for strict typing, but handle the output
            failures: List[Tuple[ClientProxy, fl.common.EvaluateRes]],
    ) -> tuple[float | None, dict[str, Scalar]]:
        # 1. Manually separate results into loss_results and metrics_results

        # --- FIX 1: Correctly unpack the tuple elements ---
        # The list comprehension structure must be:
        # [(evaluate_res.num_examples, evaluate_res.loss) for client_proxy, evaluate_res in results]

        # Loss results for MAE aggregation (the default FedAvg behavior)
        loss_results = [(res.num_examples, res.loss) for _, res in results]

        # Metrics results for NMAE aggregation (the custom behavior)
        # Note: We must also change the input structure here!
        metrics_results = [(res.num_examples, res.metrics) for _, res in results]

        # 2. Call FedAvg's default loss aggregation (for Raw MAE)
        # Note: We pass the raw loss_results tuple here, which FedAvg expects.
        aggregated_raw_mae_tuple = super().aggregate_evaluate(rnd, results, failures)

        # 3. Call our custom NMAE aggregation (for Normalized Error)
        # aggregated_nmae_dict = self.evaluate_metrics_aggregation_fn(metrics_results)
        # aggregated_nmae = aggregated_raw_mae_tuple[1]

        # 4. Save metrics for printing
        # The raw aggregated MAE is the first element of the tuple returned by super().aggregate_evaluate
        self.last_round_mae = aggregated_raw_mae_tuple[0]
        self.last_round_nmae = aggregated_raw_mae_tuple[1]["NMAE_aggregated"]

        # 5. Return the full dictionary for Flower's history logging
        return self.last_round_mae, {"NMAE_aggregated": self.last_round_nmae}

    def aggregate_fit(
            self,
            rnd: int,
            results: List[Tuple[FitRes, List[np.ndarray]]],
            failures: List[Tuple[ClientManager, Parameters]],
    ) -> Tuple[Optional[Parameters], Dict[str, Scalar]]:
        """Agrega os pesos e salva o modelo na última rodada."""

        # 1. Perform the standard weight aggregation
        # aggregated_parameters, aggregated_fit_metrics = super().aggregate_fit(rnd, results, failures)

        # We need to manually call super().aggregate_fit if we are replacing the return structure.
        # But first, we extract the data for custom logging.

        # --- FIX 1: Correctly extract the FitRes object (fit_res) from the tuple (client, fit_res) ---
        client_fit_metrics = [(fit_res.num_examples, fit_res.metrics) for _, fit_res in results]

        # Re-run the standard aggregation now that we have the raw results
        # NOTE: You MUST call the super method after extracting the data, or you skip aggregation!
        # The structure below ensures correct data extraction for custom logging AND calls the necessary super method.
        aggregated_parameters, aggregated_fit_metrics = super().aggregate_fit(rnd, results, failures)

        # 2. Aggregate and Print Training Metrics

        # This part requires the original FitRes metrics to run the custom aggregation function
        aggregated_train_loss_dict = aggregate_train_loss_metrics(client_fit_metrics)
        aggregated_train_loss = aggregated_train_loss_dict["TrainLoss_aggregated_log"]

        # 3. PRINT ALL METRICS (Fit and Evaluate)
        # We access the evaluation results (MAE/NMAE) saved in the previous step (aggregate_evaluate)

        # NOTE: This assumes aggregate_evaluate ran successfully right before this function.
        # Since Flower runs evaluation before fit in Rnd > 1, this is generally safe.

        # --- CUSTOM PRINTING LOGIC ---
        print("\n" + "=" * 60)
        print(f"| 🚀 ROUND {rnd:02d}/{NUM_ROUNDS} COMPLETED |")
        print("=" * 60)
        print(f"| 📉 TRAINING METRICS (FIT):")
        print(f"|    -> Aggregated Train Loss (L1 Log Scale): {aggregated_train_loss:.4f}")
        print("------------------------------------------------------------")
        print(f"| ✅ EVALUATION METRICS (EVALUATE):")
        # Access metrics saved from aggregate_evaluate
        print(f"|    -> Aggregated MAE (Raw Error): {self.last_round_mae:.4f}")
        print(f"|    -> Aggregated NMAE (Normalized Error): {self.last_round_nmae:.2%}")
        print("=" * 60 + "\n")
        # -----------------------------

        # 4. Salva o modelo após a última rodada
        if aggregated_parameters is not None and rnd == NUM_ROUNDS:
            print("-" * 50)
            print(f"✅ Rodada {rnd}/{NUM_ROUNDS} completa. Salvando o modelo global...")

            # Converte os parâmetros agregados em um modelo PyTorch
            # a. Cria uma instância do modelo com as dimensões corretas
            final_model = WideAndDeepModel(in_features=self.in_features)

            # b. Converte Parameters do Flower para uma lista de arrays NumPy
            aggregated_weights = fl.common.parameters_to_ndarrays(aggregated_parameters)

            # c. Carrega os pesos no modelo PyTorch
            set_weights(final_model, aggregated_weights)

            # d. Salva o estado do modelo
            torch.save(final_model.state_dict(), self.model_path)
            print(f"💾 Modelo global salvo com sucesso em: {self.model_path}")
            print("-" * 50)

        return aggregated_parameters, aggregated_fit_metrics


# --- Funções Auxiliares (mantidas do código anterior) ---

def get_initial_parameters(in_features):
    """Inicializa os pesos do modelo global (baseline)."""
    dummy_y_log_mean = np.array([1.0], dtype=np.float32)
    initial_model = WideAndDeepModel(in_features=in_features)
    initialize_weights(initial_model, dummy_y_log_mean)
    return fl.common.ndarrays_to_parameters(get_weights(initial_model))


def start_server(server_address, initial_data_path, initial_target_path, target_col, model_path, rem_perc):
    """Inicia o servidor Flower e a Federação."""

    # 1. Determina as dimensões de entrada do modelo
    try:
        # Nota: O servidor não roda o fit/transform, apenas precisa da contagem de features
        _, _, in_features, _ = preprocess_data(initial_data_path, initial_target_path, target_col, rem_perc)
    except Exception as e:
        print(f"Erro ao pré-processar dados iniciais para dimensionamento: {e}")
        sys.exit(1)

    initial_parameters = get_initial_parameters(in_features)

    # 2. Usa a NOVA Estratégia para salvar o modelo
    strategy = SaveModelStrategy(
        in_features=in_features,  # Passa o número de features para a nova classe
        model_path=model_path,
        min_available_clients=CLIENTS_REQUIRED,
        min_fit_clients=CLIENTS_REQUIRED,
        min_evaluate_clients=CLIENTS_REQUIRED,
        initial_parameters=initial_parameters,
        on_fit_config_fn=lambda rnd: {"local_epochs": 5, "learning_rate": 1e-4},
        fit_metrics_aggregation_fn=aggregate_train_loss_metrics,
        evaluate_metrics_aggregation_fn=aggregate_metrics,
    )

    # Initialize metric storage variables on the strategy instance
    strategy.last_round_mae = 0.0
    strategy.last_round_nmae = 0.0

    # 3. Inicia o Servidor
    print(f"Servidor Flower iniciado em {server_address}. Aguardando {CLIENTS_REQUIRED} clientes...")
    fl.server.start_server(
        server_address=server_address,
        config=fl.server.ServerConfig(num_rounds=NUM_ROUNDS),
        strategy=strategy,
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Flower Federated Server")
    parser.add_argument("--server", type=str, default="0.0.0.0:8080", help="Endereço IP:Porta para o servidor escutar")
    parser.add_argument("--sample-data", type=str, required=True,
                        help="Arquivo de features de um cliente (para dimensionar o modelo)")
    parser.add_argument("--sample-target", type=str, required=True,
                        help="Arquivo de target de um cliente (para dimensionar o modelo)")
    parser.add_argument("--col", type=str, required=True, help="Nome da coluna Target (ex: cpu_usage)")
    parser.add_argument("--rem-perc", type=float, required=True, help="Valor de percentil a remover")
    parser.add_argument("--model-path", type=str, required=True, help="Path para salvar modelo")
    args = parser.parse_args()

    start_server(args.server, args.sample_data, args.sample_target, args.col, args.model_path, args.rem_perc)