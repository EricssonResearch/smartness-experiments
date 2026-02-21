import argparse

import flwr as fl
import torch
import torch.nn as nn
import torch.optim as optim
import sys
import os

# Importa as classes e funções do seu arquivo utils
from utils import WideAndDeepModel, set_weights, get_weights, initialize_weights, preprocess_data

# --- FedProx Constant ---
MU = 0.05


class FedProxClient(fl.client.NumPyClient):
    def __init__(self, cid, data_path, target_path, target_col, rem_perc):
        self.cid = cid
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        # 1. Load e Preprocess (sem filtro de colinearidade)
        self.trainloader, self.valloader, self.in_features, self.y_val_log_mean = \
            preprocess_data(data_path, target_path, target_col, rem_perc)

        # 2. Inicializa o Modelo
        self.model = WideAndDeepModel(in_features=self.in_features, dropout=0.5).to(self.device)
        self.criterion = nn.L1Loss()

    def get_parameters(self, config):
        return get_weights(self.model)

    def set_parameters(self, parameters):
        set_weights(self.model, parameters)
        # Armazena os pesos globais para o termo proximal
        self.global_weights = [p.clone().detach() for p in self.model.parameters()]

    def fit(self, parameters, config):
        self.set_parameters(parameters)
        num_epochs = config.get("local_epochs", 5)
        lr = config.get("learning_rate", 1e-4)

        optimizer = optim.AdamW(self.model.parameters(), lr=lr, weight_decay=1e-3)
        # --- NEW: Cosine Annealing Scheduler for Local Fit ---
        # T_0 should be set to the total number of local epochs
        scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=num_epochs, eta_min=1e-6)

        # --- NEW: Store Local Training Metrics for Printing ---
        local_fit_history = []

        self.model.train()
        for epoch in range(num_epochs):
            epoch_loss_sum = 0.0

            for xb, yb in self.trainloader:
                xb, yb = xb.to(self.device), yb.to(self.device)

                optimizer.zero_grad()
                preds = self.model(xb)
                preds = torch.clamp(preds, min=0.0, max=10.0)

                # 1. Standard Regression Loss (L1)
                l1_loss = self.criterion(preds, yb)

                # 2. Termo Proximal FedProx
                proximal_term = 0.0
                for local_param, global_param in zip(self.model.parameters(), self.global_weights):
                    proximal_term += torch.norm(local_param - global_param, p=2)

                total_loss = l1_loss + (MU / 2) * proximal_term

                total_loss.backward()
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
                optimizer.step()

                epoch_loss_sum += total_loss.item() * xb.size(0)

            # Step the scheduler at the end of each epoch
            scheduler.step()

            avg_epoch_loss = epoch_loss_sum / len(self.trainloader.dataset)
            local_fit_history.append(avg_epoch_loss)

        # Calculate final metric to return to server
        last_n_losses = local_fit_history[-3:]
        final_loss_metric = sum(last_n_losses) / len(last_n_losses) if last_n_losses else 0.0

        print(f"\n[Client {self.cid}] --- Local Fit Complete ---")
        print(f"| Final Avg. Fit Loss (L1 Log): {final_loss_metric:.4f}")

        return get_weights(self.model), len(self.trainloader.dataset), {"client_train_loss_log": float(final_loss_metric)}

    def evaluate(self, parameters, config):
        self.set_parameters(parameters)

        # # --- NEW: Local Fine-Tuning Step (Personalization) ---
        # print(f"[Client {self.cid}] Starting local fine-tuning (2 epochs)...")
        #
        # # 1. Freeze the feature extractor (deep layers)
        # for param in self.model.deep_input.parameters():
        #     param.requires_grad = False
        # for block in self.model.blocks:
        #     for param in block.parameters():
        #         param.requires_grad = False
        #
        # # 2. Train ONLY the Wide layer and the Deep Head (final layers)
        # trainable_params = list(self.model.wide.parameters()) + list(self.model.deep_head.parameters())
        #
        # # We need a NEW optimizer that only sees the trainable parameters
        # optimizer_ft = optim.AdamW(trainable_params, lr=1e-4, weight_decay=1e-3)
        # criterion_ft = nn.L1Loss()
        #
        # self.model.train()
        # for epoch in range(2):  # Fine-tune for some epochs on local data
        #     for xb, yb in self.trainloader:
        #         xb, yb = xb.to(self.device), yb.to(self.device)
        #         optimizer_ft.zero_grad()
        #         preds = self.model(xb)
        #         preds = torch.clamp(preds, min=0.0, max=10.0)
        #         loss = criterion_ft(preds, yb)
        #         loss.backward()
        #         optimizer_ft.step()
        #
        # # 3. UNFREEZE the whole model before returning to the server (crucial for next fit round)
        # for param in self.model.parameters():
        #     param.requires_grad = True
        #
        # # --- END Fine-Tuning ---

        self.model.eval()
        val_mae_sum = 0.0
        val_samples = 0

        with torch.no_grad():
            for xb, yb in self.valloader:
                xb, yb = xb.to(self.device), yb.to(self.device)
                preds = self.model(xb)
                preds = torch.clamp(preds, min=0.0, max=10.0)

                # Converte para escala real para NMAE
                preds_real = torch.expm1(preds)
                yb_real = torch.expm1(yb)

                val_mae_sum += torch.sum(torch.abs(preds_real - yb_real)).item()
                val_samples += xb.size(0)

        raw_mae = val_mae_sum / val_samples

        # Calculate NMAE (assuming y_val_log_mean holds the denominator's value)
        if hasattr(self, 'y_val_log_mean') and self.y_val_log_mean != 0:
            val_nmae = raw_mae / self.y_val_log_mean
        else:
            val_nmae = 0.0  # Avoid division by zero

        # --- Printing Local Evaluation Metrics ---
        print(f"[Client {self.cid}] --- Local Evaluation (Round End) ---")
        print(f"| Raw MAE (Local Test Set): {raw_mae:.4f}")
        print(f"| Local NMAE: {val_nmae:.2%}")
        print("--------------------------------------\n")

        # Retorna o MAE bruto como métrica principal e NMAE como métrica adicional
        return float(raw_mae), val_samples, {"mae": float(raw_mae), "nmae": float(val_nmae)}


def start_client(cid, server_address, data_file, target_file, target_col, rem_perc):
    """Inicializa e conecta o cliente ao servidor."""

    # Verifica se os arquivos de dados existem antes de iniciar o cliente
    if not os.path.exists(data_file) or not os.path.exists(target_file):
        print(f"Erro: Arquivos de dados não encontrados: {data_file} ou {target_file}")
        sys.exit(1)

    client = FedProxClient(cid, data_file, target_file, target_col, rem_perc)

    # Inicia o cliente sem simulação
    fl.client.start_client(
        server_address=server_address,
        client=client,
    )


if __name__ == "__main__":
    # O cliente deve ser executado no terminal da seguinte forma:
    # python client.py --server "127.0.0.1:8080" --data "client_A_features.csv" --target "client_A_target.csv" --col "cpu_usage"

    parser = argparse.ArgumentParser(description="Flower Federated Client")
    parser.add_argument("--cid", type=str, required=True,
                        help="Id do client")
    parser.add_argument("--server", type=str, required=True,
                        help="Endereço IP:Porta do servidor Flower (ex: 127.0.0.1:8080)")
    parser.add_argument("--data", type=str, required=True, help="Caminho para o arquivo CSV de features do cliente")
    parser.add_argument("--target", type=str, required=True, help="Caminho para o arquivo CSV do target do cliente")
    parser.add_argument("--col", type=str, required=True, help="Nome da coluna Target (ex: cpu_usage)")
    parser.add_argument("--rem-perc", type=float, required=True, help="Valor de percentil a remover")
    args = parser.parse_args()

    start_client(args.cid, args.server, args.data, args.target, args.col, args.rem_perc)