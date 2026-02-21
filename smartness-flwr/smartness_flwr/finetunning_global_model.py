import argparse

import torch
import torch.nn as nn
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
from torch.utils.data import TensorDataset, DataLoader
import os
import sys
import torch.optim as optim

from smartness_flwr.utils import WideAndDeepModel, preprocess_data

# --- Configuration Constants (Must match the training setup) ---
DROPOUT = 0.5


# --- Import Model Architecture from utils.py ---
# For simplicity, we assume you copy the model architecture classes
# (SEBlock, ModernResBlock, WideAndDeepModel) directly into this file
# or ensure utils.py is importable.


# --- Evaluation Function ---
def evaluate_model(model_path, data_path, target_path, target_col, ft_model_path):
    # 1. Check Model File
    if not os.path.exists(model_path):
        print(f"Error: Model file not found at {model_path}")
        sys.exit(1)

    print("--- Starting Global Model Evaluation ---")

    # 2. Load Data
    train_loader, test_loader, in_features, y_raw_mean = preprocess_data(data_path, target_path, target_col)
    print(f"Test data loaded. Total samples: {len(test_loader.dataset)}")

    # 3. Initialize Model and Load Weights
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = WideAndDeepModel(in_features=in_features, dropout=DROPOUT).to(device)

    # Load the state dictionary
    model.load_state_dict(torch.load(model_path, map_location=device))

    # --- NEW: Local Fine-Tuning Step (Personalization) ---
    print("Simulating local personalization (2 epochs of fine-tuning)...")

    # 1. Freeze the feature extractor (deep layers)
    for param in model.deep_input.parameters():
        param.requires_grad = False
    for block in model.blocks:
        for param in block.parameters():
            param.requires_grad = False

    # 2. Train ONLY the Wide layer and the Deep Head (final layers)
    trainable_params = list(model.wide.parameters()) + list(model.deep_head.parameters())

    # We need a NEW optimizer that only sees the trainable parameters
    optimizer_ft = optim.AdamW(trainable_params, lr=1e-4, weight_decay=1e-3)
    criterion_ft = nn.L1Loss()

    model.train()
    for epoch in range(2):  # Fine-tune for some epochs on local data
        for xb, yb in train_loader:
            xb, yb = xb.to(device), yb.to(device)
            optimizer_ft.zero_grad()
            preds = model(xb)
            preds = torch.clamp(preds, min=0.0, max=10.0)
            loss = criterion_ft(preds, yb)
            loss.backward()
            optimizer_ft.step()

    # 3. UNFREEZE the whole model before returning to the server (crucial for next fit round)
    for param in model.parameters():
        param.requires_grad = True

    # --- END Fine-Tuning ---

    model.eval()  # Set model to evaluation mode (crucial for Dropout/LayerNorm)

    # 4. Run Inference
    total_mae_sum = 0.0
    total_samples = 0

    with torch.no_grad():
        for xb, yb_log in test_loader:
            xb = xb.to(device)
            yb_log = yb_log.to(device)

            preds_log = model(xb)

            # Apply safety clamp and inverse transform
            preds_log = torch.clamp(preds_log, min=0.0, max=10.0)
            preds_real = torch.expm1(preds_log)
            yb_real = torch.expm1(yb_log)

            # Calculate MAE
            total_mae_sum += torch.sum(torch.abs(preds_real - yb_real)).item()
            total_samples += xb.size(0)

    # 5. Calculate Final Metrics
    final_mae = total_mae_sum / total_samples
    final_nmae = final_mae / y_raw_mean

    print("\n========================================")
    print("           EVALUATION RESULTS           ")
    print("========================================")
    print(f"Raw MAE (Mean Absolute Error): {final_mae:.4f}")
    print(f"Normalized MAE (NMAE):         {final_nmae:.2%}")
    print(f"Average Target Value (for context): {y_raw_mean:.4f}")
    print("========================================")

    print("-" * 50)
    # d. Salva o estado do modelo
    torch.save(model.state_dict(), ft_model_path)
    print(f"💾 Modelo ft + global salvo com sucesso em: {ft_model_path}")
    print("-" * 50)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Flower Federated Server Evaluator")
    parser.add_argument("--data", type=str, required=True,
                        help="Arquivo de features de um cliente (para dimensionar o modelo)")
    parser.add_argument("--target", type=str, required=True,
                        help="Arquivo de target de um cliente (para dimensionar o modelo)")
    parser.add_argument("--col", type=str, required=True, help="Nome da coluna Target (ex: cpu_usage)")
    parser.add_argument("--model-path", type=str, required=True, help="Path para salvar modelo")
    parser.add_argument("--ft-model-path", type=str, required=True, help="Path para salvar modelo apos fine tunning")
    args = parser.parse_args()

    evaluate_model(args.model_path, args.data, args.target, args.col, args.ft_model_path)