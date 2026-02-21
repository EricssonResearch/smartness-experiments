import argparse

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import TensorDataset, DataLoader
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, RobustScaler
from sklearn.decomposition import PCA

D = 101  # number of features
batch_size = 128


def nmae(pred, true, y_train):
    y_mean_abs = np.mean(np.abs(y_train))
    mae = torch.mean(torch.abs(pred - true))
    return mae / y_mean_abs


# --------------------------
# Model: bottleneck MLP with LayerNorm + dropout
# --------------------------
class HighDimMLP(nn.Module):
    def __init__(self, in_features, hidden_units=(1024, 512, 256, 64), dropout=0.2, input_dropout=0.1):
        super().__init__()
        layers = []
        seq_net = nn.Sequential()

        # optional input dropout
        if input_dropout > 0:
            seq_net.add_module("input_dropout", nn.Dropout(input_dropout))
        prev = in_features
        for i, h in enumerate(hidden_units):
            seq_net.add_module(f"linear{i}", nn.Linear(prev, h))
            # LayerNorm per-layer (works well with varied batch distributions)
            seq_net.add_module(f"ln{i}", nn.LayerNorm(h))
            seq_net.add_module(f"relu{i}", nn.ReLU())
            seq_net.add_module(f"drop{i}", nn.Dropout(dropout))
            prev = h

        seq_net.add_module("out", nn.Linear(prev, 1))
        self.net = seq_net  # keep readable naming

    def forward(self, x):
        # Sequential through ModuleDict: call each module in insertion order
        # out = x
        # for name, module in self.net.named_children():
        #     out = module(out)
        # return out
        return self.net(x)


# to tensors & dataloaders
def run(dataset, target_col):
    y = dataset[[target_col]].copy()
    # X = dataset.drop(columns=[target_col])
    X = dataset[['timestamp', 'network_receive_bytes_per_container_7', 'network_receive_bytes_per_container_4',
                 'network_transmit_bytes_per_container_7', 'container_memory_rss_0', 'container_memory_rss_1',
                 'network_transmit_bytes_per_container_6', 'network_receive_bytes_per_container_8',
                 'container_memory_working_set_bytes_4', 'network_receive_bytes_per_container_2',
                 'container_memory_cache_7', 'container_fs_read_seconds_total_13', 'container_memory_rss_5',
                 'container_memory_rss_19', 'container_memory_rss_4', 'container_memory_rss_3',
                 'network_receive_bytes_per_container_0', 'container_memory_rss_2', 'container_memory_mapped_file_2',
                 'container_memory_rss_7', 'container_memory_mapped_file_4', 'container_memory_working_set_bytes_3',
                 'container_memory_cache_4', 'container_memory_total_active_file_bytes_4',
                 'container_memory_mapped_file_0', 'container_memory_usage_bytes_3',
                 'container_memory_working_set_bytes_7', 'container_memory_working_set_bytes_1',
                 'container_memory_usage_bytes_5', 'network_transmit_bytes_per_container_4',
                 'container_memory_working_set_bytes_0', 'container_memory_total_active_file_bytes_1',
                 'container_memory_working_set_bytes_2', 'container_memory_total_active_file_bytes_0',
                 'container_memory_mapped_file_3', 'timestamp', 'container_memory_cache_1',
                 'container_memory_mapped_file_1', 'container_memory_usage_bytes_7', 'container_memory_cache_0',
                 'container_memory_cache_3', 'container_memory_total_active_file_bytes_3', 'container_fs_usage_bytes_3',
                 'container_fs_usage_bytes_0', 'container_memory_mapped_file_7', 'container_memory_usage_bytes_4',
                 'container_memory_total_active_file_bytes_2', 'memory_usage_per_container_3',
                 'network_transmit_bytes_per_container_0', 'network_transmit_bytes_per_container_2',
                 'container_memory_cache_2', 'system_cpu_usage_0', 'container_memory_usage_bytes_1',
                 'system_cpu_usage_1', 'container_cpu_system_seconds_total_2', 'container_memory_usage_bytes_0',
                 'container_memory_failures_total_28', 'network_transmit_bytes_per_container_8',
                 'container_memory_usage_bytes_19', 'user_cpu_usage_3', 'network_receive_bytes_per_container_6',
                 'system_cpu_usage_4', 'memory_usage_per_container_4', 'container_fs_read_seconds_total_14',
                 'container_memory_usage_bytes_2', 'system_cpu_usage_3', 'container_blkio_device_usage_total_1',
                 'user_cpu_usage_1', 'user_cpu_usage_0', 'memory_usage_per_container_2',
                 'container_memory_total_inactive_file_bytes_1', 'container_fs_usage_bytes_1', 'container_memory_rss_6',
                 'container_memory_total_inactive_file_bytes_7', 'user_cpu_usage_2', 'user_cpu_usage_4',
                 'container_memory_total_inactive_file_bytes_0', 'container_fs_usage_bytes_2',
                 'container_memory_total_inactive_file_bytes_4', 'system_cpu_usage_2', 'container_memory_rss_11',
                 'container_memory_usage_bytes_13', 'container_fs_reads_total_40', 'container_memory_rss_13',
                 'system_cpu_usage_10', 'container_memory_usage_bytes_6', 'container_fs_usage_bytes_4',
                 'user_cpu_usage_5', 'memory_usage_per_container_1', 'container_cpu_system_seconds_total_1',
                 'system_cpu_usage_11', 'container_memory_working_set_bytes_5', 'system_cpu_usage_13',
                 'memory_usage_per_container_0', 'container_memory_usage_bytes_18',
                 'container_memory_total_inactive_file_bytes_3', 'system_cpu_usage_18',
                 'container_memory_usage_bytes_11', 'container_memory_rss_8', 'container_memory_rss_17',
                 'container_memory_rss_9']].copy()

    # train/val split + scaling
    X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, random_state=42)
    scaler = RobustScaler()
    X_train = scaler.fit_transform(X_train)
    X_val = scaler.transform(X_val)

    print("Cast to numpy array")
    X_train = X_train.astype(np.float32)
    X_val = X_val.astype(np.float32)
    y_train = y_train.values.astype(np.float32)
    y_val = y_val.values.astype(np.float32)

    # pca = PCA(n_components=400, random_state=42)
    # X_train_p = pca.fit_transform(X_train)
    # X_val_p = pca.transform(X_val)
    X_train_p = X_train
    X_val_p = X_val

    print("Creating TersorDatasets and DataLoaders")
    train_ds = TensorDataset(torch.from_numpy(X_train_p), torch.from_numpy(y_train).unsqueeze(1))
    val_ds = TensorDataset(torch.from_numpy(X_val_p), torch.from_numpy(y_val).unsqueeze(1))
    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True, pin_memory=True)
    val_loader = DataLoader(val_ds, batch_size=batch_size, shuffle=False, pin_memory=True)

    print("Creating Custom MLP")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = HighDimMLP(in_features=D, hidden_units=(512, 256, 128),
                       dropout=0.3, input_dropout=0.1).to(device)

    # --------------------------
    # Loss, optimizer, scheduler
    # --------------------------
    criterion = nn.SmoothL1Loss()
    optimizer = optim.AdamW(model.parameters(), lr=1e-4, weight_decay=1e-4)  # AdamW + weight decay
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=5)

    # --------------------------
    # Train loop with grad clipping & validation
    # --------------------------
    epochs = 60
    clip_norm = 2.0

    for epoch in range(1, epochs + 1):
        model.train()
        train_loss = 0.0
        train_nmae = 0.0  # NEW

        for xb, yb in train_loader:
            xb = xb.to(device)
            yb = yb.to(device)

            preds = model(xb)
            loss = criterion(preds, yb)

            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), clip_norm)
            optimizer.step()

            train_loss += loss.item() * xb.size(0)
            train_nmae += nmae(preds, yb, y_train).item() * xb.size(0)  # NEW

        train_loss /= len(train_loader.dataset)
        train_nmae /= len(train_loader.dataset)  # NEW

        # validation
        model.eval()
        val_loss = 0.0
        val_nmae = 0.0  # NEW

        with torch.no_grad():
            for xb, yb in val_loader:
                xb = xb.to(device)
                yb = yb.to(device)
                preds = model(xb)

                val_loss += criterion(preds, yb).item() * xb.size(0)
                val_nmae += nmae(preds, yb, y_val).item() * xb.size(0)  # NEW
        val_loss /= len(val_loader.dataset)
        val_nmae /= len(val_loader.dataset)

        scheduler.step(val_loss)

        if epoch % 5 == 0 or epoch == 1:
            print(
                f"Epoch {epoch:03d} | "
                f"Train MSE: {train_loss:.4f} | Train NMAE: {train_nmae:.2%} | "
                f"Val MSE: {val_loss:.4f} | Val NMAE: {val_nmae:.2%}"
            )

    # --------------------------
    # Quick inference example
    # --------------------------
    model.eval()
    with torch.no_grad():
        sample = torch.from_numpy(X_val_p[:5]).to(device)
        out = model(sample).cpu()
        print("example preds:", out.numpy().squeeze())
        print("example true:", y_val[:5])
        print(f"NMAE: {nmae(out, y_val[:5], y_val):.2%}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", required=True, help="Path to client's CSV (features)")
    parser.add_argument("--target_data", required=True, help="Path to client's CSV (target)")
    parser.add_argument("--target_col", required=True, help="Path to client's CSV (target)")
    parser.add_argument("--model_name", required=True, help="Model name to save model on disk")
    args = parser.parse_args()

    x_dataset = pd.read_csv(args.data, low_memory=True).apply(pd.to_numeric, errors='coerce').fillna(0)

    y_dataset = pd.read_csv(args.target_data, low_memory=True).apply(pd.to_numeric, errors='coerce').fillna(0)

    y_dataset = y_dataset[["timestamp", args.target_col]].copy()

    df_merged = pd.merge(x_dataset, y_dataset, on="timestamp", how="inner")

    print(f"Min {df_merged[args.target_col].min()} | Max {df_merged[args.target_col].max()}")

    df_filtered = df_merged[df_merged[args.target_col] <= 20.0]

    print(df_filtered.shape)

    run(df_filtered, args.target_col)
