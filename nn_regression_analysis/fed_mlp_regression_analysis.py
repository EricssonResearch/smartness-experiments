import argparse

import numpy as np
import pandas as pd
import time
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import TensorDataset, DataLoader
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, RobustScaler
from scipy import stats

# D = 1814  # number of features
batch_size = 128


def nmae(pred, true, y_train):
    y_mean_abs = np.mean(np.abs(y_train))
    mae = torch.mean(torch.abs(pred - true))
    return mae / y_mean_abs


def remove_collinear_features(x, threshold=0.95):
    # Calculate the correlation matrix
    corr_matrix = x.corr().abs()

    # Select upper triangle of correlation matrix
    upper = corr_matrix.where(np.triu(np.ones(corr_matrix.shape), k=1).astype(bool))

    # Find features with correlation greater than threshold
    to_drop = [column for column in upper.columns if any(upper[column] > threshold)]

    print(f"Dropping {len(to_drop)} columns due to high correlation (> {threshold})")
    return x.drop(columns=to_drop)


def add_time_features(df, target_col):
    # 1. Sort by time to ensure order
    df = df.sort_values(by='timestamp')

    # 2. Create Lag Features (What happened 1, 2, and 3 steps ago?)
    # This gives the model context: "Target was 50.0 last time, so it might be 50.0 again."
    df[f'{target_col}_lag1'] = df[target_col].shift(1)
    df[f'{target_col}_lag2'] = df[target_col].shift(2)
    df[f'{target_col}_lag3'] = df[target_col].shift(3)

    # 3. Create Rolling Features (Trend)
    # Average of last 5 steps
    df[f'{target_col}_roll_mean5'] = df[target_col].rolling(window=5).mean().shift(1)

    # 4. Drop the NaN values created by shifting
    df = df.dropna()

    print(f"Added time features. New shape: {df.shape}")
    return df

def initialize_weights(model, train_y_log):
    # 1. Initialize hidden weights (MLP layers)
    for name, param in model.named_parameters():
        if 'weight' in name and param.dim() > 1:
            nn.init.kaiming_normal_(param, mode='fan_out', nonlinearity='relu')

    mean_target = float(np.mean(train_y_log))

    # 2. WIDE HEAD: Initialize to the Mean
    # The 'wide' part is just a Linear layer. We treat it as the baseline.
    # bias = mean, weight = near zero.
    if hasattr(model, 'wide'):
        model.wide.bias.data.fill_(mean_target)
        model.wide.weight.data.fill_(0.001)

    # 3. DEEP HEAD: Initialize to Zero
    # The deep part should start neutral and only learn the "correction" to the wide part.
    if hasattr(model, 'deep_head'):
        model.deep_head.bias.data.fill_(0.0)
        model.deep_head.weight.data.fill_(0.001)

    print(f"Initialized Wide bias to: {mean_target:.4f} | Deep bias to: 0.0")

def remove_outliers(df, target_col, percentile):
    # REMOVING ROWS WITH TARGET COL GREATER THAN rem_perc PERC.
    if percentile != -1.0:
        percentile_value = np.percentile(df[target_col], percentile, method='linear')

        # Count how many samples in the original data are less than this value
        # Note: The '<' is typically used as percentile is the value *below* which a percentage falls
        count_actual = np.sum(df[target_col] >= percentile_value)

        print(f"The 95th percentile threshold value is: {percentile_value}")
        print(f"Actual count of samples strictly below the threshold: {count_actual}")

        df = df[df[target_col] < percentile_value]
    # END -- REMOVING ROWS WITH TARGET COL GREATER THAN rem_perc PERC.
    return df


def replace_outliers_zscore(df, target_col):
    # 1. Calculate Z-scores
    # We use the Target column defined in your args
    z_scores = np.abs(stats.zscore(df[target_col]))

    # 2. Define threshold (3 is standard, corresponds to ~99.7% of data)
    threshold = 1.96

    # Count how many outliers we found
    outlier_mask = np.abs(z_scores) > threshold
    count_outliers = np.sum(outlier_mask)

    print(f"Z-score threshold: {threshold}")
    print(f"Number of outliers detected: {count_outliers}")

    # 3. Instead of dropping rows, set outliers to NaN
    df.loc[outlier_mask, target_col] = np.nan

    # 4. Fill the gaps using interpolation (preserving the time series structure)
    df[target_col] = df[target_col].interpolate(method='linear')

    # 5. Handle edges (if the first or last row was an outlier)
    df[target_col] = df[target_col].ffill().bfill()

    return df

# --------------------------
# Model: bottleneck MLP with LayerNorm + dropout
# --------------------------
class ResBlock(nn.Module):
    """
    A Residual Block with LayerNorm and Dropout.
    Helps prevents signal loss in deep FL networks.
    """

    def __init__(self, hidden_dim, dropout):
        super().__init__()
        self.layer = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout)
        )

    def forward(self, x):
        # The skip connection: x + layer(x)
        return x + self.layer(x)


class SEBlock(nn.Module):
    """
    Squeeze-and-Excitation Block for Tabular Data.
    Reweights the hidden features dynamically.
    """

    def __init__(self, in_channels, reduction=16):
        super().__init__()
        # We reduce the dimension to force the model to find correlations
        reduced_dim = max(in_channels // reduction, 4)

        self.fc = nn.Sequential(
            nn.Linear(in_channels, reduced_dim),
            nn.GELU(),  # Change 1: ReLU -> GELU
            nn.Linear(reduced_dim, in_channels),
            nn.Sigmoid()
        )

    def forward(self, x):
        # x shape: [batch, in_channels]
        # scale shape: [batch, in_channels]
        scale = self.fc(x)
        return x * scale


class ModernResBlock(nn.Module):
    def __init__(self, hidden_dim, dropout):
        super().__init__()
        self.layer = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.GELU(),  # Change 1: ReLU -> GELU
            nn.Dropout(dropout)
        )
        self.se = SEBlock(hidden_dim)  # Change 2: Add SE Attention

    def forward(self, x):
        # Residual connection + Attention
        out = self.layer(x)
        out = self.se(out)
        return x + out


class TabularModel(nn.Module):
    def __init__(self, in_features, hidden_dim=512, num_blocks=2, dropout=0.2):
        super().__init__()
        # Compression layer: 1814 -> 512
        self.input_layer = nn.Sequential(
            nn.Linear(in_features, hidden_dim),
            nn.ReLU()
        )
        # Residual blocks for deep feature extraction
        self.blocks = nn.ModuleList([
            ResBlock(hidden_dim, dropout) for _ in range(num_blocks)
        ])
        # Output head
        self.output_head = nn.Linear(hidden_dim, 1)

    def forward(self, x):
        x = self.input_layer(x)
        for block in self.blocks:
            x = block(x)
        return self.output_head(x)


class WideAndDeepModel(nn.Module):
    def __init__(self, in_features, hidden_dim=512, num_blocks=2, dropout=0.4):  # Back to 0.4 dropout
        super().__init__()

        # Deep Part Input
        self.deep_input = nn.Sequential(
            nn.Linear(in_features, hidden_dim),
            nn.GELU()  # Change 1: ReLU -> GELU
        )

        # Stacking Modern Blocks
        self.blocks = nn.ModuleList([
            ModernResBlock(hidden_dim, dropout) for _ in range(num_blocks)
        ])

        self.deep_head = nn.Linear(hidden_dim, 1)

        # Wide Part
        self.wide = nn.Linear(in_features, 1)

    def forward(self, x):
        wide_out = self.wide(x)

        deep = self.deep_input(x)
        for block in self.blocks:
            deep = block(deep)
        deep_out = self.deep_head(deep)

        return deep_out + wide_out


# to tensors & dataloaders
def run(dataset, target_col, model_path):


    # 3. Separação X e Y (mantendo TODAS as colunas originais + lags)
    X_cols = [col for col in dataset.columns if col not in [target_col]]
    X = dataset[X_cols].copy()
    y = dataset[[target_col]].copy()

    # 1. Prepare Targets (Log Transform)
    y_raw = y.values.astype(np.float32)
    y_log = np.log1p(y_raw)

    # 2. Split
    X_train, X_val, y_train_log, y_val_log = train_test_split(X, y_log, test_size=0.2, random_state=42)

    # Keep raw validation targets for NMAE calc
    _, _, _, y_val_raw = train_test_split(X, y_raw, test_size=0.2, random_state=42)
    y_val_raw_mean = np.mean(np.abs(y_val_raw))
    if y_val_raw_mean < 1e-6: y_val_raw_mean = 1.0

    # 3. Scale Features
    # --- CHANGED: RobustScaler -> StandardScaler + Log1p on Inputs ---
    # RobustScaler often fails on sparse data (IQR=0).
    # Log1p is better for metrics (bytes/memory) which follow power laws.

    # A. Log transform inputs to squash outliers
    X_train = np.log1p(X_train).astype(np.float32)
    X_val = np.log1p(X_val).astype(np.float32)

    # B. Standard Scale (Safe division)
    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train)
    X_val = scaler.transform(X_val)

    # 4. DataLoaders
    train_ds = TensorDataset(torch.from_numpy(X_train), torch.from_numpy(y_train_log))
    val_ds = TensorDataset(torch.from_numpy(X_val), torch.from_numpy(y_val_log))

    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True, pin_memory=True)
    val_loader = DataLoader(val_ds, batch_size=batch_size, shuffle=False, pin_memory=True)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = WideAndDeepModel(in_features=X_train.shape[1], hidden_dim=512, dropout=0.5).to(device)

    initialize_weights(model, y_train_log)

    # criterion = nn.SmoothL1Loss()
    criterion = nn.L1Loss()
    optimizer = optim.AdamW(model.parameters(), lr=1e-4, weight_decay=1e-3)
    # Change 3: Cosine Annealing with Warm Restarts
    # T_0=10 means it restarts the LR every 10 epochs.
    # T_mult=2 means the next cycle will be 20 epochs, then 40...
    scheduler = optim.lr_scheduler.CosineAnnealingWarmRestarts(optimizer, T_0=10, T_mult=2, eta_min=1e-6)

    epochs = 60

    start_time = time.perf_counter()

    for epoch in range(1, epochs + 1):
        model.train()
        train_loss = 0.0

        for xb, yb in train_loader:
            xb, yb = xb.to(device), yb.to(device)
            preds = model(xb)

            # Clamp output to prevent explosion (0 to ~22,000 in real scale)
            preds = torch.clamp(preds, min=0.0, max=10.0)

            loss = criterion(preds, yb)

            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()

            train_loss += loss.item() * xb.size(0)

        train_loss /= len(train_loader.dataset)

        # Validation
        model.eval()
        val_loss = 0.0
        val_mae_sum = 0.0

        with torch.no_grad():
            for xb, yb in val_loader:
                xb, yb = xb.to(device), yb.to(device)
                preds = model(xb)

                # --- CRITICAL FIX: Clamp Validation too ---
                preds = torch.clamp(preds, min=0.0, max=10.0)
                # ------------------------------------------

                val_loss += criterion(preds, yb).item() * xb.size(0)

                preds_real = torch.expm1(preds)
                yb_real = torch.expm1(yb)
                val_mae_sum += torch.sum(torch.abs(preds_real - yb_real)).item()

        val_loss /= len(val_loader.dataset)
        val_nmae = (val_mae_sum / len(val_loader.dataset)) / y_val_raw_mean

        scheduler.step()

        if epoch % 5 == 0 or epoch == 1:
            print(f"Epoch {epoch:03d} | Train Loss (Log): {train_loss:.4f} | Val NMAE (Real): {val_nmae:.2%}")

    end_time = time.perf_counter()
    elapsed_time = end_time - start_time
    print(f"Training time: {elapsed_time:.4f} seconds")

    print("-" * 50)
    # d. Salva o estado do modelo
    torch.save(model.state_dict(), model_path)
    print(f"💾 Modelo global salvo com sucesso em: {model_path}")
    print("-" * 50)

    return elapsed_time

    # --------------------------
    # Quick inference example
    # --------------------------
    # model.eval()
    # with torch.no_grad():
    #     sample = torch.from_numpy(X_val_p[:5]).to(device)
    #     out = model(sample).cpu()
    #     print("example preds:", out.numpy().squeeze())
    #     print("example true:", y_val[:5])
    #     print(f"NMAE: {nmae(out, y_val[:5], y_val):.2%}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", required=True, help="Path to client's CSV (features)")
    parser.add_argument("--target_data", required=True, help="Path to client's CSV (target)")
    parser.add_argument("--target_col", required=True, help="Path to client's CSV (target)")
    parser.add_argument("--model_path", required=True, help="Model name to save model on disk")
    parser.add_argument("--rounds", type=int, required=True, help="Quantity of rounds to train")
    args = parser.parse_args()

    x_dataset = pd.read_csv(args.data, low_memory=True).apply(pd.to_numeric, errors='coerce').fillna(0)
    x_dataset = x_dataset.drop(columns=['node_network_speed_bytes_0', 'node_network_speed_bytes_2'])
    print(f"Dataframe original shape: {x_dataset.shape}")

    y_dataset = pd.read_csv(args.target_data, low_memory=True).apply(pd.to_numeric, errors='coerce').fillna(0)

    y_dataset = y_dataset[["timestamp", args.target_col]].copy()

    df_merged = pd.merge(x_dataset, y_dataset, on="timestamp", how="inner")

    # Removing collinear features...
    # X_clean = remove_collinear_features(df_merged.drop(columns=[args.target_col]))
    # df_merged = pd.concat([X_clean, df_merged[[args.target_col]]], axis=1)

    print(f"Dataframe after remove collinear features: {df_merged.shape}")

    df_merged = replace_outliers_zscore(df_merged, args.target_col)
    # df_merged = remove_outliers(df_merged, args.target_col, 95)

    print(f"Min {df_merged[args.target_col].min()} | Max {df_merged[args.target_col].max()}")

    zero_count = (df_merged[args.target_col] == 0).sum()
    total_count = len(df_merged)
    mean_val = df_merged[args.target_col].mean()

    print(f"Zeros: {zero_count}/{total_count} ({zero_count / total_count:.2%})")
    print(f"Mean Value: {mean_val:.4f}")

    # --- PLACE THIS IN MAIN ---
    # After merging and before removing collinear features:
    df_merged = add_time_features(df_merged, args.target_col)

    # df_filtered = df_merged[df_merged[args.target_col] <= 20.0]
    df_filtered = df_merged

    print(f"Dataframe filtered shape: {df_filtered.shape}")

    # run(df_filtered, args.target_col, args.model_path)

    if args.rounds > 0:
        training_time_list = []
        for rnd in range(args.rounds):
            print(f"ROUND: {rnd + 1}")
            training_time_list.append(run(df_filtered, args.target_col, f"{args.model_path}_{rnd + 1}.pth"))

        print(f"Average Traning Time: {np.mean(training_time_list):.4f} seconds | Std Dev: {np.std(training_time_list):.4f}")
    else:
        run(df_filtered, args.target_col, args.model_path)


