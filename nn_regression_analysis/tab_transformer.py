import gc

import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

def nmae(pred, true, y_dt):
    y_mean_abs = np.mean(np.abs(y_dt))
    mae = torch.mean(torch.abs(pred - true))
    return mae / y_mean_abs

# -----------------------
# 1) Synthetic numeric dataset
# -----------------------
x_dataset = pd.read_csv("./datasets/exp90c_11h/t100/prometheus_metrics_wide.csv", low_memory=True).apply(pd.to_numeric, errors='coerce').fillna(0)

y_dataset = pd.read_csv("./datasets/exp90c_11h/t100/20251128_143528564_w.csv", low_memory=True).apply(pd.to_numeric, errors='coerce').fillna(0)

y_dataset = y_dataset[["timestamp", "w_99th_percentile"]].copy()

df_merged = pd.merge(x_dataset, y_dataset, on="timestamp", how="inner")

print(f"Min {df_merged["w_99th_percentile"].min()} | Max {df_merged["w_99th_percentile"].max()}")

df_filtered = df_merged[df_merged["w_99th_percentile"] <= 20.0]

print(df_filtered.shape)

y = df_filtered[["w_99th_percentile"]].copy()
X = df_filtered.drop(columns=["w_99th_percentile"])

# normalize features
scaler = StandardScaler()
X = scaler.fit_transform(X)

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# -----------------------
# 2) Model: numeric tokens -> embedding -> transformer -> pool -> MLP
# -----------------------
class NumericTabTransformer(nn.Module):
    def __init__(self, num_features, emb_dim=16, transformer_layers=2, n_heads=4, mlp_hidden=128, dropout=0.1):
        super().__init__()
        self.num_features = num_features
        self.emb_dim = emb_dim

        # Project each scalar feature to an embedding vector (shared linear layer)
        # Option A: shared projection (learn common projection for all features)
        # ---- Shared projection: one linear layer for all features (1 -> emb_dim)
        # We'll apply it to each scalar feature separately (via unsqueeze & reshape)
        self.shared_proj = nn.Linear(1, emb_dim, bias=True)
        # Option B: feature-specific projection (one linear per feature). We'll use feature-specific.
        # self.feature_projections = nn.ModuleList([
        #     nn.Linear(1, emb_dim) for _ in range(num_features)
        # ])

        # optional small token-type embedding to let transformer know which feature is which
        self.token_type_embeddings = nn.Embedding(num_features, emb_dim)

        # ---- Optional small per-feature bias (learnable) — can help if you want tiny per-feature offsets
        self.feature_bias = nn.Parameter(torch.zeros(num_features, emb_dim))

        # Transformer encoder
        encoder_layer = nn.TransformerEncoderLayer(d_model=emb_dim, nhead=n_heads, batch_first=True, dropout=dropout)
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=transformer_layers)
        self.norm = nn.LayerNorm(emb_dim)

        # MLP head: take pooled transformer output + optionally original global stats
        self.mlp = nn.Sequential(
            nn.Linear(emb_dim, mlp_hidden),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(mlp_hidden, 1)
        )

    def forward(self, x):
        # x shape: (batch, num_features)
        batch = x.size(0)

        # Expand scalar features into shape (batch * num_features, 1)
        x_reshaped = x.contiguous().view(-1, 1)  # (batch*num_features, 1)

        # Shared projection applied to all scalars
        proj = self.shared_proj(x_reshaped)  # (batch*num_features, emb_dim)

        # reshape back to tokens: (batch, num_features, emb_dim)
        tokens = proj.view(batch, self.num_features, self.emb_dim)

        # add token-type embedding
        token_ids = torch.arange(self.num_features, device=x.device).unsqueeze(0).expand(batch, -1)
        tokens = tokens + self.token_type_embeddings(token_ids)

        # add optional feature-specific bias broadcasted across batch
        tokens = tokens + self.feature_bias.unsqueeze(0)  # (1, num_features, emb_dim) -> broadcast

        # Transformer encoder -> (batch, num_features, emb_dim)
        x_trans = self.transformer(tokens)
        x_trans = self.norm(x_trans)

        # Pooling: mean over tokens (alternatives: [CLS] token, max, attention pooling)
        pooled = x_trans.mean(dim=1)  # (batch, emb_dim)

        out = self.mlp(pooled)  # (batch, 1)
        return out

# -----------------------
# 3) Prepare tensors & train
# -----------------------
device = "cuda" if torch.cuda.is_available() else "cpu"
# device = "cpu"

scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

print("Cast to numpy array")
X_train = X_train.astype(np.float32)
X_test = X_test.astype(np.float32)
y_train = y_train.values.astype(np.float32)
y_test = y_test.values.astype(np.float32)

print("Creating tensors")
X_train_t = torch.tensor(X_train, dtype=torch.float32).to(device)
y_train_t = torch.tensor(y_train, dtype=torch.float32).to(device)
X_test_t = torch.tensor(X_test, dtype=torch.float32).to(device)
y_test_t = torch.tensor(y_test, dtype=torch.float32).to(device)

model = NumericTabTransformer(num_features=1814, emb_dim=16, transformer_layers=2, n_heads=4, mlp_hidden=128, dropout=0.1).to(device)
optimizer = optim.Adam(model.parameters(), lr=1e-3, weight_decay=1e-5)
criterion = nn.MSELoss()

epochs = 40
batch_size = 8
n = X_train_t.size(0)

for epoch in range(1, epochs + 1):
    model.train()
    perm = torch.randperm(n)
    epoch_loss = 0.0
    for i in range(0, n, batch_size):
        idx = perm[i:i+batch_size]
        xb = X_train_t[idx]
        yb = y_train_t[idx]

        optimizer.zero_grad()
        preds = model(xb)
        loss = criterion(preds, yb)
        loss.backward()
        optimizer.step()
        epoch_loss += loss.item() * xb.size(0)

    epoch_loss /= n

    if epoch % 5 == 0 or epoch == 1:
        model.eval()
        preds_cpu = []

        with torch.no_grad():
            # free some memory before eval
            torch.cuda.empty_cache()
            gc.collect()

            N = X_test_t.size(0)
            for i in range(0, N, batch_size):
                xb = X_test_t[i: i + batch_size].to(device, non_blocking=True)
                yb = y_test_t[i: i + batch_size].to(device, non_blocking=True)

                # forward
                p = model(xb)  # (bsz, 1) or appropriate shape
                preds_cpu.append(p.detach().cpu())

                # free per-batch GPU objects
                del xb, yb, p
                torch.cuda.empty_cache()

            # concat on CPU
            pred_test = torch.cat(preds_cpu, dim=0)  # shape (N, 1)
            y_cpu = y_test_t.detach().cpu()

            mse = criterion(pred_test, y_cpu).item()
            mae = torch.mean(torch.abs(pred_test - y_cpu)).item()
            nmae_res = nmae(pred_test, y_cpu, y_cpu.numpy())
        print(f"Epoch {epoch:02d} train_loss={epoch_loss:.4f}  test_mse={mse:.4f}  test_mae={mae:.4f}  test_nmae={nmae_res:.2%}")

# Final evaluation
model.eval()
with torch.no_grad():
    preds = model(X_test_t)

    preds = torch.cat(preds, dim=0)  # shape (N, 1)
    y_cpu = y_test_t.detach().cpu()

    mse = criterion(preds, y_cpu).item()
    mae = torch.mean(torch.abs(preds - y_cpu)).item()
    nmae_res = nmae(pred_test, y_cpu, y_cpu.numpy())
print("\nFinal Test MSE: {:.4f}, MAE: {:.4f}, NMAE: {:.2%}".format(mse, mae, nmae_res))
