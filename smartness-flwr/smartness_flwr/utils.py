import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import TensorDataset, DataLoader
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from scipy import stats


# --- Componentes do Modelo (GELU, SEBlock, ModernResBlock) ---

class SEBlock(nn.Module):
    # ... (O código das classes SEBlock, ModernResBlock e WideAndDeepModel é idêntico ao último fornecido) ...
    def __init__(self, in_channels, reduction=16):
        super().__init__()
        reduced_dim = max(in_channels // reduction, 4)
        self.fc = nn.Sequential(
            nn.Linear(in_channels, reduced_dim),
            nn.GELU(),
            nn.Linear(reduced_dim, in_channels),
            nn.Sigmoid()
        )

    def forward(self, x):
        scale = self.fc(x)
        return x * scale


class ModernResBlock(nn.Module):
    def __init__(self, hidden_dim, dropout):
        super().__init__()
        self.layer = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout)
        )
        self.se = SEBlock(hidden_dim)

    def forward(self, x):
        out = self.layer(x)
        out = self.se(out)
        return x + out


class WideAndDeepModel(nn.Module):
    def __init__(self, in_features, hidden_dim=512, num_blocks=2, dropout=0.5):
        super().__init__()
        self.deep_input = nn.Sequential(
            nn.Linear(in_features, hidden_dim),
            nn.GELU()
        )
        self.blocks = nn.ModuleList([
            ModernResBlock(hidden_dim, dropout) for _ in range(num_blocks)
        ])
        self.deep_head = nn.Linear(hidden_dim, 1)
        self.wide = nn.Linear(in_features, 1)

    def forward(self, x):
        wide_out = self.wide(x)
        deep = self.deep_input(x)
        for block in self.blocks:
            deep = block(deep)
        deep_out = self.deep_head(deep)
        return deep_out + wide_out


# --- FL Helper Functions (set/get weights) ---

def get_weights(model):
    return [val.cpu().numpy() for _, val in model.state_dict().items()]


def set_weights(model, weights):
    """Sets the model's weights from a list of NumPy arrays (NDArrays)."""

    # 1. Get the current state dictionary keys
    state_dict_keys = list(model.state_dict().keys())

    # 2. Iterate through parameters and convert NumPy array back to Tensor
    for p_name, w_array in zip(state_dict_keys, weights):
        # Find the parameter in the model
        p = model.state_dict()[p_name]

        # CRITICAL FIX: Convert NumPy array (w_array) to a PyTorch Tensor
        # and copy the data.
        p.copy_(torch.from_numpy(w_array))


def initialize_weights(model, train_y_log):
    # ... (Inicialização de pesos idêntica ao último fornecido) ...
    for name, param in model.named_parameters():
        if 'weight' in name and param.dim() > 1:
            nn.init.kaiming_normal_(param, mode='fan_out', nonlinearity='relu')
    mean_target = float(np.mean(train_y_log))
    if hasattr(model, 'wide'):
        model.wide.bias.data.fill_(mean_target)
        model.wide.weight.data.fill_(0.001)
    if hasattr(model, 'deep_head'):
        model.deep_head.bias.data.fill_(0.0)
        model.deep_head.weight.data.fill_(0.001)


def add_time_features(df, target_col):
    # ... (Lags e Rolling Mean idênticos ao último fornecido) ...
    df = df.sort_values(by='timestamp')
    df[f'{target_col}_lag1'] = df[target_col].shift(1)
    df[f'{target_col}_lag2'] = df[target_col].shift(2)
    df[f'{target_col}_lag3'] = df[target_col].shift(3)
    df[f'{target_col}_roll_mean5'] = df[target_col].rolling(window=5).mean().shift(1)
    return df.dropna()


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

def remove_collinear_features(x, threshold=0.95):
    # Calculate the correlation matrix
    corr_matrix = x.corr().abs()

    # Select upper triangle of correlation matrix
    upper = corr_matrix.where(np.triu(np.ones(corr_matrix.shape), k=1).astype(bool))

    # Find features with correlation greater than threshold
    to_drop = [column for column in upper.columns if any(upper[column] > threshold)]

    print(f"Dropping {len(to_drop)} columns due to high correlation (> {threshold})")
    return x.drop(columns=to_drop)


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

# --- Preprocessing de Dados ---
def preprocess_data(data_path, target_path, target_col, rem_perc, remove_collinearity=False):
    """
    Carrega, faz o Feature Engineering e retorna os DataLoaders.
    NÃO USA FILTRO DE COLINEARIDADE AQUI.
    """
    # 1. Load e Merge
    x_dataset = pd.read_csv(data_path, low_memory=True).apply(pd.to_numeric, errors='coerce').fillna(0)
    x_dataset = x_dataset.drop(columns=['node_network_speed_bytes_0', 'node_network_speed_bytes_2'])

    y_dataset = pd.read_csv(target_path, low_memory=True).apply(pd.to_numeric, errors='coerce').fillna(0)
    y_dataset = y_dataset[["timestamp", target_col]].copy()
    df_merged = pd.merge(x_dataset, y_dataset, on="timestamp", how="inner")

    # Removing collinear features...
    # X_clean = remove_collinear_features(df_merged.drop(columns=[target_col]))
    # df_merged = pd.concat([X_clean, df_merged[[target_col]]], axis=1)

    print(f"Dataframe after remove collinear features: {df_merged.shape}")

    df_merged = replace_outliers_zscore(df_merged, target_col)
    # df_merged = remove_outliers(df_merged, target_col, rem_perc)

    # 2. Lag Features
    df_lagged = add_time_features(df_merged, target_col)

    # 3. Separação X e Y (mantendo TODAS as colunas originais + lags)
    X_cols = [col for col in df_lagged.columns if col not in [target_col]]
    X = df_lagged[X_cols].copy()
    y = df_lagged[[target_col]].copy()

    # 4. Target Transform (Log1p)
    y_raw = y.values.astype(np.float32)

    # 5. Split
    X_train, X_val, y_train, y_val = train_test_split(X, y_raw, test_size=0.2, random_state=42)

    # 6. Feature Scaling (Log1p + StandardScaler)
    X_train_np = np.log1p(X_train.values).astype(np.float32)
    X_val_np = np.log1p(X_val.values).astype(np.float32)
    y_train_np = np.log1p(y_train).astype(np.float32)
    y_val_np = np.log1p(y_val).astype(np.float32)

    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train_np)
    X_val_scaled = scaler.transform(X_val_np)

    # 7. DataLoaders
    train_loader = DataLoader(TensorDataset(torch.from_numpy(X_train_scaled), torch.from_numpy(y_train_np)),
                              batch_size=128, shuffle=True)
    val_loader = DataLoader(TensorDataset(torch.from_numpy(X_val_scaled), torch.from_numpy(y_val_np)), batch_size=128,
                            shuffle=False)

    return train_loader, val_loader, X_train_scaled.shape[1], y_val.mean()