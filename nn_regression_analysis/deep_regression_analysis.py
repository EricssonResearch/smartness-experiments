import datetime
import matplotlib.pyplot as plt
import numpy as np
import os
import pandas as pd
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader

import logging

logging.getLogger().setLevel(logging.ERROR)

class Data(Dataset):
    '''Dataset Class to store the samples and their corresponding labels,
    and DataLoader wraps an iterable around the Dataset to enable easy access to the samples.
    '''

    def __init__(self, X: np.ndarray, y: np.ndarray) -> None:
        # need to convert float64 to float32 else
        # will get the following error
        # RuntimeError: expected scalar type Double but found Float
        self.X = torch.from_numpy(X.astype(np.float32))
        self.y = torch.from_numpy(y.astype(np.float32))
        self.len = self.X.shape[0]

    def __getitem__(self, index: int) -> tuple:
        return self.X[index], self.y[index]

    def __len__(self) -> int:
        return self.len

class DeepRegressor(nn.Module):
    def __init__(self, input_dim):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, 512),
            nn.BatchNorm1d(512),
            nn.ReLU(),
            nn.Dropout(0.3),

            nn.Linear(512, 256),
            nn.BatchNorm1d(256),
            nn.ReLU(),
            nn.Dropout(0.2),

            nn.Linear(256, 64),
            nn.ReLU(),

            nn.Linear(64, 1)
        )

    def forward(self, x):
        return self.net(x)

# Normalized Mean Absolute Error
# def nmae(y_pred, y_test):
#     mae = mean_absolute_error(y_test, y_pred)
#     mean_true = np.mean(np.abs(y_test))
#     return (mae / mean_true)

# def normalized_mean_absolute_error(y_true, y_pred):
#     """
#     Calculates the Normalized Mean Absolute Error (NMAE).
#
#     Args:
#         y_true (array-like): Ground truth (correct) target values.
#         y_pred (array-like): Estimated target values.
#
#     Returns:
#         float: The Normalized Mean Absolute Error.
#     """
#     y_true = np.asarray(y_true)
#     y_pred = np.asarray(y_pred)
#
#     if len(y_true) != len(y_pred):
#         raise ValueError("y_true and y_pred must have the same length.")
#
#     mae = np.mean(np.abs(y_true - y_pred))
#
#     # Calculate the range of actual values
#     y_range = np.max(y_true) - np.min(y_true)
#
#     # Avoid division by zero if the range is zero
#     if y_range == 0:
#         return 0.0 if mae == 0 else np.inf
#     else:
#         nmae = mae / y_range
#         return nmae

def run(x_ds, y_ds) -> None:
    print("Running...")

    # Normalize the data
    x_scaler = MinMaxScaler()
    x_ds_scaled = x_scaler.fit_transform(x_ds)

    y_scaler = MinMaxScaler()
    y_ds_w_99th_percentile = y_ds[['w_99th_percentile']].copy()
    y_ds_scaled = y_scaler.fit_transform(y_ds_w_99th_percentile)

    # Split the data into training and testing
    X_train, X_test, y_train, y_test = train_test_split(
        x_ds_scaled, y_ds_scaled, test_size=0.30, random_state=42)

    # Generate the training dataset
    traindata = Data(X_train, y_train)
    testdata = Data(X_test, y_test)

    batch_size = 64
    # tells the data loader instance how many sub-processes to use for data loading
    # if the num_worker is zero (default) the GPU has to weight for CPU to load data
    # Theoretically, greater the num_workers,
    # more efficiently the CPU load data and less the GPU has to wait
    num_workers = 2

    # Load the training data into data loader with the
    # respective batch_size and num_workers
    trainloader = DataLoader(traindata, batch_size=batch_size,
                             shuffle=True, num_workers=num_workers)
    testloader = DataLoader(testdata, batch_size=batch_size,
                            shuffle=True, num_workers=num_workers)

    # Detect device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # number of features (len of X cols)
    input_dim = X_train.shape[1]
    # number of hidden layers
    hidden_layers = 50
    # output dimension is 1 because of linear regression
    output_dim = 1
    # initiate the linear regression model
    model = DeepRegressor(input_dim).to(device)
    print(model)

    # criterion to computes the loss between input and target
    criterion = nn.MSELoss()
    # optimizer that will be used to update weights and biases
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)

    # start training
    epochs = 500
    for epoch in range(epochs):
        running_loss = 0.0
        for i, (inputs, labels) in enumerate(trainloader):
            # Move to GPU
            inputs = inputs.to(device)
            labels = labels.to(device)

            # forward propagation
            outputs = model(inputs)
            loss = criterion(outputs, labels)

            # set optimizer to zero grad
            # to remove previous epoch gradients
            optimizer.zero_grad()

            # backward propagation
            loss.backward()

            # optimize
            optimizer.step()
            running_loss += loss.item()

        # display statistics
        if not ((epoch + 1) % (epochs // 10)):
            print(f'Epochs:{epoch + 1:5d} | ' \
                  f'Batches per epoch: {i + 1:3d} | ' \
                  f'Loss: {running_loss / (i + 1):.10f}')

    model.eval()
    # Validate trained model using the test dataset
    with torch.no_grad():
        total_abs_error = 0.0
        total_samples = 0
        total_mean = 0.0

        for i, (inputs, labels) in enumerate(testloader):
            inputs = inputs.to(device)
            labels = labels.to(device)

            # calculate output by running through the network
            predictions = model(inputs)

            # Move to CPU for inverse transform
            y_true = y_scaler.inverse_transform(labels.cpu().numpy())
            y_pred = y_scaler.inverse_transform(predictions.cpu().numpy())

            # Convert back to torch tensors
            y_true = torch.from_numpy(y_true)
            y_pred = torch.from_numpy(y_pred)

            # Compute absolute errors
            abs_error = torch.abs(y_pred - y_true)
            total_abs_error += abs_error.sum().item()
            total_samples += y_true.numel()
            total_mean += y_true.mean().item()

        # Compute mean of true values
        mae = total_abs_error / total_samples
        mean_label = total_mean / len(testloader)
        nmae = mae / mean_label
        print(f"MAE: {mae:.4f} | NMAE: {nmae * 100:.2f}%")

if __name__ == '__main__':
    print("#### EXP 60C 2H T500 ####")
    x_ds_c500 = (pd.read_csv('../datasets/exp60c_2h/t500/prometheus_metrics_wide.csv', low_memory=True)
        .apply(pd.to_numeric, errors='coerce').fillna(0))
    print(x_ds_c500.shape)

    y_ds_c500 = (pd.read_csv('datasets/exp60c_2h/t500/20251109_200426169_w.csv', low_memory=True)
        .apply(pd.to_numeric, errors='coerce').fillna(0))
    print(y_ds_c500.shape)

    run(x_ds_c500, y_ds_c500)
