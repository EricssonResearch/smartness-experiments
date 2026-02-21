import argparse

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from statsmodels.tsa.arima.model import ARIMA
from pmdarima import auto_arima


def run(dataset, target_col, plot_name):

    # 2. Fit the ARIMA model
    # (p,d,q) parameters depend on your data; (1,1,1) is a common starting point
    # t100: ARIMA(2, 1, 3)
    # t300: ???
    # t500: ARIMA(3, 0, 3)
    model = ARIMA(dataset[target_col], order=(2, 0, 3))
    model_fit = model.fit()

    # 3. Calculate Residuals
    dataset['forecast'] = model_fit.fittedvalues
    dataset['residual'] = dataset[target_col] - dataset['forecast']

    # 4. Define Threshold (Standard Deviation)
    std_dev = dataset['residual'].std()
    threshold = 3 * std_dev

    # 5. Identify and "Clean" Outliers
    dataset['is_outlier'] = dataset['residual'].abs() > threshold

    # Option: Replace outlier with the model's prediction (interpolation)
    dataset['cleaned_value'] = np.where(dataset['is_outlier'], dataset['forecast'], dataset[target_col])

    # 6. Visualize
    plt.figure(figsize=(10,6))
    plt.plot(dataset[target_col], label='Original (with outlier)', color='red', alpha=0.5)
    plt.plot(dataset['cleaned_value'], label='Cleaned (ARIMA adjusted)', color='blue')
    plt.legend()
    plt.savefig(f"./images/arima/{plot_name}.png")

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Flower Federated Client")
    parser.add_argument("--data", type=str, required=True, help="Caminho para o arquivo CSV de features do cliente")
    parser.add_argument("--target", type=str, required=True, help="Caminho para o arquivo CSV do target do cliente")
    parser.add_argument("--col", type=str, required=True, help="Nome da coluna Target (ex: cpu_usage)")
    parser.add_argument("--plot-name", required=True, help="Nome do arquivo do plot")
    args = parser.parse_args()

    x_dataset = pd.read_csv(args.data, low_memory=True).apply(pd.to_numeric, errors='coerce').fillna(0)
    x_dataset = x_dataset.drop(columns=['node_network_speed_bytes_0', 'node_network_speed_bytes_2'])

    y_dataset = pd.read_csv(args.target, low_memory=True).apply(pd.to_numeric, errors='coerce').fillna(0)
    y_dataset = y_dataset[["timestamp", args.col]].copy()
    df_merged = pd.merge(x_dataset, y_dataset, on="timestamp", how="inner")

    # This will automatically test various (p, d, q) combinations
    # stepwise_fit = auto_arima(df_merged[args.col],
    #                           start_p=1, start_q=1,
    #                           max_p=3, max_q=3,
    #                           seasonal=True,  # Set to True if data is seasonal
    #                           trace=True,
    #                           error_action='ignore',
    #                           suppress_warnings=True,
    #                           stepwise=True)
    #
    # print(stepwise_fit.summary())

    run(df_merged, args.col, args.plot_name)