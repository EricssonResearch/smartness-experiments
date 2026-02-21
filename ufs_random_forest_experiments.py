import datetime

import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error
from sklearn.feature_selection import SelectKBest, f_regression
from sklearn.tree import DecisionTreeRegressor


def nmae(y_pred, y_test):
    return mean_absolute_error(y_test, y_pred) / y_test.mean()


def training_and_nmae(name_experiment, x_path, y_path, y_column):
    print(f'Experiment: {name_experiment}')

    x = pd.read_csv(x_path, low_memory=True).apply(pd.to_numeric, errors='coerce').fillna(0)
    # print(x.head(5))
    y = pd.read_csv(y_path, low_memory=True).apply(pd.to_numeric, errors='coerce').fillna(0)
    y = y[[y_column]].copy()

    # apply recursive feature elimination:
    time = datetime.datetime.now(tz=datetime.timezone.utc)
    n_features_to_select = 10
    ufs = SelectKBest(f_regression, k=n_features_to_select)
    ufs.fit(x, y)
    ufs_support = ufs.get_support()
    x1 = x.loc[:, ufs_support]
    # print(x1.head(5))
    print(f'Univariate Feature Selection time: {
        (datetime.datetime.now(tz=datetime.timezone.utc) - time).total_seconds()}s')

    x_train, x_test, y_train, y_test = train_test_split(x1, y, test_size=0.33, random_state=None)
    random_forest_model = RandomForestRegressor(n_estimators=120, random_state=None, n_jobs=-1)

    time = datetime.datetime.now(tz=datetime.timezone.utc)
    random_forest_model.fit(x_train, y_train)
    print(f'Rand. Forest Training time: {(datetime.datetime.now(tz=datetime.timezone.utc) - time).total_seconds()}s')

    predicted = random_forest_model.predict(x_test)
    print(f'Rand. Forest NMAE: {nmae(predicted, y_test)}')


if __name__ == '__main__':
    # TABLE6
    training_and_nmae('VoD_Periodic_Single_App',
                      'datasets/VoD-SingleApp-PeriodicLoad/X_port.csv',
                      'datasets/VoD-SingleApp-PeriodicLoad/Y.csv',
                      'DispFrames')
    training_and_nmae('KV_Periodic_Single_App',
                      'datasets/KV-SingleApp-PeriodicLoad/X_port.csv',
                      'datasets/KV-SingleApp-PeriodicLoad/Y.csv',
                      'ReadsAvg')
    training_and_nmae('VoD_Periodic_Both_Apps',
                      'datasets/VoD-BothApps-PeriodicLoad/X_port.csv',
                      'datasets/VoD-BothApps-PeriodicLoad/Y.csv',
                      'DispFrames')
    training_and_nmae('KV_Periodic_Both_Apps',
                      'datasets/KV-BothApps-PeriodicLoad/X_port.csv',
                      'datasets/KV-BothApps-PeriodicLoad/Y.csv',
                      'ReadsAvg')
    training_and_nmae('VoD_Flashcrowd_Single_App',
                      'datasets/VoD-SingleApp-FlashcrowdLoad/X_port.csv',
                      'datasets/VoD-SingleApp-FlashcrowdLoad/Y.csv',
                      'DispFrames')
    training_and_nmae('KV_Flashcrowd_Single_App',
                      'datasets/KV-SingleApp-FlashcrowdLoad/X_port.csv',
                      'datasets/KV-SingleApp-FlashcrowdLoad/Y.csv',
                      'ReadsAvg')
    training_and_nmae('VoD_Flashcrowd_Both_Apps',
                      'datasets/VoD-BothApps-FlashcrowdLoad/X_port.csv',
                      'datasets/VoD-BothApps-FlashcrowdLoad/Y.csv',
                      'DispFrames')
    training_and_nmae('KV_Flashcrowd_Both_Apps',
                      'datasets/KV-BothApps-FlashcrowdLoad/X_port.csv',
                      'datasets/KV-BothApps-FlashcrowdLoad/Y.csv',
                      'ReadsAvg')
