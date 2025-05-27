import pandas as pd
import numpy as np
import torch
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split
from torch.utils.data import DataLoader, TensorDataset
from typing import Tuple


DATA_PATH: str = "data/hour.csv"
TARGET_COL: str = 'cnt'
FEATURE_COLS: list[str] = [
    'season', 'yr', 'mnth', 'hr', 'holiday', 'weekday',
    'workingday', 'weathersit', 'temp', 'atemp', 'hum', 'windspeed'
]


def load_and_preprocess_data() -> Tuple[np.ndarray, np.ndarray, MinMaxScaler, MinMaxScaler]:
    """Загрузка и предобработка данных"""
    df = pd.read_csv(DATA_PATH, sep=",")
    scaler_X = MinMaxScaler()
    scaler_y = MinMaxScaler()
    X = scaler_X.fit_transform(df[FEATURE_COLS])
    y = scaler_y.fit_transform(df[[TARGET_COL]])
    return X, y, scaler_X, scaler_y


def create_sequences(X: np.ndarray, y: np.ndarray, seq_len: int) -> Tuple[np.ndarray, np.ndarray]:
    """Получение всех возможных последовательностей (окон) длины seq_len"""
    X_seq, y_seq = [], []
    for i in range(len(X) - seq_len):
        X_seq.append(X[i:i+seq_len])
        y_seq.append(y[i+seq_len])
    return np.array(X_seq), np.array(y_seq)


def get_dataloaders(X_seq: np.ndarray, y_seq: np.ndarray, batch_size: int) -> Tuple[DataLoader, DataLoader, np.ndarray, np.ndarray]:
    """
    Подготовление данных для обучения и тестирования RNN-модели

    Parameters
    ----------
    X_seq : np.ndarray of shape (num_samples, seq_len, num_features) - массив входных последовательностей
    y_seq : np.ndarray of shape (num_samples, 1) - соответствующие цели

    Returns
    -------
    train_dl : DataLoader - DataLoader обучающей выборки для обучении модели
    test_dl : DataLoader - DataLoader тестовой выборки для обучения модели
    y_test : np.ndarray - целевые значения для оценки
    test_ds.tensors[1].numpy() : np.ndarray - то же самое, но в виде numpy массива (для визуализации после обратного масштабирования)
    """

    X_train, X_test, y_train, y_test = train_test_split(X_seq, y_seq, test_size=0.2, shuffle=False)

    # TensorDataset - простой контейнер: на вход подаёшь пары (X, y), и он будет по очереди возвращать их при итерации.
    train_ds = TensorDataset(torch.tensor(X_train, dtype=torch.float32),
                             torch.tensor(y_train, dtype=torch.float32))
    test_ds = TensorDataset(torch.tensor(X_test, dtype=torch.float32),
                            torch.tensor(y_test, dtype=torch.float32))

    # DataLoader - класс для разбиения данных на батчи и эффективной подачи их в модель при обучении
    # train_dl - батчит обучающие данные и перемешивает их (shuffle=True)
    # test_dl - только разбивает на батчи, но не перемешивает (важно для последовательной оценки)
    train_dl = DataLoader(train_ds, batch_size=batch_size, shuffle=True)
    test_dl = DataLoader(test_ds, batch_size=batch_size)
    return train_dl, test_dl, y_test, test_ds.tensors[1].numpy()
