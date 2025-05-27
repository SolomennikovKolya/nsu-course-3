import pandas as pd
import numpy as np
import torch
import torch.nn as nn
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split
from torch.utils.data import DataLoader, TensorDataset
import matplotlib.pyplot as plt
from typing import Tuple


SEQ_LEN: int = 24             # длина окна
BATCH_SIZE: int = 64          # размер батча
HIDDEN_SIZE: int = 64         # размер скрытых слоёв
NUM_LAYERS: int = 3           # кол-во скрытых слоёв
NUM_EPOCHS: int = 10          # кол-во эпох
LEARNING_RATE: float = 0.001  # скорость обучения

DATA_PATH: str = "data/hour.csv"
TARGET_COL: str = 'cnt'
FEATURE_COLS: list[str] = [
    'season', 'yr', 'mnth', 'hr', 'holiday', 'weekday',
    'workingday', 'weathersit', 'temp', 'atemp', 'hum', 'windspeed'
]


class LibRNN(nn.Module):
    """Библиотечная RNN"""

    def __init__(self, input_size: int, hidden_size: int = HIDDEN_SIZE, num_layers: int = NUM_LAYERS):
        super().__init__()

        # Создаём RNN-слои
        # input_size: число признаков на входе на каждом шаге
        # batch_first=True: означает, что входной тензор имеет форму (batch_size, seq_len, input_size) (удобно и привычно)
        self.rnn = nn.RNN(input_size, hidden_size, num_layers=num_layers, batch_first=True)

        # Создаём полносвязный слой
        self.fc = nn.Linear(hidden_size, 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Пропускаем входной батч x через RNN
        # x — входной батч: тензор формы (batch_size, seq_len, input_size)
        # out — это все выходы последнего слоя RNN; out.shape=(batch_size, seq_len, hidden_size)
        # hidden (_) — это последние скрытые состояния всех слоёв; hidden.shape=(num_layers, batch_size, hidden_size)
        out, _ = self.rnn(x)

        # self.fc — это линейный слой: Он принимает на вход вектор размером hidden_size и выдаёт одно число (регрессия)
        # self.fc(out[:, -1, :]) — означает: «Для каждой последовательности взять её последнее скрытое состояние и получить предсказание»
        return self.fc(out[:, -1, :])


def load_and_preprocess_data(url: str) -> Tuple[np.ndarray, np.ndarray, MinMaxScaler, MinMaxScaler]:
    """Загрузка и предобработка данных"""
    df = pd.read_csv(url, sep=",")
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


def get_dataloaders(X_seq: np.ndarray, y_seq: np.ndarray) -> Tuple[DataLoader, DataLoader, np.ndarray, np.ndarray]:
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
    train_dl = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True)
    test_dl = DataLoader(test_ds, batch_size=BATCH_SIZE)
    return train_dl, test_dl, y_test, test_ds.tensors[1].numpy()


def train_model(model: nn.Module, train_dl: DataLoader) -> None:
    """Обучение RNN-модели на тренировочных данных"""
    # Функция потерь и оптимизатор (метод обновления весов)
    loss_fn = nn.MSELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=LEARNING_RATE)

    for epoch in range(NUM_EPOCHS):
        model.train()                 # Включает режим обучения
        for xb, yb in train_dl:       # xb, yb - батчи
            pred = model(xb)          # Предсказание
            loss = loss_fn(pred, yb)  # Вычисление ошибки
            optimizer.zero_grad()     # Обнуление градиентов
            loss.backward()           # Обратное распространение ошибки
            optimizer.step()          # Обновление параметров


def evaluate_model(model: nn.Module, test_dl: DataLoader) -> np.ndarray:
    """Возвращает предсказания обученной модели по тестовой выборке и возвращает всё как 1 NumPy массив"""
    model.eval()  # Переводит модель в режим оценки
    preds = []
    with torch.no_grad():  # Отключение градиентов для оптимизации
        for xb, _ in test_dl:
            preds.append(model(xb).numpy())
    return np.vstack(preds)


def plot_predictions(y_true: np.ndarray, y_pred: np.ndarray, limit: int = 300) -> None:
    """Отрисовка графиков для сравнения предсказанного значения и настоящего"""
    plt.figure(figsize=(12, 5))
    plt.plot(y_true[:limit], label='Истинное значение')
    plt.plot(y_pred[:limit], label='Предсказание')
    plt.legend()
    plt.title("Прогноз количества арендованных велосипедов")
    plt.xlabel("Час")
    plt.ylabel("Количество")
    plt.tight_layout()
    plt.show()


def main() -> None:
    X, y, _, scaler_y = load_and_preprocess_data(DATA_PATH)
    X_seq, y_seq = create_sequences(X, y, SEQ_LEN)
    train_dl, test_dl, y_test_scaled, y_test_orig = get_dataloaders(X_seq, y_seq)

    model = LibRNN(input_size=X.shape[1])
    train_model(model, train_dl)

    preds_scaled = evaluate_model(model, test_dl)
    preds = scaler_y.inverse_transform(preds_scaled)
    y_true = scaler_y.inverse_transform(y_test_orig)

    plot_predictions(y_true, preds)


if __name__ == "__main__":
    main()
