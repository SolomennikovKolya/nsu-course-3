import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader


def train_model(model: nn.Module, train_dl: DataLoader, learning_rate: float, num_epochs: int) -> None:
    """Обучение RNN-модели на тренировочных данных"""
    # Функция потерь и оптимизатор (метод обновления весов)
    loss_fn = nn.MSELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)

    for epoch in range(num_epochs):
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
