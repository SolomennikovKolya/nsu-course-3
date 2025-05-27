import numpy as np
import pandas as pd
from typing import List
from sklearn.metrics import mean_squared_error
from config import ModelConfig


class MLPManual:
    def __init__(self, hidden_layer_sizes: List[int] = [], learning_rate_init: float = 0.01,
                 max_iter: int = 200, verbose: bool = False, tol: float = 1e-4, n_iter_no_change: int = 10):
        """
        :param hidden_layer_sizes: Список с числом нейронов в каждом скрытом слое
        :param learning_rate_init: Скорость обучения
        :param max_iter: Количество эпох обучения
        :param verbose: Вывод ошибки на каждой эпохе
        :param tol: Пороговое значение изменения ошибки
        :param n_iter_no_change: Количество итераций для остановки, если улучшение меньше tol
        """
        self.layer_sizes = [0] + hidden_layer_sizes + [1]
        self.learning_rate_init = learning_rate_init
        self.max_iter = max_iter
        self.verbose = verbose
        self.tol = tol
        self.n_iter_no_change = n_iter_no_change
        self.weights = []
        self.biases = []
        self.n_iter_ = 0
        loss_curve_ = []

    def forward(self, X):
        """Прямой обход сети"""
        self.layer_outputs = [X]
        for i in range(len(self.weights)):
            self.layer_outputs.append(
                np.dot(self.layer_outputs[-1], self.weights[i]) + self.biases[i])
        return self.layer_outputs[-1]

    def backward(self, X, y, output):
        """Обратное распространение ошибки."""
        errors = [output - y]

        # Ошибки для скрытых слоёв
        for i in range(len(self.weights)-1, 0, -1):
            errors.append(np.dot(errors[-1], self.weights[i].T))
        errors.reverse()

        # Обновление весов и смещений
        for i in range(len(self.weights)):
            grad = np.dot(self.layer_outputs[i].T, errors[i]) / len(X)
            self.weights[i] -= self.learning_rate_init * grad
            self.biases[i] -= self.learning_rate_init * \
                np.mean(errors[i], axis=0)

    def fit(self, X_pd: pd.DataFrame, y_pd: pd.Series):
        """Обучение модели"""
        # Преобразование входных данных в numpy массивы
        X = X_pd.values
        y = y_pd.values.reshape(-1, 1)

        self.layer_sizes[0] = X.shape[1]

        # Инициализация весов и смещений
        for i in range(len(self.layer_sizes) - 1):
            self.weights.append(np.random.randn(
                self.layer_sizes[i], self.layer_sizes[i+1]) * 0.1)
            self.biases.append(np.zeros(self.layer_sizes[i+1]))

        # best_loss = float('inf')
        # no_improve_count = 0
        self.loss_curve_ = []

        # Прохождение эпох и ранняя остановка если улучшений нет
        for epoch in range(self.max_iter):
            self.n_iter_ = epoch + 1
            output = self.forward(X)
            self.backward(X, y, output)
            loss = mean_squared_error(y, output)
            self.loss_curve_.append(loss)

            if self.verbose:
                print(f"Iteration {epoch}, loss = {loss}")

            # if best_loss - loss > self.tol:
            #     best_loss = loss
            #     no_improve_count = 0
            # else:
            #     no_improve_count += 1

            # if no_improve_count >= self.n_iter_no_change:
            #     if self.verbose:
            #         print(f"Early stopping at iteration {epoch}")
            #     break

    def predict(self, X: pd.DataFrame) -> pd.Series:
        """Предсказание"""
        return self.forward(X.values).squeeze()


def train_regressor(X_train, y_train, config: ModelConfig):
    """Обучение ручной реализации персептрона"""
    model = MLPManual(
        hidden_layer_sizes=config.hidden_layer_sizes,
        learning_rate_init=config.learning_rate_init,
        max_iter=config.max_iter,
        tol=config.tol,
        verbose=config.verbose,
    )
    model.fit(X_train, y_train)
    return model
