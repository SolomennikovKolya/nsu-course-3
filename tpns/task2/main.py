import time
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from tabulate import tabulate
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.neural_network import MLPRegressor
from imblearn.over_sampling import SMOTE

from data_preprocessing import get_data
from perceptron_lib import train_regressor as train_regressor_lib
from config import ModelConfig


def show_graphics(y_test, y_pred) -> None:
    """Отрисовка графиков"""
    plt.figure(figsize=(12, 6))

    # График 1: предсказания vs настоящие значения
    plt.subplot(1, 2, 1)
    plt.scatter(y_test, y_pred, alpha=0.5, color=(0, 0.33, 1, 1))
    plt.plot([1, 10], [1, 10], 'r--')
    plt.xticks(np.arange(1, 11, 1))
    plt.yticks(np.arange(1, 11, 1))
    plt.grid(True, alpha=0.3)
    plt.xlabel("Настоящее качество")
    plt.ylabel("Предсказанное качество")
    plt.title("Предсказание vs Реальность")

    # График 2: ошибка
    plt.subplot(1, 2, 2)
    errors = y_test - y_pred
    plt.hist(errors, bins=20, color='salmon', edgecolor='black')
    plt.xlabel("Ошибка")
    plt.ylabel("Частота")
    plt.title("Распределение ошибок")

    plt.tight_layout()
    plt.show()


def main() -> None:
    # Парсим входные параметры
    config = ModelConfig()
    config.parse_args()

    # Загружаем данные
    data = get_data(shuffle=False)
    X = data.drop(columns='quality')
    y = data['quality']
    X_train, X_test = train_test_split(X, test_size=0.2, random_state=54)
    y_train, y_test = train_test_split(y, test_size=0.2, random_state=54)

    smote = SMOTE(sampling_strategy='auto', random_state=42, k_neighbors=2)
    X_res, y_res = smote.fit_resample(X_train, y_train)

    # Обучаем модель
    start_time = time.time()
    model = train_regressor_lib(X_train, y_train, config) if config.mode == 'manual' \
        else train_regressor_lib(X_train, y_train, config)
    elapsed_time = time.time() - start_time

    # Предсказания и оценка
    y_pred = model.predict(X_test)
    mse = mean_squared_error(y_test, y_pred)
    r2 = r2_score(y_test, y_pred)

    # Выводим результаты
    table_data = [(f"{elapsed_time:.4f}", f"{mse:.4f}", f"{r2:.4f}")]
    print(tabulate(table_data, headers=[
          "Время обучения", "MSE", "R²"], tablefmt="pretty"))
    show_graphics(y_test, y_pred)


if __name__ == "__main__":
    main()
