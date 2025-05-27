import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import mean_absolute_error, root_mean_squared_error, r2_score
from tabulate import tabulate


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


def print_regression_metrics_table(y_true: np.ndarray, y_pred: np.ndarray) -> None:
    """Вывод метрик для оценки качества регрессии"""
    mae = mean_absolute_error(y_true, y_pred)
    rmse = root_mean_squared_error(y_true, y_pred)
    r2 = r2_score(y_true, y_pred)

    headers = ["Метрика", "Значение"]
    data = [
        ["MAE",  f"{mae:.2f}"],
        ["RMSE", f"{rmse:.2f}"],
        ["R²",   f"{r2:.4f}"],
    ]
    print(tabulate(data, headers=headers, tablefmt="grid"))
