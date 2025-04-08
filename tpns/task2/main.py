import time
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, r2_score
import argparse

from data_preprocessing import get_data
from perceptron_lib import train_regressor as train_regressor_lib


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
    parser = argparse.ArgumentParser()
    parser.add_argument('--mode', type=str, default="lib",
                        help='Способ реализации и обучения модели: "lib" или "manual"')
    # parser.add_argument('mode', help='Тип модели: "lib" или "manual"')
    args = parser.parse_args()

    # Загружаем данные
    data = get_data()
    X = data.drop(columns='quality')
    y = data['quality']
    X_train, X_test = train_test_split(X, test_size=0.2, random_state=42)
    y_train, y_test = train_test_split(y, test_size=0.2, random_state=42)

    # Обучаем персептрон
    start_time = time.time()
    if args.mode == 'lib':
        model = train_regressor_lib(X_train, y_train)
    else:
        model = train_regressor_lib(X_train, y_train)
    elapsed_time = time.time() - start_time

    # Предсказания и оценка
    y_pred = model.predict(X_test)
    mse = mean_squared_error(y_test, y_pred)
    r2 = r2_score(y_test, y_pred)

    # Выводим результаты
    print(f"Время обучения: {elapsed_time:.4f}")
    print(f"MSE: {mse:.4f}")
    print(f"R²: {r2:.4f}")
    show_graphics(y_test, y_pred)


if __name__ == "__main__":
    main()
