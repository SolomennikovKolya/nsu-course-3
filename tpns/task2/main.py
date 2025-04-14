import time
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from tabulate import tabulate
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.neural_network import MLPRegressor
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.model_selection import GridSearchCV

from data_preprocessing import get_data
from perceptron_lib import train_regressor as train_regressor_lib
from config import ModelConfig


def show_data_distribution(X, y) -> None:
    lda = LinearDiscriminantAnalysis(n_components=2)
    X_lda = lda.fit_transform(X, y)

    plt.figure(figsize=(8, 6))
    plt.scatter(X_lda[:, 0], X_lda[:, 1], c=y, cmap="Spectral", alpha=0.7)
    plt.colorbar(label="Quality")
    plt.title("LDA-проекция")
    plt.show()


def show_graphics(model, y_test, y_pred) -> None:
    """Отрисовка графиков"""
    plt.figure(figsize=(12, 4))

    # График 1: предсказания vs настоящие значения
    plt.subplot(1, 3, 1)
    plt.scatter(y_test, y_pred, alpha=0.5, color=(0, 0.33, 1, 1))
    plt.plot([1, 10], [1, 10], 'r--')
    plt.xticks(np.arange(1, 11, 1))
    plt.yticks(np.arange(1, 11, 1))
    plt.grid(True, alpha=0.3)
    plt.xlabel("Настоящее качество")
    plt.ylabel("Предсказанное качество")
    plt.title("Предсказание vs Реальность")

    # График 2: ошибка
    plt.subplot(1, 3, 2)
    errors = y_test - y_pred
    plt.hist(errors, bins=20, color='salmon', edgecolor='black')
    plt.xlabel("Ошибка")
    plt.ylabel("Частота")
    plt.title("Распределение ошибок")

    # График 3: изменение ошибки
    plt.subplot(1, 3, 3)
    plt.plot(model.loss_curve_)
    plt.xlabel("Эпоха")
    plt.ylabel("Loss (MSE)")
    plt.title("Изменение ошибки по эпохам")
    plt.grid(True)

    plt.tight_layout()
    plt.show()


def find_optimal_params(X_train, y_train) -> None:
    """Режим поиска оптимальных параметров модели через кросс-валидацию"""
    param_grid = {
        'hidden_layer_sizes': [(36, 18), (36, 12), (36, 10), (36, 4), (36, 12, 4)],
        'max_iter': [1000],
        # 'learning_rate': ['constant', 'invscaling', 'adaptive'],
        # 'solver': ['lbfgs', 'adam', 'sgd'],
        # 'early_stopping': [False],
        # 'learning_rate_init': [0.1, 0.001, 0.0001, 0.00001, 0.000001]
    }

    model = MLPRegressor(random_state=54)
    random_search = GridSearchCV(
        model, param_grid=param_grid, cv=3, refit=True)
    random_search.fit(X_train, y_train)

    print("Best parameters:", random_search.best_params_)
    return random_search.best_estimator_


def main() -> None:
    # Парсинг входных параметров
    config = ModelConfig()
    config.parse_args()

    # Загрузка данных
    data = get_data(clean_features=False, shuffle=False,
                    oversample=False, undersample=False)
    X = data.drop(columns='quality')
    y = data['quality']
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=config.test_size, random_state=54)

    # Обучение модели
    start_time = time.time()
    model = None
    if config.mode == 'lib':
        model = train_regressor_lib(X_train, y_train, config)
    elif config.mode == 'manual':
        model = train_regressor_lib(X_train, y_train, config)
    elif config.mode == "find_optimal_params":
        model = find_optimal_params(X_train, y_train)
    elapsed_time = time.time() - start_time

    # Предсказание и оценка
    y_pred = model.predict(X_test)
    mse = mean_squared_error(y_test, y_pred)
    r2 = r2_score(y_test, y_pred)

    # Вывод результатов
    table_data = [(f"{elapsed_time:.4f}", f"{model.n_iter_}",
                   f"{mse:.4f}", f"{r2:.4f}")]
    print(tabulate(table_data, tablefmt="pretty",
                   headers=["Время обучения", "Эпохи", "MSE", "R²"]))
    show_graphics(model, y_test, y_pred)


if __name__ == "__main__":
    main()
