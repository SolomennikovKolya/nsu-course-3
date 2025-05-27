import pandas as pd
from sklearn.neural_network import MLPRegressor
from config import ModelConfig


def train_regressor(X_train: pd.DataFrame, y_train: pd.Series, config: ModelConfig):
    """Обучение персептрона для линейной регрессии с помощью sklearn"""
    model = MLPRegressor(
        hidden_layer_sizes=config.hidden_layer_sizes,
        activation='identity',
        solver=config.solver,
        learning_rate=config.learning_rate,
        learning_rate_init=config.learning_rate_init,
        max_iter=config.max_iter,
        random_state=54,
        tol=1e-6,
        n_iter_no_change=config.max_iter,  # ! чтобы не было ранней остановки
        early_stopping=config.early_stopping,
        verbose=config.verbose,
    )
    model.fit(X_train, y_train)
    return model


"""
Параметры MLPRegressor:
+ hidden_layer_sizes,  # Структура скрытых слоёв
+ activation,          # Функция активации: 'identity', 'logistic', 'tanh', 'relu'
solver,              # Алгоритм оптимизации: 'lbfgs', 'sgd', 'adam'
+ alpha,               # L2-регуляризация - способ предотвратить переобучение, добавляя штраф за слишком большие веса
batch_size,          # Размер мини-батча
+ learning_rate,       # Тип изменения скорости обучения: 'constant', 'invscaling', 'adaptive'
+ learning_rate_init,  # Начальная скорость обучения (начальные размер шага в градиентном спуске)
+ power_t,             # Параметр при 'invscaling'
+ max_iter,            # Максимум итераций обучения
+ shuffle,             # Перемешивание обучающей выборки
+ random_state,        # Фиксированное начальное состояние для воспроизводимости
+ tol,                 # Порог изменения ошибки для остановки
+ verbose,             # Вывод прогресса обучения (для отлаживания)
warm_start,          # Продолжить обучение с предыдущего состояния
momentum,            # Импульс для SGD
nesterovs_momentum,  # Использовать Nesterov momentum
+ early_stopping,      # Включает кросс валидацию. Останавливает обучение при отсутствии уменьшения ошибки на tol на валидациионной выборке за n_iter_no_change шагов
+ validation_fraction, # Доля тренировочной выборки, которая станет валидационной при включенном early stopping
beta_1,              # Экспоненциальное затухание для первого момента (Adam)
beta_2,              # Для второго момента (Adam)
epsilon,             # Малое значение для численной устойчивости
+ n_iter_no_change,    # Число итераций без улучшения до остановки
max_fun              # Макс. число вызовов функции потерь при solver='lbfgs'
+ функция потерь по умолчанию — это среднеквадратичная ошибка (MSE)
"""
