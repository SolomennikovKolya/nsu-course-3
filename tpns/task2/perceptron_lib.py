from sklearn.neural_network import MLPRegressor
from config import ModelConfig


def train_regressor(X_train, y_train, config: ModelConfig):
    """Обучение персептрона для линейной регрессии с помощью sklearn"""
    model = MLPRegressor(
        hidden_layer_sizes=config.layers,
        max_iter=config.epochs,
        learning_rate_init=config.step,
        learning_rate='adaptive',
        solver='adam',
        activation='identity',
        early_stopping=False,
        tol=1e-6,
        random_state=54
    )
    model.fit(X_train, y_train)
    print(f"Модель прошла {model.n_iter_} эпох")
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
+ early_stopping,      # Остановить обучение при отсутствии уменьшения ошибки на tol на валидациионной выборке за n_iter_no_change шагов
+ validation_fraction, # Доля данных для валидации при early stopping
beta_1,              # Экспоненциальное затухание для первого момента (Adam)
beta_2,              # Для второго момента (Adam)
epsilon,             # Малое значение для численной устойчивости
+ n_iter_no_change,    # Число итераций без улучшения до остановки
max_fun              # Макс. число вызовов функции потерь при solver='lbfgs'
+ функция потерь по умолчанию — это среднеквадратичная ошибка (MSE)
"""
