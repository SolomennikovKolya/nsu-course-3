import argparse


class ModelConfig:
    """Мини-класс для хранения основных параметров программы и модели"""

    def __init__(self):
        self.mode: str
        self.test_size: float
        self.hidden_layer_sizes: list[int]
        self.solver: str
        self.learning_rate: str
        self.learning_rate_init: float
        self.max_iter: int
        self.early_stopping: bool
        self.tol: float
        self.verbose: bool

    def parse_args(self):
        parser = argparse.ArgumentParser()

        # Параметры программы
        parser.add_argument('--mode', type=str, default="lib",
                            help='Способ реализации и обучения модели: "lib" или "manual". \
                                При "find_optimal_params" ищутся оптимальные параметры для "lib" реализации персептрона')
        parser.add_argument('--test_size', type=float, default=0.2,
                            help='Доля валидационных сэмплов среди всего датасета: например 0.2')

        # Параметры модели
        parser.add_argument('--hidden_layer_sizes', type=int, nargs='+', default=[36, 10],
                            help='Структура скрытых слоёв: например 20 10')
        parser.add_argument('--solver', type=str, default='adam',
                            help='Алгоритм оптимизации: "lbfgs", "sgd", "adam"')
        parser.add_argument('--max_iter', type=int, default=1000,
                            help='Максимальное количество эпох при обучении: например 1000')
        parser.add_argument('--learning_rate', type=str, default='adaptive',
                            help='Тип изменения скорости обучения: "constant", "invscaling", "adaptive"')
        parser.add_argument('--learning_rate_init', type=float, default=0.1,
                            help='Начальный размер шага в градиентном спуске: например 0.001')
        parser.add_argument('--early_stopping', type=bool, default=False,
                            help='Остановить обучение при отсутствии уменьшения ошибки')
        parser.add_argument('--tol', type=float, default=1e-6,
                            help='Пороговое значение изменения ошибки')
        parser.add_argument('--verbose', type=bool, default=False,
                            help='Вывод ошибки на каждой эпохе по ходу обучения')

        args = parser.parse_args()
        self.mode = args.mode
        self.test_size = args.test_size

        self.learning_rate = args.learning_rate
        self.hidden_layer_sizes = args.hidden_layer_sizes
        self.solver = args.solver
        self.learning_rate_init = args.learning_rate_init
        self.max_iter = args.max_iter
        self.early_stopping = args.early_stopping
        self.tol = args.tol
        self.verbose = args.verbose
