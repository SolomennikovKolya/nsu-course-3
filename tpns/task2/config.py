import argparse


class ModelConfig:
    """Мини-класс для хранения основных параметров модели"""

    def __init__(self):
        self.mode: str = None
        self.layers: list[int] = None
        self.epochs: int = None
        self.step: float = None

    def parse_args(self):
        parser = argparse.ArgumentParser()
        parser.add_argument('--mode', type=str, default="lib",
                            help='Способ реализации и обучения модели: "lib" или "manual"')
        parser.add_argument('--layers', type=int, nargs='+',
                            default=[], help='Структура скрытых слоёв, например: 100 50')
        parser.add_argument('--epochs', type=int, default=1000,
                            help='Максимальное количество эпох при обучении')
        parser.add_argument('--step', type=float, default=0.001,
                            help='Начальный размер шага в градиентном спуске')

        args = parser.parse_args()
        self.mode = args.mode
        self.layers = args.layers
        self.epochs = args.epochs
        self.step = args.step

    def __str__(self):
        return f"ModelConfig(mode={self.mode}, layers={self.layers}, epochs={self.epochs}, step={self.step})"
