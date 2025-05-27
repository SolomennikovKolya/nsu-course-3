import torch
import torch.nn as nn


class LibRNN(nn.Module):
    """Библиотечная реализация RNN"""

    def __init__(self, input_size: int, hidden_size: int, num_layers: int):
        super().__init__()

        # Создаём RNN-слои
        # input_size: число признаков на входе на каждом шаге
        # batch_first=True: означает, что входной тензор имеет форму (batch_size, seq_len, input_size) (удобно и привычно)
        self.rnn = nn.RNN(input_size, hidden_size, num_layers=num_layers, batch_first=True)

        # Создаём полносвязный слой
        self.fc = nn.Linear(hidden_size, 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Пропускаем входной батч x через RNN
        # x — входной батч: тензор формы (batch_size, seq_len, input_size)
        # out — это все выходы последнего слоя RNN; out.shape=(batch_size, seq_len, hidden_size)
        # hidden (_) — это последние скрытые состояния всех слоёв; hidden.shape=(num_layers, batch_size, hidden_size)
        out, _ = self.rnn(x)

        # self.fc — это линейный слой: Он принимает на вход вектор размером hidden_size и выдаёт одно число (регрессия)
        # self.fc(out[:, -1, :]) — означает: «Для каждой последовательности взять её последнее скрытое состояние и получить предсказание»
        return self.fc(out[:, -1, :])
