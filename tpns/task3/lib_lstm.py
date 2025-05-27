import torch
import torch.nn as nn


class LibLSTM(nn.Module):
    """Библиотечная реализация LSTM"""

    def __init__(self, input_size: int, hidden_size: int = 64, num_layers: int = 1):
        super().__init__()
        self.lstm = nn.LSTM(
            input_size=input_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True
        )
        self.fc = nn.Linear(hidden_size, 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        out, _ = self.lstm(x)     # out: (batch_size, seq_len, hidden_size)
        last_out = out[:, -1, :]  # берём скрытое состояние последнего шага
        return self.fc(last_out)  # прогноз на основе последнего шага
