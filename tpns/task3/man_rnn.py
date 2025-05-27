import torch
import torch.nn as nn


class ManRNN(nn.Module):
    """Ручная реализация RNN"""

    def __init__(self, input_size: int, hidden_size: int = 64, num_layers: int = 1):
        super().__init__()
        self.hidden_size = hidden_size
        self.num_layers = num_layers

        # Создаём список полносвязных слоёв W_xh и W_hh для каждого слоя
        self.W_xh = nn.ModuleList([
            nn.Linear(input_size if i == 0 else hidden_size, hidden_size)
            for i in range(num_layers)
        ])
        self.W_hh = nn.ModuleList([
            nn.Linear(hidden_size, hidden_size, bias=False)
            for _ in range(num_layers)
        ])

        self.activation = torch.tanh
        self.fc = nn.Linear(hidden_size, 1)  # Последний выходной слой

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        x: (batch_size, seq_len, input_size)
        """
        batch_size, seq_len, _ = x.size()

        # Инициализируем скрытые состояния нулями: (num_layers, batch_size, hidden_size)
        h = [torch.zeros(batch_size, self.hidden_size, device=x.device) for _ in range(self.num_layers)]

        # Проходим по временным шагам
        for t in range(seq_len):
            x_t = x[:, t, :]  # (batch_size, input_size)
            for layer in range(self.num_layers):
                # Вход для текущего слоя: либо из x, либо из предыдущего слоя
                input_t = x_t if layer == 0 else h[layer - 1]
                h[layer] = self.activation(
                    self.W_xh[layer](input_t) + self.W_hh[layer](h[layer])
                )

        # Используем последнее скрытое состояние верхнего слоя
        return self.fc(h[-1])
