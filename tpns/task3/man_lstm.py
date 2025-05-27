import torch
import torch.nn as nn


class ManLSTM(nn.Module):
    """Ручная реализация LSTM"""

    def __init__(self, input_size: int, hidden_size: int = 64, num_layers: int = 1):
        super().__init__()
        self.hidden_size = hidden_size
        self.num_layers = num_layers

        # Весовые слои для каждого слоя LSTM
        self.W_i = nn.ModuleList([
            nn.Linear(input_size if i == 0 else hidden_size, hidden_size)
            for i in range(num_layers)
        ])
        self.U_i = nn.ModuleList([
            nn.Linear(hidden_size, hidden_size, bias=False)
            for _ in range(num_layers)
        ])

        self.W_f = nn.ModuleList([
            nn.Linear(input_size if i == 0 else hidden_size, hidden_size)
            for i in range(num_layers)
        ])
        self.U_f = nn.ModuleList([
            nn.Linear(hidden_size, hidden_size, bias=False)
            for _ in range(num_layers)
        ])

        self.W_o = nn.ModuleList([
            nn.Linear(input_size if i == 0 else hidden_size, hidden_size)
            for i in range(num_layers)
        ])
        self.U_o = nn.ModuleList([
            nn.Linear(hidden_size, hidden_size, bias=False)
            for _ in range(num_layers)
        ])

        self.W_c = nn.ModuleList([
            nn.Linear(input_size if i == 0 else hidden_size, hidden_size)
            for i in range(num_layers)
        ])
        self.U_c = nn.ModuleList([
            nn.Linear(hidden_size, hidden_size, bias=False)
            for _ in range(num_layers)
        ])

        self.sigmoid = torch.sigmoid
        self.tanh = torch.tanh
        self.fc = nn.Linear(hidden_size, 1)  # выходной слой регрессии

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        batch_size, seq_len, _ = x.size()

        # Инициализация скрытых и ячеек памяти
        h = [torch.zeros(batch_size, self.hidden_size, device=x.device) for _ in range(self.num_layers)]
        c = [torch.zeros(batch_size, self.hidden_size, device=x.device) for _ in range(self.num_layers)]

        for t in range(seq_len):
            x_t = x[:, t, :]
            for l in range(self.num_layers):
                input_t = x_t if l == 0 else h[l - 1]

                i_t = self.sigmoid(self.W_i[l](input_t) + self.U_i[l](h[l]))
                f_t = self.sigmoid(self.W_f[l](input_t) + self.U_f[l](h[l]))
                o_t = self.sigmoid(self.W_o[l](input_t) + self.U_o[l](h[l]))
                g_t = self.tanh(self.W_c[l](input_t) + self.U_c[l](h[l]))

                c[l] = f_t * c[l] + i_t * g_t
                h[l] = o_t * self.tanh(c[l])

        return self.fc(h[-1])  # регрессионный выход из последнего слоя
