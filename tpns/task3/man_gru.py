import torch
import torch.nn as nn


class ManGRU(nn.Module):
    """Ручная реализация GRU"""

    def __init__(self, input_size: int, hidden_size: int = 64, num_layers: int = 1):
        super().__init__()
        self.hidden_size = hidden_size
        self.num_layers = num_layers

        self.W_z = nn.ModuleList([
            nn.Linear(input_size if i == 0 else hidden_size, hidden_size)
            for i in range(num_layers)
        ])
        self.U_z = nn.ModuleList([
            nn.Linear(hidden_size, hidden_size, bias=False)
            for _ in range(num_layers)
        ])

        self.W_r = nn.ModuleList([
            nn.Linear(input_size if i == 0 else hidden_size, hidden_size)
            for i in range(num_layers)
        ])
        self.U_r = nn.ModuleList([
            nn.Linear(hidden_size, hidden_size, bias=False)
            for _ in range(num_layers)
        ])

        self.W_h = nn.ModuleList([
            nn.Linear(input_size if i == 0 else hidden_size, hidden_size)
            for i in range(num_layers)
        ])
        self.U_h = nn.ModuleList([
            nn.Linear(hidden_size, hidden_size, bias=False)
            for _ in range(num_layers)
        ])

        self.sigmoid = torch.sigmoid
        self.activation = torch.tanh
        self.fc = nn.Linear(hidden_size, 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Вход: x — (batch_size, seq_len, input_size)
        """
        batch_size, seq_len, _ = x.size()
        h = [torch.zeros(batch_size, self.hidden_size, device=x.device) for _ in range(self.num_layers)]

        for t in range(seq_len):
            x_t = x[:, t, :]
            for l in range(self.num_layers):
                input_t = x_t if l == 0 else h[l - 1]

                z = self.sigmoid(self.W_z[l](input_t) + self.U_z[l](h[l]))
                r = self.sigmoid(self.W_r[l](input_t) + self.U_r[l](h[l]))
                h_tilde = self.activation(self.W_h[l](input_t) + self.U_h[l](r * h[l]))
                h[l] = (1 - z) * h[l] + z * h_tilde

        return self.fc(h[-1])  # Предсказание на последн
