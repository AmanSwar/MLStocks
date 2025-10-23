import torch
import torch.nn as nn
import torch.nn.functional as F


def get_model_device(model):
    return next(iter(model.parameters())).device


class RGLRU(nn.Module):
    def __init__(self, hidden_size: int, c: float = 8.0):
        super().__init__()
        self.hidden_size = hidden_size
        self.c = c

        self.input_gate = nn.Linear(hidden_size, hidden_size, bias=False)
        self.recurrence_gate = nn.Linear(hidden_size, hidden_size, bias=False)
        self.a = nn.Parameter(torch.empty(hidden_size))

    def forward(self, x_t: torch.Tensor, state: torch.Tensor) -> torch.Tensor:
        batch_size, hidden_size = x_t.shape
        assert hidden_size == self.hidden_size
        assert state.shape[0] == batch_size

        i_t = torch.sigmoid(self.input_gate(x_t))
        r_t = torch.sigmoid(self.recurrence_gate(x_t))

        # Compute recurrence
        a_t = self.a ** (self.c * r_t)
        multiplier = torch.sqrt(1 - a_t**2)
        new_state = (state * a_t) + (multiplier * (i_t * x_t))

        return new_state

    def init_state(self, batch_size: int, device: torch.device | None = None):
        if device is None:
            device = get_model_device(self)
        return torch.zeros(batch_size, self.hidden_size, device=device)


class CausalConv1d(nn.Module):

    def __init__(self, hidden_size, kernel_size):
        super().__init__()
        self.hidden_size = hidden_size
        self.kernel_size = kernel_size
        self.conv = nn.Conv1d(
            hidden_size, hidden_size, kernel_size, groups=hidden_size, bias=True
        )

    def init_state(self, batch_size: int, device: torch.device | None = None):
        if device is None:
            device = get_model_device(self)
        return torch.zeros(
            batch_size, self.hidden_size, self.kernel_size - 1, device=device
        )

    def forward(self, x: torch.Tensor, state: torch.Tensor):
        x_with_state = torch.concat([state, x[:, :, None]], dim=-1)
        out = self.conv(x_with_state)
        new_state = x_with_state[:, :, 1:]
        return out.squeeze(-1), new_state


class Hawk(nn.Module):
    def __init__(self, hidden_size: int, conv_kernel_size: int = 4):
        super().__init__()

        self.conv_kernel_size = conv_kernel_size
        self.hidden_size = hidden_size

        self.gate_proj = nn.Linear(hidden_size, hidden_size, bias=False)
        self.recurrent_proj = nn.Linear(hidden_size, hidden_size, bias=False)
        self.conv = CausalConv1d(hidden_size, conv_kernel_size)
        self.rglru = RGLRU(hidden_size)
        self.out_proj = nn.Linear(hidden_size, hidden_size, bias=False)

    def forward(
        self, x: torch.Tensor, state: tuple[torch.Tensor, torch.Tensor]
    ) -> tuple[torch.Tensor, list[torch.Tensor]]:
        conv_state, rglru_state = state

        batch_size, hidden_size = x.shape
        
        assert batch_size == conv_state.shape[0] == rglru_state.shape[0]
        assert self.hidden_size == hidden_size == rglru_state.shape[1]

        gate = F.gelu(self.gate_proj(x))
        x = self.recurrent_proj(x)

        x, new_conv_state = self.conv(x, conv_state)
        new_rglru_state = self.rglru(x, rglru_state)

        gated = gate * new_rglru_state
        out = self.out_proj(gated)

        new_state = [new_conv_state, new_rglru_state]
        return out, new_state

    def init_state(
        self, batch_size: int, device: torch.device | None = None
    ) -> list[torch.Tensor]:
        return [
            self.conv.init_state(batch_size, device),
            self.rglru.init_state(batch_size, device),
        ]
