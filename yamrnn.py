"""PyTorch implementation of YamRNN.

The recurrence follows Equations (4.1)--(4.3) in Yamada, Lin, and Nakamura,
"Developing a Novel Recurrent Neural Network Architecture with Fewer
Parameters and Good Learning Performance".
"""

from __future__ import annotations

import math

import torch
from torch import Tensor, nn
from torch.nn import functional as F

__all__ = ["YamRNN", "compile_yamrnn"]
__version__ = "0.1.0"


class YamRNN(nn.Module):
    """Single-layer, unidirectional YamRNN.

    The interface follows the basic input and output convention of ``nn.GRU``:
    ``forward`` returns ``(output, h_n)``.

    Args:
        input_size: Number of expected features in the input.
        hidden_size: Number of features in the hidden state.
        bias: Add the two biases used by the published recurrence.
        batch_first: Use ``(batch, sequence, feature)`` input and output tensors.
        paper_initialization: Initialize every parameter from ``U(-0.2, 0.2)``
            as in the paper's experiments. Otherwise, use PyTorch-style RNN
            scaling based on ``hidden_size``.
    """

    __constants__ = ["input_size", "hidden_size", "batch_first"]

    def __init__(
        self,
        input_size: int,
        hidden_size: int,
        *,
        bias: bool = True,
        batch_first: bool = False,
        paper_initialization: bool = False,
    ) -> None:
        super().__init__()
        if not isinstance(input_size, int) or isinstance(input_size, bool):
            raise TypeError("input_size must be an integer")
        if input_size <= 0:
            raise ValueError("input_size must be positive")
        if not isinstance(hidden_size, int) or isinstance(hidden_size, bool):
            raise TypeError("hidden_size must be an integer")
        if hidden_size <= 0:
            raise ValueError("hidden_size must be positive")

        self.input_size = input_size
        self.hidden_size = hidden_size
        self.batch_first = batch_first
        self.paper_initialization = paper_initialization

        self.weight_ih = nn.Parameter(torch.empty(2 * hidden_size, input_size))
        self.weight_hh = nn.Parameter(torch.empty(hidden_size, hidden_size))
        if bias:
            self.bias_ih = nn.Parameter(torch.empty(2 * hidden_size))
        else:
            self.register_parameter("bias_ih", None)
        self.reset_parameters()

    def reset_parameters(self) -> None:
        """Reset parameters using the selected initialization."""

        bound = 0.2 if self.paper_initialization else 1.0 / math.sqrt(self.hidden_size)
        for parameter in self.parameters():
            nn.init.uniform_(parameter, -bound, bound)

    def forward(self, x: Tensor, h0: Tensor | None = None) -> tuple[Tensor, Tensor]:
        """Process an input sequence and return ``(output, h_n)``."""

        if x.ndim != 3:
            raise ValueError(f"expected a 3-D input, got shape {tuple(x.shape)}")
        if self.batch_first:
            x = x.transpose(0, 1)

        steps, batch, input_size = x.shape
        if steps == 0:
            raise ValueError("sequence length must be positive")
        if input_size != self.input_size:
            raise ValueError(f"expected input_size={self.input_size}, got {input_size}")

        if h0 is None:
            h = x.new_zeros(batch, self.hidden_size)
        else:
            expected_shape = (1, batch, self.hidden_size)
            if h0.shape != expected_shape:
                raise ValueError(
                    f"h0 must have shape {expected_shape}, got {tuple(h0.shape)}"
                )
            h = h0[0]

        projected = F.linear(x, self.weight_ih, self.bias_ih)
        projected_v1, projected_v2 = projected.chunk(2, dim=-1)
        v2 = torch.tanh(projected_v2)
        constant_term = (1.0 - v2).square()
        v2_squared = v2.square()

        outputs: list[Tensor] = []
        for t in range(steps):
            v1 = torch.sigmoid(projected_v1[t] + F.linear(h, self.weight_hh))
            h = constant_term[t] - (h + v2_squared[t]) * v1
            outputs.append(h)

        output = torch.stack(outputs, dim=0)
        if self.batch_first:
            output = output.transpose(0, 1)
        return output, h.unsqueeze(0)

    def extra_repr(self) -> str:
        return (
            f"input_size={self.input_size}, hidden_size={self.hidden_size}, "
            f"bias={self.bias_ih is not None}, batch_first={self.batch_first}, "
            f"paper_initialization={self.paper_initialization}"
        )


def compile_yamrnn(model: YamRNN) -> nn.Module:
    """Compile YamRNN for repeated use with fixed tensor shapes."""

    if not hasattr(torch, "compile"):
        raise RuntimeError("this PyTorch version does not provide torch.compile")
    return torch.compile(model, fullgraph=True, mode="reduce-overhead")
