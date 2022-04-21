from typing import Sequence, Optional, Tuple

import torch
from torch import nn, Tensor

from flowtorch.parameters import Parameters


class ZeroConv2d(Parameters):
    autoregressive = False

    def __init__(
        self,
        param_shapes: Sequence[torch.Size],
        input_shape: torch.Size,
        context_shape: Optional[torch.Size],
        *,
        kernel_size: int = 3,
        padding: int = 1
    ) -> None:
        super().__init__(param_shapes, input_shape, context_shape)

        self.kernel_size = kernel_size
        self.channels = self.input_shape[-3] // 2
        self.padding = padding

        self._build()

    def _build(
        self,
    ) -> None:
        self.conv2d = nn.Conv2d(
            self.channels,
            2 * self.channels,
            kernel_size=self.kernel_size,
            padding=self.padding)
        for p in self.conv2d.parameters():
            p.data.zero_()

    def _forward(
        self,
        *input: torch.Tensor,
        inverse: bool,
        context: Optional[torch.Tensor] = None,
    ) -> Tensor:
        x1, x2_or_y2 = input
        return self.conv2d(x1).chunk(2, dim=-3)

    def __repr__(self):
        string = f"{self.__class__.__name__}(conv={self.conv2d})"
        return string