# Copyright (c) Meta Platforms, Inc

from typing import Optional, Sequence

import torch
import torch.nn as nn
from flowtorch.parameters.base import Parameters


class Tensor(Parameters):
    def __init__(
        self,
        param_shapes: Sequence[torch.Size],
        input_shape: torch.Size,
        context_shape: Optional[torch.Size] = None,
    ) -> None:
        super().__init__(param_shapes, input_shape, context_shape)

        # TODO: Initialization strategies and constraints!
        self.params = nn.ParameterList(
            [nn.Parameter(torch.randn(shape) * 0.001) for shape in param_shapes]
        )

    def _forward(
        self, x: Optional[torch.Tensor] = None, context: Optional[torch.Tensor] = None
    ) -> Optional[Sequence[torch.Tensor]]:
        return list(self.params)
