# Copyright (c) Meta Platforms, Inc

# pyre-unsafe

from collections.abc import Sequence
from typing import Optional

import torch
import torch.nn as nn
from flowtorch.parameters.base import Parameters


class Tensor(Parameters):
    def __init__(
        self,
        param_shapes: Sequence[torch.Size],
        input_shape: torch.Size,
        context_shape: torch.Size | None = None,
    ) -> None:
        super().__init__(param_shapes, input_shape, context_shape)

        # TODO: Initialization strategies and constraints!
        self.params = nn.ParameterList(
            [nn.Parameter(torch.randn(shape) * 0.001) for shape in param_shapes]
        )

    def _forward(
        self, x: torch.Tensor | None = None, context: torch.Tensor | None = None
    ) -> Sequence[torch.Tensor] | None:
        return list(self.params)
