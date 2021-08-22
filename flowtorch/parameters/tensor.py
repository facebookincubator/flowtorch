# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved
# SPDX-License-Identifier: MIT

from typing import Optional, Sequence

import torch
import torch.nn as nn
from flowtorch.parameters.base import Parameters


class Tensor(Parameters):
    def __init__(
        self,
        input_shape: torch.Size,
        param_shapes: Sequence[torch.Size],
        context_dims: int,
    ) -> None:
        super().__init__(input_shape, param_shapes, context_dims)

        # TODO: Initialization strategies and constraints!
        self.params = nn.ParameterList(
            [nn.Parameter(torch.randn(shape) * 0.001) for shape in param_shapes]
        )

    def _forward(
        self, x: torch.Tensor, context: Optional[torch.Tensor] = None
    ) -> Sequence[torch.Tensor]:
        return list(self.params)
