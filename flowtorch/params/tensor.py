# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved
# SPDX-License-Identifier: MIT

from typing import Any, Dict, Optional, Sequence, Tuple

import torch
import torch.nn as nn
from flowtorch.params.base import Params, ParamsImpl


class ParameterModule(nn.Module):
    # TODO: Better way of coding this, maybe reconsidering _build...
    # Thought 1: Could we do Optional[nn.ModuleList, nn.ParameterList]?
    # Thought 2: Or pass `modules` and `parameters` to `_forward`?
    def __init__(self, shape):
        super().__init__()
        self.parameters = nn.Parameter(torch.randn(shape) * 0.001)


class Tensor(Params):
    # TODO: Initialization strategies and constraints!
    def _build(
        self,
        input_shape: torch.Size,
        param_shapes: Sequence[torch.Size],
        context_dims: int,
    ) -> Tuple["TensorImpl", nn.ModuleList, Dict[str, Any]]:
        layers = [ParameterModule(shape) for shape in param_shapes]
        return (
            TensorImpl(input_shape, param_shapes, context_dims),
            nn.ModuleList(layers),
            {},
        )


class TensorImpl(ParamsImpl):
    def __init__(
        self,
        input_shape: torch.Size,
        param_shapes: Sequence[torch.Size],
        context_dims: int,
    ):
        self.input_shape = input_shape
        self.param_shapes = param_shapes
        self.context_dims = context_dims

    def _forward(
        self,
        x: torch.Tensor,
        context: Optional[torch.Tensor],
        modules: nn.ModuleList,
    ) -> Sequence[torch.Tensor]:
        return [p.parameters for p in modules]
