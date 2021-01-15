# Copyright (c) FlowTorch Development Team. All Rights Reserved
# SPDX-License-Identifier: MIT
from typing import Dict, Optional, Sequence, Tuple

import torch
import torch.nn as nn


class ParamsModule(torch.nn.Module):
    def __init__(
        self,
        params: "Params",
        modules: Optional[nn.ModuleList] = None,
        buffers: Optional[Dict[str, torch.Tensor]] = None,
    ) -> None:
        super(ParamsModule, self).__init__()
        self.params = params
        self.mods = modules

        if buffers is not None:
            for n, v in buffers.items():
                self.register_buffer(n, v)

    def forward(self, x: torch.Tensor) -> Optional[Sequence[torch.Tensor]]:
        return self.params.forward(x, modules=self.mods)


class Params(object):
    """
    Deferred initialization of parameters.
    """

    def __init__(self) -> None:
        super(Params, self).__init__()

    def __call__(
        self,
        input_shape: torch.Size,
        param_shapes: Sequence[torch.Size],
    ) -> ParamsModule:
        return ParamsModule(self, *self.build(input_shape, param_shapes))

    def forward(
        self,
        x: torch.Tensor,
        context: Optional[torch.Tensor] = None,
        modules: Optional[nn.ModuleList] = None,
    ) -> Optional[Sequence[torch.Tensor]]:
        return self._forward(x, context=context, modules=modules)

    def _forward(
        self,
        x: torch.Tensor,
        context: Optional[torch.Tensor] = None,
        modules: Optional[nn.ModuleList] = None,
    ) -> Optional[Sequence[torch.Tensor]]:
        """
        Abstract method to ***
        """
        raise NotImplementedError

    def build(
        self,
        input_shape: torch.Size,
        param_shapes: Sequence[torch.Size],
    ) -> Tuple[nn.ModuleList, Dict[str, torch.Tensor]]:
        self.input_shape = input_shape
        self.param_shapes = param_shapes
        return self._build(input_shape, param_shapes)

    def _build(
        self,
        input_shape: torch.Size,
        param_shapes: Sequence[torch.Size],
    ) -> Tuple[nn.ModuleList, Dict[str, torch.Tensor]]:
        """
        Abstract method to ***
        """
        raise NotImplementedError
