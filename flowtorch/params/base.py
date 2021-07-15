# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved
# SPDX-License-Identifier: MIT
from abc import ABC, abstractmethod
from typing import Dict, Optional, Sequence, Tuple

import torch
import torch.nn as nn


class ParamsModuleList(torch.nn.Module):
    params_modules: nn.ModuleList

    def __init__(
        self,
        params_modules: Sequence["ParamsModule"],
    ) -> None:
        super().__init__()
        self.params_modules = nn.ModuleList(params_modules)

    def forward(
        self, x: torch.Tensor, context: Optional[torch.Tensor] = None
    ) -> Sequence[Optional[Sequence[torch.Tensor]]]:
        return [p.forward(x, context=context) for p in self.params_modules]

    def __iter__(self):
        return iter(self.params_modules)

    def __call__(self):
        return self.params_modules

    def __len__(self):
        return len(self.params_modules)

    def __reversed__(self):
        return reversed(self.params_modules)


class ParamsModule(torch.nn.Module):
    def __init__(
        self,
        params: "ParamsImpl",
        modules: Optional[nn.ModuleList] = None,
        buffers: Optional[Dict[str, torch.Tensor]] = None,
    ) -> None:
        super().__init__()
        self.params = params
        self.mods = modules

        if buffers is not None:
            for n, v in buffers.items():
                self.register_buffer(n, v)

    def forward(
        self, x: torch.Tensor, context: Optional[torch.Tensor] = None
    ) -> Optional[Sequence[torch.Tensor]]:
        return self.params.forward(x, modules=self.mods, context=context)


class Params(ABC):
    """
    Deferred initialization of parameters.
    """

    def __call__(
        self,
        input_shape: torch.Size,
        param_shapes: Sequence[torch.Size],
        context_dims: int,
    ) -> Optional[ParamsModule]:
        return ParamsModule(*self._build(input_shape, param_shapes, context_dims))

    @abstractmethod
    def _build(
        self,
        input_shape: torch.Size,
        param_shapes: Sequence[torch.Size],
        context_dims: int,
    ) -> Tuple["ParamsImpl", nn.ModuleList, Dict[str, torch.Tensor]]:
        pass


class ParamsImpl(ABC):
    """
    Parameter hypernet for a bijector.
    """

    input_shape: torch.Size
    param_shapes: Sequence[torch.Size]

    def forward(
        self,
        x: torch.Tensor,
        context: Optional[torch.Tensor] = None,
        modules: Optional[nn.ModuleList] = None,
    ) -> Optional[Sequence[torch.Tensor]]:
        if modules is None:
            modules = nn.ModuleList()
        return self._forward(x, context=context, modules=modules)

    @abstractmethod
    def _forward(
        self,
        x: torch.Tensor,
        context: Optional[torch.Tensor],
        modules: nn.ModuleList,
    ) -> Sequence[torch.Tensor]:
        pass
