# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved
# SPDX-License-Identifier: MIT
from typing import Dict, Optional, Sequence

import torch
import torch.nn as nn


class ParamsList(torch.nn.Module):
    params_modules: nn.ModuleList

    def __init__(
        self,
        params_modules: Sequence["Params"],
    ) -> None:
        super().__init__()
        self.params_modules = nn.ModuleList(params_modules)

    def forward(
        self, x: torch.Tensor, context: Optional[torch.Tensor] = None
    ) -> Sequence[Optional[Sequence[torch.Tensor]]]:
        # TODO: I believe this is a bug, since we should feed output of previous
        # module into next one
        return [p.forward(x, context=context) for p in self.params_modules]

    def __iter__(self):
        return iter(self.params_modules)

    def __call__(self):
        return self.params_modules

    def __len__(self):
        return len(self.params_modules)

    def __reversed__(self):
        return reversed(self.params_modules)


class Params(torch.nn.Module):
    """
    Deferred initialization of parameters.
    """
    
    def __init__(
        self,
        input_shape: torch.Size,
        param_shapes: Sequence[torch.Size],
        context_dims: int,
    ) -> None:
        super().__init__()


    def forward(
        self,
        x: torch.Tensor,
        context: Optional[torch.Tensor] = None,
    ) -> Sequence[torch.Tensor]:
        # TODO: Caching etc.
        return self._forward(x, context)


    def _forward(
        self,
        x: torch.Tensor,
        context: Optional[torch.Tensor] = None,
    ) -> Sequence[torch.Tensor]:
        # I raise an exception rather than using @abstractmethod and
        # metaclass=ABC so that we can reserve the metaclass for lazy
        # evaluation.
        raise NotImplementedError()
