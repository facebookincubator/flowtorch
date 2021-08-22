# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved
# SPDX-License-Identifier: MIT
from typing import Optional, Sequence

import torch
from flowtorch import LazyMeta


class Parameters(torch.nn.Module, metaclass=LazyMeta):
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
        self.input_shape = input_shape
        self.param_shapes = param_shapes
        self.context_dims = context_dims

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
