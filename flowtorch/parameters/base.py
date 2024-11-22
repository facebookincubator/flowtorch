# Copyright (c) Meta Platforms, Inc

# pyre-unsafe

from collections.abc import Sequence
from typing import Optional

import torch
from flowtorch import LazyMeta


class Parameters(torch.nn.Module, metaclass=LazyMeta):
    """
    Deferred initialization of parameters.
    """

    def __init__(
        self,
        param_shapes: Sequence[torch.Size],
        input_shape: torch.Size,
        context_shape: torch.Size | None,
    ) -> None:
        super().__init__()
        self.input_shape = input_shape
        self.param_shapes = param_shapes
        self.context_shape = context_shape

    def forward(
        self,
        x: torch.Tensor | None = None,
        context: torch.Tensor | None = None,
    ) -> Sequence[torch.Tensor] | None:
        # TODO: Caching etc.
        return self._forward(x, context)

    def _forward(
        self,
        x: torch.Tensor | None = None,
        context: torch.Tensor | None = None,
    ) -> Sequence[torch.Tensor] | None:
        # I raise an exception rather than using @abstractmethod and
        # metaclass=ABC so that we can reserve the metaclass for lazy
        # evaluation.
        raise NotImplementedError()
