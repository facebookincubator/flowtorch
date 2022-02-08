# Copyright (c) Meta Platforms, Inc

from typing import Optional, Sequence

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
        context_shape: Optional[torch.Size],
    ) -> None:
        super().__init__()
        self.input_shape = input_shape
        self.param_shapes = param_shapes
        self.context_shape = context_shape

    def forward(
        self,
        x: Optional[torch.Tensor] = None,
        context: Optional[torch.Tensor] = None,
    ) -> Optional[Sequence[torch.Tensor]]:
        # TODO: Caching etc.
        return self._forward(x, context)

    def _forward(
        self,
        x: Optional[torch.Tensor] = None,
        context: Optional[torch.Tensor] = None,
    ) -> Optional[Sequence[torch.Tensor]]:
        # I raise an exception rather than using @abstractmethod and
        # metaclass=ABC so that we can reserve the metaclass for lazy
        # evaluation.
        raise NotImplementedError()
