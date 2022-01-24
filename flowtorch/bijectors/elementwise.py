# Copyright (c) Meta Platforms, Inc
from typing import Any, Optional

import flowtorch
import torch
import torch.distributions
from flowtorch.bijectors.base import Bijector
from flowtorch.parameters.tensor import Tensor


class Elementwise(Bijector):
    def __init__(
        self,
        hypernet: Optional[flowtorch.Lazy] = None,
        *,
        shape: torch.Size,
        context_shape: Optional[torch.Size] = None,
        **kwargs: Any
    ) -> None:
        if not hypernet:
            hypernet = Tensor()  # type: ignore

        assert hypernet is None or issubclass(hypernet.cls, Tensor)

        super().__init__(hypernet, shape=shape, context_shape=context_shape)
