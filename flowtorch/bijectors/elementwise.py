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
        params_fn: Optional[flowtorch.Lazy] = None,
        *,
        shape: torch.Size,
        context_shape: Optional[torch.Size] = None,
        **kwargs: Any
    ) -> None:
        if not params_fn:
            params_fn = Tensor()  # type: ignore

        assert params_fn is None or issubclass(params_fn.cls, Tensor)

        super().__init__(params_fn, shape=shape, context_shape=context_shape)
