# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved
# SPDX-License-Identifier: MIT
from typing import Any, Optional

import flowtorch
import torch
import torch.distributions
from flowtorch.bijectors.base import Bijector
from flowtorch.parameters.tensor import Tensor


class Elementwise(Bijector):
    def __init__(
        self,
        params: Optional[flowtorch.Lazy] = None,
        *,
        shape: torch.Size,
        context_shape: Optional[torch.Size] = None,
        **kwargs: Any
    ) -> None:
        if not params:
            params = Tensor()  # type: ignore

        assert params is None or issubclass(params.cls, Tensor)

        super().__init__(params, shape=shape, context_shape=context_shape)
