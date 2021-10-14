# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved
# SPDX-License-Identifier: MIT
from typing import Optional

import flowtorch
import torch
import torch.distributions
from flowtorch.bijectors.base import Bijector
from flowtorch.parameters.dense_autoregressive import DenseAutoregressive

"""
This file is a temporary fix so that the `autogen_more_inits` feature branch
works before having merged the `autoregressive_bijector` one.

"""


class Autoregressive(Bijector):
    def __init__(
        self,
        params: Optional[flowtorch.Lazy] = None,
        *,
        shape: torch.Size,
        context_shape: Optional[torch.Size] = None,
    ) -> None:
        super().__init__(params, shape=shape, context_shape=context_shape)

        assert params is not None and issubclass(params.cls, DenseAutoregressive)
