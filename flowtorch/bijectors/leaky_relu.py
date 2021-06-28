# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved
# SPDX-License-Identifier: MIT

import math
from typing import Optional

import flowtorch.params
import torch
import torch.nn.functional as F
from flowtorch.bijectors.base import Bijector


class LeakyReLU(Bijector):
    # TODO: Setting the slope of Leaky ReLU as __init__ argument

    def _forward(
        self,
        x: torch.Tensor,
        params: Optional[flowtorch.params.ParamsModule] = None,
        context: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        return F.leaky_relu(x)

    def _inverse(
        self,
        y: torch.Tensor,
        params: Optional[flowtorch.params.ParamsModule] = None,
        context: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        return F.leaky_relu(y, negative_slope=100.0)

    def _log_abs_det_jacobian(
        self,
        x: torch.Tensor,
        y: torch.Tensor,
        params: Optional[flowtorch.params.ParamsModule] = None,
        context: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        return torch.where(
            x >= 0.0, torch.zeros_like(x), torch.ones_like(x) * math.log(0.01)
        )
