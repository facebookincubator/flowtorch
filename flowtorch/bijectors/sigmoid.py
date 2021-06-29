# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved
# SPDX-License-Identifier: MIT

from typing import Optional

import flowtorch.params
import torch
import torch.distributions.constraints as constraints
import torch.nn.functional as F
from flowtorch.bijectors.base import Bijector
from flowtorch.ops import clipped_sigmoid


class Sigmoid(Bijector):
    codomain = constraints.unit_interval

    def _forward(
        self,
        x: torch.Tensor,
        params: Optional[flowtorch.params.ParamsModule] = None,
        context: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        return clipped_sigmoid(x)

    def _inverse(
        self,
        y: torch.Tensor,
        params: Optional[flowtorch.params.ParamsModule] = None,
        context: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        finfo = torch.finfo(y.dtype)
        y = y.clamp(min=finfo.tiny, max=1.0 - finfo.eps)
        return y.log() - (-y).log1p()

    def _log_abs_det_jacobian(
        self,
        x: torch.Tensor,
        y: torch.Tensor,
        params: Optional[flowtorch.params.ParamsModule] = None,
        context: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        return -F.softplus(-x) - F.softplus(x)
