# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved
# SPDX-License-Identifier: MIT

from typing import Optional

import flowtorch.params
import torch
import torch.distributions.constraints as constraints
import torch.nn.functional as F
from flowtorch.bijectors.base import Bijector
from flowtorch.ops import eps


class ELU(Bijector):
    codomain = constraints.greater_than(-1.0)

    # TODO: Setting the alpha value of ELU as __init__ argument

    def _forward(
        self,
        x: torch.Tensor,
        params: Optional[flowtorch.params.ParamsModule] = None,
        context: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        return F.elu(x)

    def _inverse(
        self,
        y: torch.Tensor,
        params: Optional[flowtorch.params.ParamsModule] = None,
        context: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        return torch.max(y, torch.zeros_like(y)) + torch.min(
            torch.log1p(y + eps), torch.zeros_like(y)
        )

    def _log_abs_det_jacobian(
        self,
        x: torch.Tensor,
        y: torch.Tensor,
        params: Optional[flowtorch.params.ParamsModule] = None,
        context: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        return -F.relu(-x)
