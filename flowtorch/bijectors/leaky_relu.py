# Copyright (c) Meta Platforms, Inc

import math
from typing import Optional, Sequence

import torch
import torch.nn.functional as F
from flowtorch.bijectors.fixed import Fixed


class LeakyReLU(Fixed):
    # TODO: Setting the slope of Leaky ReLU as __init__ argument

    def _forward(
        self,
        x: torch.Tensor,
        params: Optional[Sequence[torch.Tensor]]
    ) -> torch.Tensor:
        return F.leaky_relu(x)

    def _inverse(
        self,
        y: torch.Tensor,
        params: Optional[Sequence[torch.Tensor]]
    ) -> torch.Tensor:
        return F.leaky_relu(y, negative_slope=100.0)

    def _log_abs_det_jacobian(
        self,
        x: torch.Tensor,
        y: torch.Tensor,
        params: Optional[Sequence[torch.Tensor]]
    ) -> torch.Tensor:
        return torch.where(
            x >= 0.0, torch.zeros_like(x), torch.ones_like(x) * math.log(0.01)
        )
