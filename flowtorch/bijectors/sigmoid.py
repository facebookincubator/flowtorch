# Copyright (c) Meta Platforms, Inc

from typing import Optional, Sequence

import torch
import torch.distributions.constraints as constraints
import torch.nn.functional as F
from flowtorch.bijectors.fixed import Fixed
from flowtorch.ops import clipped_sigmoid


class Sigmoid(Fixed):
    codomain = constraints.unit_interval

    def _forward(
        self,
        x: torch.Tensor,
        params: Optional[Sequence[torch.Tensor]]
    ) -> torch.Tensor:
        return clipped_sigmoid(x)

    def _inverse(
        self,
        y: torch.Tensor,
        params: Optional[Sequence[torch.Tensor]]
    ) -> torch.Tensor:
        finfo = torch.finfo(y.dtype)
        y = y.clamp(min=finfo.tiny, max=1.0 - finfo.eps)
        return y.log() - torch.log1p(-y)

    def _log_abs_det_jacobian(
        self,
        x: torch.Tensor,
        y: torch.Tensor,
        params: Optional[Sequence[torch.Tensor]]
    ) -> torch.Tensor:
        return -F.softplus(-x) - F.softplus(x)
