# Copyright (c) Meta Platforms, Inc

from typing import Optional, Sequence, Tuple

import torch
import torch.distributions.constraints as constraints
import torch.nn.functional as F
from flowtorch.bijectors.fixed import Fixed
from flowtorch.ops import clipped_sigmoid


class Sigmoid(Fixed):
    codomain = constraints.unit_interval

    def _forward(
        self, x: torch.Tensor, params: Optional[Sequence[torch.Tensor]]
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
        y = clipped_sigmoid(x)
        ladj = self._log_abs_det_jacobian(x, y, params)
        return y, ladj

    def _inverse(
        self, y: torch.Tensor, params: Optional[Sequence[torch.Tensor]]
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
        finfo = torch.finfo(y.dtype)
        y = y.clamp(min=finfo.tiny, max=1.0 - finfo.eps)
        x = y.log() - torch.log1p(-y)
        ladj = self._log_abs_det_jacobian(x, y, params)
        return x, ladj

    def _log_abs_det_jacobian(
        self, x: torch.Tensor, y: torch.Tensor, params: Optional[Sequence[torch.Tensor]]
    ) -> torch.Tensor:
        return -F.softplus(-x) - F.softplus(x)
