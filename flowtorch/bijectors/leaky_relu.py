# Copyright (c) Meta Platforms, Inc

import math
from typing import Optional, Sequence, Tuple

import torch
import torch.nn.functional as F
from flowtorch.bijectors.fixed import Fixed


class LeakyReLU(Fixed):
    # TODO: Setting the slope of Leaky ReLU as __init__ argument

    def _forward(
        self, x: torch.Tensor, params: Optional[Sequence[torch.Tensor]]
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
        y = F.leaky_relu(x)
        ladj = self._log_abs_det_jacobian(x, y, params)
        return y, ladj

    def _inverse(
        self, y: torch.Tensor, params: Optional[Sequence[torch.Tensor]]
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
        x = F.leaky_relu(y, negative_slope=100.0)
        ladj = self._log_abs_det_jacobian(x, y, params)
        return x, ladj

    def _log_abs_det_jacobian(
        self, x: torch.Tensor, y: torch.Tensor, params: Optional[Sequence[torch.Tensor]]
    ) -> torch.Tensor:
        return torch.where(
            x >= 0.0, torch.zeros_like(x), torch.ones_like(x) * math.log(0.01)
        )
