# Copyright (c) Meta Platforms, Inc

import math
from typing import Optional, Sequence, Tuple

import torch
import torch.distributions.constraints as constraints
import torch.nn.functional as F
from flowtorch.bijectors.fixed import Fixed


class Tanh(Fixed):
    r"""
    Transform via the mapping :math:`y = \tanh(x)`.
    """
    codomain = constraints.interval(-1.0, 1.0)

    def _forward(
        self, x: torch.Tensor, params: Optional[Sequence[torch.Tensor]]
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
        y = torch.tanh(x)
        ladj = self._log_abs_det_jacobian(x, y, params)
        return y, ladj

    def _inverse(
        self, y: torch.Tensor, params: Optional[Sequence[torch.Tensor]]
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
        x = torch.atanh(y)
        ladj = self._log_abs_det_jacobian(x, y, params)
        return x, ladj

    def _log_abs_det_jacobian(
        self, x: torch.Tensor, y: torch.Tensor, params: Optional[Sequence[torch.Tensor]]
    ) -> torch.Tensor:
        return 2.0 * (math.log(2.0) - x - F.softplus(-2.0 * x))
