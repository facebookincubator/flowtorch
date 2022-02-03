# Copyright (c) Meta Platforms, Inc

from typing import Optional, Sequence, Tuple

import flowtorch.ops
import torch
import torch.distributions.constraints as constraints
import torch.nn.functional as F
from flowtorch.bijectors.fixed import Fixed


class Softplus(Fixed):
    r"""
    Elementwise bijector via the mapping :math:`\text{Softplus}(x) = \log(1 + \exp(x))`.
    """
    codomain = constraints.positive

    def _forward(
        self, x: torch.Tensor, params: Optional[Sequence[torch.Tensor]]
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
        y = F.softplus(x)
        ladj = self._log_abs_det_jacobian(x, y, params)
        return y, ladj

    def _inverse(
        self, y: torch.Tensor, params: Optional[Sequence[torch.Tensor]]
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
        x = flowtorch.ops.softplus_inv(y)
        ladj = self._log_abs_det_jacobian(x, y, params)
        return x, ladj

    def _log_abs_det_jacobian(
        self, x: torch.Tensor, y: torch.Tensor, params: Optional[Sequence[torch.Tensor]]
    ) -> torch.Tensor:
        return -F.softplus(-x)
