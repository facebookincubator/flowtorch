# Copyright (c) Meta Platforms, Inc

from typing import Optional, Sequence, Tuple

import torch
import torch.distributions.constraints as constraints
from flowtorch.bijectors.fixed import Fixed


class Exp(Fixed):
    r"""
    Elementwise bijector via the mapping :math:`y = \exp(x)`.
    """
    codomain = constraints.positive

    def _forward(
        self, x: torch.Tensor, params: Optional[Sequence[torch.Tensor]]
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
        y = torch.exp(x)
        ladj = self._log_abs_det_jacobian(x, y, params)
        return y, ladj

    def _inverse(
        self, y: torch.Tensor, params: Optional[Sequence[torch.Tensor]]
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
        x = y.log()
        ladj = self._log_abs_det_jacobian(x, y, params)
        return x, ladj

    def _log_abs_det_jacobian(
        self, x: torch.Tensor, y: torch.Tensor, params: Optional[Sequence[torch.Tensor]]
    ) -> torch.Tensor:
        return x
