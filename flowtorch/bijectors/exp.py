# Copyright (c) Meta Platforms, Inc

from typing import Optional, Sequence

import torch
import torch.distributions.constraints as constraints
from flowtorch.bijectors.fixed import Fixed


class Exp(Fixed):
    r"""
    Elementwise bijector via the mapping :math:`y = \exp(x)`.
    """
    codomain = constraints.positive

    def _forward(
        self,
        x: torch.Tensor,
        params: Optional[Sequence[torch.Tensor]]
    ) -> torch.Tensor:
        return torch.exp(x)

    def _inverse(
        self,
        y: torch.Tensor,
        params: Optional[Sequence[torch.Tensor]]
    ) -> torch.Tensor:
        return y.log()

    def _log_abs_det_jacobian(
        self,
        x: torch.Tensor,
        y: torch.Tensor,
        params: Optional[Sequence[torch.Tensor]]
    ) -> torch.Tensor:
        return x
