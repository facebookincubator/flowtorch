# Copyright (c) Meta Platforms, Inc

from typing import Optional, Sequence, Tuple

import flowtorch
import torch
import torch.distributions.constraints as constraints
from flowtorch.bijectors.fixed import Fixed


class Power(Fixed):
    r"""
    Elementwise bijector via the mapping :math:`y = x^{\text{exponent}}`.
    """
    domain = constraints.positive
    codomain = constraints.positive

    # TODO: Tensor valued exponents and corresponding determination of event_dim
    def __init__(
        self,
        params_fn: Optional[flowtorch.Lazy] = None,
        *,
        shape: torch.Size,
        context_shape: Optional[torch.Size] = None,
        exponent: float = 2.0,
    ) -> None:
        super().__init__(params_fn, shape=shape, context_shape=context_shape)
        self.exponent = exponent

    def _forward(
        self, x: torch.Tensor, params: Optional[Sequence[torch.Tensor]]
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
        y = x.pow(self.exponent)
        ladj = self._log_abs_det_jacobian(x, y, params)
        return y, ladj

    def _inverse(
        self,
        y: torch.Tensor,
        params: Optional[Sequence[torch.Tensor]] = None,
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
        x = y.pow(1 / self.exponent)
        ladj = self._log_abs_det_jacobian(x, y, params)
        return x, ladj

    def _log_abs_det_jacobian(
        self,
        x: torch.Tensor,
        y: torch.Tensor,
        params: Optional[Sequence[torch.Tensor]],
    ) -> torch.Tensor:
        return torch.abs(self.exponent * y / x).log()
