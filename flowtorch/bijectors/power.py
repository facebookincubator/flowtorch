# Copyright (c) Meta Platforms, Inc

# pyre-unsafe

from collections.abc import Sequence
from typing import Optional, Tuple

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
        params_fn: flowtorch.Lazy | None = None,
        *,
        shape: torch.Size,
        context_shape: torch.Size | None = None,
        exponent: float = 2.0,
    ) -> None:
        super().__init__(params_fn, shape=shape, context_shape=context_shape)
        self.exponent = exponent

    def _forward(
        self, x: torch.Tensor, params: Sequence[torch.Tensor] | None
    ) -> tuple[torch.Tensor, torch.Tensor | None]:
        y = x.pow(self.exponent)
        ladj = self._log_abs_det_jacobian(x, y, params)
        return y, ladj

    def _inverse(
        self,
        y: torch.Tensor,
        params: Sequence[torch.Tensor] | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor | None]:
        x = y.pow(1 / self.exponent)
        ladj = self._log_abs_det_jacobian(x, y, params)
        return x, ladj

    def _log_abs_det_jacobian(
        self,
        x: torch.Tensor,
        y: torch.Tensor,
        params: Sequence[torch.Tensor] | None,
    ) -> torch.Tensor:
        return torch.abs(self.exponent * y / x).log()
