# Copyright (c) Meta Platforms, Inc
import math
from typing import Optional, Sequence, Tuple

import flowtorch
import torch
import torch.distributions.constraints as constraints
from flowtorch.bijectors.fixed import Fixed


class Banana(Fixed):
    domain = constraints.real_vector
    codomain = constraints.real_vector
    b_coef = 0.02

    def __init__(
        self,
        params_fn: Optional[flowtorch.Lazy] = None,
        *,
        shape: torch.Size,
        context_shape: Optional[torch.Size] = None
    ) -> None:
        super().__init__(params_fn, shape=shape, context_shape=context_shape)

        # Check that we're operating on a bivariate base distribution
        if len(shape) != 1 or shape[-1] != 2:
            raise ValueError("Base distribution to Banana is not bivariate.")

    def _forward(
        self, x: torch.Tensor, params: Optional[Sequence[torch.Tensor]]
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
        y0 = 10.0 * x[..., 0]
        y1 = x[..., 1] + 100.0 * self.b_coef * torch.square(x[..., 0]) - 100.0 * 0.02
        y = torch.stack([y0, y1], dim=-1)
        ladj = self._log_abs_det_jacobian(x, y, params)
        return y, ladj

    def _inverse(
        self, y: torch.Tensor, params: Optional[Sequence[torch.Tensor]]
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
        x0 = 0.1 * y[..., 0]
        x1 = y[..., 1] - self.b_coef * torch.square(y[..., 0]) + 100.0 * 0.02
        x = torch.stack([x0, x1], dim=-1)
        ladj = self._log_abs_det_jacobian(x, y, params)
        return y, ladj

    def _log_abs_det_jacobian(
        self, x: torch.Tensor, y: torch.Tensor, params: Optional[Sequence[torch.Tensor]]
    ) -> torch.Tensor:
        return math.log(10.0) * torch.ones(size=x.shape[:-1])
