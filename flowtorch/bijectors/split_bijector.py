# Copyright (c) Meta Platforms, Inc
from typing import Tuple, Optional, Sequence

import flowtorch
import torch
from torch.nn.functional import softplus

from ..parameters import ZeroConv2d
from . import Bijector
from .utils import _sum_rightmost_over_tuple


class ReshapeBijector(Bijector):
    pass


class SplitBijector(ReshapeBijector):
    BIAS_SOFTPLUS = 0.54

    def __init__(
        self,
        params_fn: Optional[flowtorch.Lazy] = None,
        *,
        shape: torch.Size,
        transform: Bijector,
        chunk_dim: int = -3,
        context_shape: Optional[torch.Size] = None,
    ) -> None:
        if params_fn is None:
            params_fn = ZeroConv2d()

        super().__init__(params_fn, shape=shape, context_shape=context_shape)
        self._transform = transform
        self.chunk_dim = chunk_dim

    def _forward_pre_ops(self, x: torch.Tensor) -> Tuple[torch.Tensor, ...]:
        x1, x2 = x.chunk(2, dim=self.chunk_dim)
        return x1, x2

    def _inverse_pre_ops(self, y: torch.Tensor) -> Tuple[torch.Tensor, ...]:
        y1, y2 = y.chunk(2, dim=self.chunk_dim)
        x1 = self._transform.inverse(y1)
        return x1, y2

    def _forward(
        self,
        *x: torch.Tensor,
        params: Optional[Sequence[torch.Tensor]],
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
        x1, x2 = x
        loc, scale = params
        scale = softplus(scale + self.BIAS_SOFTPLUS)
        y1 = self._transform.forward(x1)
        y2 = (x2 - loc) / scale.clamp_min(1e-5)
        ldj = self._transform.log_abs_det_jacobian(x1, y1)
        ldj1, ldj2 = _sum_rightmost_over_tuple(ldj, -scale.log())
        return torch.cat([y1, y2], self.chunk_dim), ldj1 + ldj2

    def _inverse(
        self,
        *y: torch.Tensor,
        params: Optional[Sequence[torch.Tensor]],
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
        x1, y2 = y
        loc, scale = params
        scale = softplus(scale + self.BIAS_SOFTPLUS)
        x2 = y2 * scale + loc
        ldj = self._transform.log_abs_det_jacobian(
            x1, x1.get_parent_from_bijector(self._transform)
        )
        ldj1, ldj2 = _sum_rightmost_over_tuple(ldj, -scale.log())
        return torch.cat([x1, x2], self.chunk_dim), ldj1 + ldj2

    def param_shapes(self, shape: torch.Size) -> Tuple[torch.Size, torch.Size]:
        # A mean and log variance for every dimension of the event shape
        return shape, shape
