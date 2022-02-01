# Copyright (c) Meta Platforms, Inc

import math
from typing import Optional, Sequence, Tuple

import flowtorch
import torch
from flowtorch.bijectors.fixed import Fixed
from flowtorch.bijectors.utils import requires_log_detJ


class AffineFixed(Fixed):
    r"""
    Elementwise bijector via the affine mapping :math:`\mathbf{y} = \mu +
    \sigma \otimes \mathbf{x}` where $\mu$ and $\sigma$ are fixed rather than
    learnable.
    """

    # TODO: Handle non-scalar loc and scale with correct broadcasting semantics
    def __init__(
        self,
        params_fn: Optional[flowtorch.Lazy] = None,
        *,
        shape: torch.Size,
        context_shape: Optional[torch.Size] = None,
        loc: float = 0.0,
        scale: float = 1.0
    ) -> None:
        super().__init__(params_fn, shape=shape, context_shape=context_shape)
        self.loc = loc
        self.scale = scale

    def _forward(
        self,
        x: torch.Tensor,
        params: Optional[Sequence[torch.Tensor]],
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
        y = self.loc + self.scale * x
        ladj: Optional[torch.Tensor] = None
        if requires_log_detJ():
            ladj = self._log_abs_det_jacobian(x, y, params)
        return y, ladj

    def _inverse(
        self, y: torch.Tensor, params: Optional[Sequence[torch.Tensor]]
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
        x = (y - self.loc) / self.scale
        ladj: Optional[torch.Tensor] = None
        if requires_log_detJ():
            ladj = self._log_abs_det_jacobian(x, y, params)
        return x, ladj

    def _log_abs_det_jacobian(
        self, x: torch.Tensor, y: torch.Tensor, params: Optional[Sequence[torch.Tensor]]
    ) -> torch.Tensor:
        return torch.full_like(x, math.log(abs(self.scale)))
