from copy import deepcopy
from typing import Optional, Sequence, Tuple

import flowtorch
import torch
from torch.distributions import constraints
from torch.nn import functional as F

from ..parameters.conv11 import Conv1x1Params
from .base import Bijector

_REAL3d = deepcopy(constraints.real)
_REAL3d.event_dim = 3


class SomeOtherClass(Bijector):
    pass


class Conv1x1Bijector(SomeOtherClass):

    domain: constraints.Constraint = _REAL3d
    codomain: constraints.Constraint = _REAL3d

    def __init__(
        self,
        params_fn: Optional[flowtorch.Lazy] = None,
        *,
        shape: torch.Size,
        context_shape: Optional[torch.Size] = None,
        LU_decompose: bool = False,
        double_solve: bool = False,
        zero_init: bool = False,
    ):
        if params_fn is None:
            params_fn = Conv1x1Params(LU_decompose, zero_init=zero_init)  # type: ignore
        self._LU = LU_decompose
        self._double_solve = double_solve
        self.dims = (-3, -2, -1)
        super().__init__(
            params_fn=params_fn,
            shape=shape,
            context_shape=context_shape,
        )

    def _forward(
        self,
        x: torch.Tensor,
        params: Optional[Sequence[torch.Tensor]],
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
        assert isinstance(params, (list, tuple))
        weight, logdet = params
        unsqueeze = False
        if x.ndimension() == 3:
            x = x.unsqueeze(0)
            unsqueeze = True
        z = F.conv2d(x, weight)
        if unsqueeze:
            z = z.squeeze(0)
        return z, logdet

    def _inverse(
        self,
        y: torch.Tensor,
        params: Optional[Sequence[torch.Tensor]],
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
        assert isinstance(params, (list, tuple))
        unsqueeze = False
        if y.ndimension() == 3:
            y = y.unsqueeze(0)
            unsqueeze = True

        if self._LU:
            p, low, up, logdet = params
            dtype = low.dtype
            output_view = y.permute(0, 2, 3, 1).unsqueeze(-1)
            if self._double_solve:
                low = low.double()
                p = p.double()
                output_view = output_view.double()
                up = up.double()

            z_view = torch.triangular_solve(
                torch.triangular_solve(
                    p.transpose(-1, -2) @ output_view, low, upper=False
                )[0],
                up,
                upper=True,
            )[0]

            if self._double_solve:
                z_view = z_view.to(dtype)

            z = z_view.squeeze(-1).permute(0, 3, 1, 2)
        else:
            weight, logdet = params
            z = F.conv2d(y, weight)

        if unsqueeze:
            z = z.squeeze(0)
            logdet = logdet.squeeze(0)
        return z, logdet.expand_as(z.sum(self.dims))

    def param_shapes(self, shape: torch.Size) -> Sequence[torch.Size]:
        shape = torch.Size([shape[-3], shape[-3], 1, 1])
        if not self._LU:
            return (shape,)
        else:
            return (shape, shape, shape)
