# Copyright (c) Meta Platforms, Inc

from typing import Optional, Sequence, Tuple

import flowtorch.parameters

import torch
from flowtorch.bijectors.ops.affine import Affine as AffineOp
from flowtorch.parameters import DenseCoupling


class Coupling(AffineOp):
    def __init__(
        self,
        params_fn: Optional[flowtorch.Lazy] = None,
        *,
        shape: torch.Size,
        context_shape: Optional[torch.Size] = None,
        clamp_values: bool = False,
        log_scale_min_clip: float = -5.0,
        log_scale_max_clip: float = 3.0,
        sigmoid_bias: float = 2.0,
        positive_map: str = "softplus",
        positive_bias: Optional[float] = None,
    ) -> None:

        if params_fn is None:
            params_fn = DenseCoupling()  # type: ignore

        AffineOp.__init__(
            self,
            params_fn,
            shape=shape,
            context_shape=context_shape,
            clamp_values=clamp_values,
            log_scale_min_clip=log_scale_min_clip,
            log_scale_max_clip=log_scale_max_clip,
            sigmoid_bias=sigmoid_bias,
            positive_map=positive_map,
            positive_bias=positive_bias,
        )

    def _forward(
        self, x: torch.Tensor, params: Optional[Sequence[torch.Tensor]]
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        assert self._params_fn is not None

        x = x[..., self._params_fn.permutation]
        y, ldj = super()._forward(x, params)
        y = y[..., self._params_fn.inv_permutation]
        return y, ldj

    def _inverse(
        self, y: torch.Tensor, params: Optional[Sequence[torch.Tensor]]
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        assert self._params_fn is not None

        y = y[..., self._params_fn.inv_permutation]
        x, ldj = super()._inverse(y, params)
        x = x[..., self._params_fn.permutation]
        return x, ldj
