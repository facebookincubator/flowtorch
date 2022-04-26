# Copyright (c) Meta Platforms, Inc

from typing import Optional

import flowtorch
import flowtorch.parameters
import torch
from flowtorch.bijectors.autoregressive import Autoregressive
from flowtorch.bijectors.ops.affine import Affine as AffineOp


class AffineAutoregressive(AffineOp, Autoregressive):
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
        Autoregressive.__init__(
            self,
            params_fn,
            shape=shape,
            context_shape=context_shape,
        )
