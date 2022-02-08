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
        log_scale_min_clip: float = -5.0,
        log_scale_max_clip: float = 3.0,
        sigmoid_bias: float = 2.0,
    ) -> None:
        super().__init__(
            params_fn,
            shape=shape,
            context_shape=context_shape,
        )
        self.log_scale_min_clip = log_scale_min_clip
        self.log_scale_max_clip = log_scale_max_clip
        self.sigmoid_bias = sigmoid_bias
