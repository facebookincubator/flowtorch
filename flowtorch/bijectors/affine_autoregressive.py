# Copyright (c) Meta Platforms, Inc

# pyre-unsafe

from typing import Optional

import flowtorch
import flowtorch.parameters
import torch
from flowtorch.bijectors.autoregressive import Autoregressive
from flowtorch.bijectors.ops.affine import Affine as AffineOp


class AffineAutoregressive(AffineOp, Autoregressive):
    def __init__(
        self,
        params_fn: flowtorch.Lazy | None = None,
        *,
        shape: torch.Size,
        context_shape: torch.Size | None = None,
        clamp_values: bool = False,
        log_scale_min_clip: float = -5.0,
        log_scale_max_clip: float = 3.0,
        scale_fn: str = "softplus",
    ) -> None:
        super().__init__(
            params_fn,
            shape=shape,
            context_shape=context_shape,
        )
        self.clamp_values = clamp_values
        self.log_scale_min_clip = log_scale_min_clip
        self.log_scale_max_clip = log_scale_max_clip
        self.scale_fn = scale_fn
