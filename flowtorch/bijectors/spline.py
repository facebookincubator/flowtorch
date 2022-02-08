# Copyright (c) Meta Platforms, Inc

from typing import Optional

import flowtorch
import torch
from flowtorch.bijectors.elementwise import Elementwise
from flowtorch.bijectors.ops.spline import Spline as SplineOp


class Spline(SplineOp, Elementwise):
    def __init__(
        self,
        params_fn: Optional[flowtorch.Lazy] = None,
        *,
        shape: torch.Size,
        context_shape: Optional[torch.Size] = None,
        count_bins: int = 8,
        bound: float = 3.0,
        order: str = "linear"
    ) -> None:
        super().__init__(
            params_fn,
            shape=shape,
            context_shape=context_shape,
            count_bins=count_bins,
            bound=bound,
            order=order,
        )
