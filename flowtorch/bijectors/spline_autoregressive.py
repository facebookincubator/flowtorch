# Copyright (c) Meta Platforms, Inc

# pyre-unsafe

from typing import Optional

import flowtorch
import flowtorch.parameters
import torch
from flowtorch.bijectors.autoregressive import Autoregressive
from flowtorch.bijectors.ops.spline import Spline as SplineOp


class SplineAutoregressive(SplineOp, Autoregressive):
    def __init__(
        self,
        params_fn: flowtorch.Lazy | None = None,
        *,
        shape: torch.Size,
        context_shape: torch.Size | None = None,
        count_bins: int = 8,
        bound: float = 3.0,
        order: str = "linear",
    ) -> None:
        super().__init__(
            params_fn,
            shape=shape,
            context_shape=context_shape,
            count_bins=count_bins,
            bound=bound,
            order=order,
        )
