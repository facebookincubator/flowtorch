# Copyright (c) Meta Platforms, Inc
from typing import Optional

import flowtorch
import torch
from flowtorch.bijectors.base import Bijector


class Invert(Bijector):
    """
    Lazily inverts a bijector by swapping the forward and inverse operations.

    `Invert` flips a bijector such that forward calls inverse and inverse
    calls forward. The log-abs-det-Jacobian is adjusted accordingly.

    Args:
        bijector (Bijector): layer to be inverted

    Examples:

    """

    def __init__(
        self,
        bijector: flowtorch.Lazy,
        *,
        shape: torch.Size,
        context_shape: Optional[torch.Size] = None
    ) -> None:
        # TODO: Handle context_shape

        self.bijector = bijector(shape=shape)
        if hasattr(self.bijector, "_params_fn"):
            self._params_fn = self.bijector._params_fn  # type: ignore

    def forward(
        self,
        x: torch.Tensor,
        context: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        y = self.bijector.inverse(x, context=context)  # type: ignore
        return y

    def inverse(
        self,
        y: torch.Tensor,
        x: Optional[torch.Tensor] = None,
        context: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        if x is not None:
            raise RuntimeError("x must be None when calling InverseBijector.inverse")
        x = self.bijector.forward(y, context=context)  # type: ignore
        return x

    def log_abs_det_jacobian(
        self,
        x: torch.Tensor,
        y: torch.Tensor,
        context: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        return self.bijector.log_abs_det_jacobian(y, x, context)  # type: ignore
