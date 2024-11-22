# Copyright (c) Meta Platforms, Inc

# pyre-unsafe
from collections.abc import Sequence
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
        context_shape: torch.Size | None = None,
    ) -> None:
        b = bijector(shape=shape)
        super().__init__(None, shape=shape, context_shape=context_shape)
        self.bijector = b

    def forward(
        self,
        x: torch.Tensor,
        context: torch.Tensor | None = None,
    ) -> torch.Tensor:
        y = self.bijector.inverse(x, context=context)  # type: ignore
        return y

    def inverse(
        self,
        y: torch.Tensor,
        x: torch.Tensor | None = None,
        context: torch.Tensor | None = None,
    ) -> torch.Tensor:
        if x is not None:
            raise RuntimeError("x must be None when calling InverseBijector.inverse")
        x = self.bijector.forward(y, context=context)  # type: ignore
        return x

    def log_abs_det_jacobian(
        self,
        x: torch.Tensor,
        y: torch.Tensor,
        context: torch.Tensor | None = None,
    ) -> torch.Tensor:
        return self.bijector.log_abs_det_jacobian(y, x, context)  # type: ignore

    def param_shapes(self, shape: torch.Size) -> Sequence[torch.Size]:
        return self.bijector.param_shapes(shape)  # type: ignore

    def __repr__(self) -> str:
        return self.bijector.__repr__()  # type: ignore

    def forward_shape(self, shape: torch.Size) -> torch.Size:
        """
        Infers the shape of the forward computation, given the input shape.
        Defaults to preserving shape.
        """
        return self.bijector.forward_shape(shape)  # type: ignore

    def inverse_shape(self, shape: torch.Size) -> torch.Size:
        """
        Infers the shapes of the inverse computation, given the output shape.
        Defaults to preserving shape.
        """
        return self.bijector.inverse_shape(shape)  # type: ignore
