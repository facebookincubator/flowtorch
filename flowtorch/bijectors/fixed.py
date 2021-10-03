# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved
# SPDX-License-Identifier: MIT
from typing import Optional, Sequence

import flowtorch
import torch
import torch.distributions
from flowtorch.bijectors.base import Bijector


class Fixed(Bijector):
    def __init__(
        self,
        shape: torch.Size,
        params: Optional[flowtorch.Lazy] = None,
        context_size: int = 0,
    ) -> None:
        super().__init__(shape, params, context_size)
        assert params is None

    def forward(
        self,
        x: torch.Tensor,
        context: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        assert context is None or context.shape == (self._context_size,)
        return self._forward(x, context)

    def inverse(
        self,
        y: torch.Tensor,
        context: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        assert context is None or context.shape == (self._context_size,)
        return self._inverse(y, context)

    def log_abs_det_jacobian(
        self,
        x: torch.Tensor,
        y: torch.Tensor,
        context: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """
        Computes the log det jacobian `log |dy/dx|` given input and output.
        By default, assumes a volume preserving bijection.
        """
        assert context is None or context.shape == (self._context_size,)
        return self._log_abs_det_jacobian(x, y, context)

    def param_shapes(self, shape: torch.Size) -> Sequence[torch.Size]:
        """
        Given a base distribution, calculate the parameters for the transformation
        of that distribution under this bijector. By default, no parameters are
        set.
        """
        return []
