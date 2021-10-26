# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved
# SPDX-License-Identifier: MIT

from typing import Optional, Sequence

import flowtorch
import flowtorch.parameters
import torch
import torch.distributions
from flowtorch.bijectors.base import Bijector
from torch.distributions.utils import _sum_rightmost


class Compose(Bijector):
    def __init__(
        self,
        bijectors: Sequence[flowtorch.Lazy],
        *,
        shape: torch.Size,
        context_shape: Optional[torch.Size] = None,
    ):
        assert len(bijectors) > 0

        # Instantiate all bijectors, propagating shape information
        self.bijectors = []
        for bijector in bijectors:
            assert issubclass(bijector.cls, Bijector)

            self.bijectors.append(bijector(shape=shape))
            shape = self.bijectors[-1].forward_shape(shape)  # type: ignore

        self.domain = self.bijectors[0].domain  # type: ignore
        self.codomain = self.bijectors[-1].codomain  # type: ignore

        # Make parameters accessible to dist.Flow
        self._params = torch.nn.ModuleList(
            [
                b._params  # type: ignore
                for b in self.bijectors
                if isinstance(b._params, torch.nn.Module)  # type: ignore
            ]
        )

        self._context_shape = context_shape

    # NOTE: We overwrite forward rather than _forward so that the composed
    # bijectors can handle the caching separately!
    def forward(self, x: torch.Tensor, context: torch.Tensor = None) -> torch.Tensor:
        for bijector in self.bijectors:
            x = bijector.forward(x, context)  # type: ignore

        return x

    def inverse(
        self,
        y: torch.Tensor,
        x: Optional[torch.Tensor] = None,
        context: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        for bijector in reversed(self.bijectors):
            y = bijector.inverse(y, x, context)  # type: ignore

        return y

    def log_abs_det_jacobian(
        self, x: torch.Tensor, y: torch.Tensor, context: torch.Tensor = None
    ) -> torch.Tensor:
        """
        Computes the log det jacobian `log |dy/dx|` given input and output.
        By default, assumes a volume preserving bijection.
        """
        ldj = _sum_rightmost(
            torch.zeros_like(y),
            self.domain.event_dim,
        )
        for bijector in reversed(self.bijectors):
            y_inv = bijector.inverse(y, context)  # type: ignore
            ldj += bijector.log_abs_det_jacobian(y_inv, y, context)  # type: ignore
            y = y_inv
        return ldj

    def param_shapes(self, shape: torch.Size) -> Sequence[torch.Size]:
        """
        Given a base distribution, calculate the parameters for the transformation
        of that distribution under this bijector. By default, no parameters are
        set.
        """
        return []
