# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved
# SPDX-License-Identifier: MIT

from typing import Sequence

import flowtorch
import flowtorch.params
import torch
import torch.distributions
from flowtorch.bijectors.base import Bijector
from torch.distributions import constraints
from torch.distributions.utils import _sum_rightmost


class Compose(Bijector):
    def __init__(
        self,
        shape: torch.Size,
        # params: Optional[flowtorch.Lazy] = None,
        context_size: int = 0,
        *,
        bijectors: Sequence[flowtorch.Lazy],
    ):
        assert len(bijectors) > 0

        # Instantiate all bijectors, propagating shape information
        self.bijectors = []
        for bijector in bijectors:
            assert issubclass(bijector.cls, Bijector)

            self.bijectors.append(bijector(shape))
            shape = self.bijectors[-1].forward_shape(shape)

        # TODO: Adjust domain accordingly and check domain/codomain compatibility!
        event_dim = self.bijectors[0].domain.event_dim
        self.domain = constraints.independent(constraints.real, event_dim)
        self.codomain = constraints.independent(constraints.real, event_dim)
        self._inv = None

        # self.identity_initialization = all(
        #    b.identity_initialization for b in self.bijectors
        # )
        self.autoregressive = all(b.autoregressive for b in self.bijectors)
        self._context_size = context_size

    # NOTE: We overwrite forward rather than _forward so that the composed
    # bijectors can handle the caching separately!
    def forward(self, x, context=None):
        for bijector in self.bijectors:
            x = bijector.forward(x, context)

        return x

    def inverse(self, y, context=None):
        for bijector in reversed(self.bijectors):
            y = bijector.inverse(y, context)

        return y

    def log_abs_det_jacobian(self, x, y, context=None):
        """
        Computes the log det jacobian `log |dy/dx|` given input and output.
        By default, assumes a volume preserving bijection.
        """
        ldj = _sum_rightmost(
            torch.zeros_like(y),
            self.domain.event_dim,
        )
        for bijector in reversed(self.bijectors):
            y_inv = bijector.inverse(y, context)
            ldj += bijector.log_abs_det_jacobian(y_inv, y, context)
            y = y_inv
        return ldj

    def param_shapes(self, shape):
        """
        Given a base distribution, calculate the parameters for the transformation
        of that distribution under this bijector. By default, no parameters are
        set.
        """
        return None
