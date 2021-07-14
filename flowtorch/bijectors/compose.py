# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved
# SPDX-License-Identifier: MIT

from typing import Optional, Sequence, Tuple

import torch
import torch.distributions
from flowtorch.bijectors.base import Bijector
from flowtorch.distributions.transformed_distribution import TransformedDistribution
from flowtorch.params.base import ParamsModule, ParamsModuleList
from torch.distributions import Distribution
from torch.distributions import constraints
from torch.distributions.utils import _sum_rightmost


class Compose(Bijector):
    def __init__(self, bijectors: Sequence[Bijector], context_size: int = 0):
        self.bijectors = bijectors

        # TODO: Adjust domain accordingly and check domain/codomain compatibility!
        event_dim = (
            1 if len(self.bijectors) == 0 else self.bijectors[0].domain.event_dim
        )
        self.domain = constraints.independent(constraints.real, event_dim)
        self.codomain = constraints.independent(constraints.real, event_dim)
        self._inv = None

        self.identity_initialization = all(
            b.identity_initialization for b in self.bijectors
        )
        self.autoregressive = all(b.autoregressive for b in self.bijectors)
        self._context_size = context_size

    def __call__(
        self, base_dist: Distribution
    ) -> Tuple[TransformedDistribution, Optional[ParamsModuleList]]:
        """
        Returns the distribution formed by passing dist through the bijection
        """
        # If the input is a distribution then return transformed distribution
        if isinstance(base_dist, Distribution):
            # Create transformed distribution
            # TODO: Check that if bijector is autoregressive then parameters
            # are as well Possibly do this in simplex.Bijector.__init__ and
            # call from simple.bijectors.*.__init__
            input_shape = base_dist.batch_shape + base_dist.event_shape
            params = self.param_fn(
                input_shape, self.param_shapes(base_dist), self._context_size
            )  # <= this is where hypernets etc. are instantiated
            new_dist = TransformedDistribution(base_dist, self, params)
            return new_dist, params

        # TODO: Handle other types of inputs such as tensors
        else:
            raise TypeError(f"Bijector called with invalid type: {type(base_dist)}")

    def param_fn(
        self,
        input_shape: Sequence[int],
        param_shapes: Sequence[Sequence[int]],
        context_size: int,
    ) -> ParamsModuleList:
        return ParamsModuleList(
            [
                b.param_fn(input_shape, pshape, context_size)
                for b, pshape in zip(self.bijectors, param_shapes)
            ]
        )

    # NOTE: We overwrite forward rather than _forward so that the composed
    # bijectors can handle the caching separately!
    def forward(
        self,
        x: torch.Tensor,
        params: Optional[ParamsModuleList] = None,
        context: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        assert len(params) == len(self.bijectors)

        for bijector, param in zip(self.bijectors, params):
            x = bijector.forward(x, param, context)

        return x

    def inverse(
        self,
        y: torch.Tensor,
        params: Optional[ParamsModuleList] = None,
        context: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        assert len(params) == len(self.bijectors)

        for bijector, param in zip(reversed(self.bijectors), reversed(params)):
            y = bijector.inverse(y, param, context)

        return y

    def log_abs_det_jacobian(
        self,
        x: torch.Tensor,
        y: torch.Tensor,
        params: Optional[ParamsModuleList] = None,
        context: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """
        Computes the log det jacobian `log |dy/dx|` given input and output.
        By default, assumes a volume preserving bijection.
        """
        ldj = _sum_rightmost(
            torch.zeros_like(y),
            self.domain.event_dim,
        )
        for bijector, param in zip(reversed(self.bijectors), reversed(params)):
            y_inv = bijector.inverse(y, param, context)
            ldj += bijector.log_abs_det_jacobian(y_inv, y, param, context)
            y = y_inv
        return ldj

    def param_shapes(
        self, dist: torch.distributions.Distribution
    ) -> Sequence[Sequence[int]]:
        """
        Given a base distribution, calculate the parameters for the transformation
        of that distribution under this bijector. By default, no parameters are
        set.
        """
        p_shapes = []

        for b in self.bijectors:
            p_shapes.append(b.param_shapes(dist=dist))  # TODO: fix dist

        return p_shapes
