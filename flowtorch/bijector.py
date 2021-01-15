# Copyright (c) FlowTorch Development Team. All Rights Reserved
# SPDX-License-Identifier: MIT

from typing import Optional, Sequence, Tuple

import torch
import torch.distributions
from torch.distributions import constraints

import flowtorch
import flowtorch.distributions


class Bijector(object):
    # Metadata about (the default) bijector
    event_dim = 0
    domain = constraints.real_vector
    codomain = constraints.real_vector
    identity_initialization = True
    autoregressive = False

    # TODO: Returning inverse of bijection
    def __init__(self, param_fn: "flowtorch.Params") -> None:
        super(Bijector, self).__init__()
        self.param_fn = param_fn

    def __call__(
        self, base_dist: torch.distributions.Distribution
    ) -> Tuple[
        flowtorch.distributions.TransformedDistribution, "flowtorch.ParamsModule"
    ]:
        """
        Returns the distribution formed by passing dist through the bijection
        """
        # If the input is a distribution then return transformed distribution
        if isinstance(base_dist, torch.distributions.Distribution):
            # Create transformed distribution
            # TODO: Check that if bijector is autoregressive then parameters are as
            # well Possibly do this in simplex.Bijector.__init__ and call from
            # simple.bijectors.*.__init__
            input_shape = base_dist.batch_shape + base_dist.event_shape
            params = self.param_fn(
                input_shape, self.param_shapes(base_dist)
            )  # <= this is where hypernets etc. are instantiated
            new_dist = flowtorch.distributions.TransformedDistribution(
                base_dist, self, params
            )
            return new_dist, params

        # TODO: Handle other types of inputs such as tensors
        else:
            raise TypeError(f"Bijector called with invalid type: {type(base_dist)}")

    def forward(
        self, x: torch.Tensor, params: Optional["flowtorch.ParamsModule"]
    ) -> torch.Tensor:
        return self._forward(x, params)

    def _forward(
        self, x: torch.Tensor, params: Optional["flowtorch.ParamsModule"]
    ) -> torch.Tensor:
        """
        Abstract method to compute forward transformation.
        """
        raise NotImplementedError

    def inverse(
        self, y: torch.Tensor, params: Optional["flowtorch.ParamsModule"]
    ) -> torch.Tensor:
        return self._inverse(y, params)

    def _inverse(
        self, y: torch.Tensor, params: Optional["flowtorch.ParamsModule"]
    ) -> torch.Tensor:
        """
        Abstract method to compute inverse transformation.
        """
        raise NotImplementedError

    def log_abs_det_jacobian(
        self,
        x: torch.Tensor,
        y: torch.Tensor,
        params: Optional["flowtorch.ParamsModule"],
    ) -> torch.Tensor:
        """
        Computes the log det jacobian `log |dy/dx|` given input and output.
        By default, assumes a volume preserving bijection.
        """
        return self._log_abs_det_jacobian(x, y, params)

    def _log_abs_det_jacobian(
        self,
        x: torch.Tensor,
        y: torch.Tensor,
        params: Optional["flowtorch.ParamsModule"],
    ) -> torch.Tensor:
        """
        Computes the log det jacobian `log |dy/dx|` given input and output.
        By default, assumes a volume preserving bijection.
        """

        # TODO: Sum out self.event_dim right-most dimensions
        # self.event_dim may be > 0 for derived classes!
        return torch.zeros_like(x)

    def param_shapes(
        self, dist: torch.distributions.Distribution
    ) -> Sequence[torch.Size]:
        """
        Given a base distribution, calculate the parameters for the transformation
        of that distribution under this bijector. By default, no parameters are
        set.
        """
        return (torch.Size([]),)

    def __repr__(self) -> str:
        return self.__class__.__name__ + "()"
