# Copyright (c) Simplex Development Team. All Rights Reserved
# SPDX-License-Identifier: MIT

import torch
import torch.distributions
from torch.distributions import constraints

import simplex


class Compose(simplex.Bijector):
    def __init__(self, bijectors):
        self.bijectors = bijectors
        self.event_dim = max([b.event_dim for b in self.bijectors])

        # TODO: Adjust domain accordingly and check domain/codomain compatibility!
        self.domain = constraints.real_vector
        self.codomain = constraints.real_vector

        self.identity_initialization = all(
            [b.identity_intialization for b in self.bijectors]
        )
        self.autoregressive = all([b.autoregressive for b in self.bijectors])

    def __call__(self, x):
        """
        Returns the distribution formed by passing dist through the bijection
        """
        # If the input is a distribution then return transformed distribution
        if isinstance(x, torch.distributions.Distribution):
            # Create transformed distribution
            # TODO: Check that if bijector is autoregressive then parameters are as well
            # Possibly do this in simplex.Bijector.__init__ and call from simple.bijectors.*.__init__
            input_shape = x.batch_shape + x.event_shape
            params = self.param_fn(
                input_shape, self.param_shapes(x)
            )  # <= this is where hypernets etc. are instantiated
            new_dist = simplex.distributions.TransformedDistribution(x, self, params)
            return new_dist, params

        # TODO: Handle other types of inputs such as tensors
        else:
            raise TypeError(f"Bijector called with invalid type: {type(x)}")

    # NOTE: We overwrite forward rather than _forward so that the composed bijectors
    # can handle the caching separately!
    def forward(self, x, params=None):
        assert len(params) == len(self.bijectors)

        for bijector, param in zip(self.bijectors, params):
            x = bijector.forward(x, param)

        return x

    def inverse(self, y, params=None):
        assert len(params) == len(self.bijectors)

        for bijector, param in zip(reversed(self.bijectors), reversed(params)):
            y = bijector.inverse(y, param)

        return y

    def log_abs_det_jacobian(self, x, y, params=None):
        """
        Computes the log det jacobian `log |dy/dx|` given input and output.
        By default, assumes a volume preserving bijection.
        """

        # TODO: Sum out self.event_dim right-most dimensions
        # self.event_dim may be > 0 for derived classes!
        return torch.zeros_like(x)

    def param_shapes(self, dist):
        """
        Given a base distribution, calculate the parameters for the transformation
        of that distribution under this bijector. By default, no parameters are
        set.
        """
        p_shapes = []

        for b in self.bijectors:
            p_shapes.append(b.param_shapes())

        return None
