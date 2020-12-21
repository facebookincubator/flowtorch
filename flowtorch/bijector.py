# Copyright (c) FlowTorch Development Team. All Rights Reserved
# SPDX-License-Identifier: MIT

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

    x_cache = None
    y_cache = None
    J_cache = None
    state_cache = 0

    # TODO: Returning inverse of bijection
    def __init__(self, param_fn, **kwargs):
        super(Bijector, self).__init__()
        # self._inv = None
        self.param_fn = param_fn
        for n, v in kwargs.items():
            setattr(self, n, v)

    def __call__(self, base_dist):
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

    def forward(self, x, params=None):
        """
        Layer of indirection to implement caching
        """
        # if self.x_cache is x and self.state_cache == params.param.state:
        #     return self.y_cache
        # else:
        #     y = self._forward(x, params)
        #     self.x_cache = x
        #     self.y_cache = y
        #     self.state_cache = params.params.state
        #     return y
        return self._forward(x, params)

    def _forward(self, x, params=None):
        """
        Abstract method to compute forward transformation.
        """
        raise NotImplementedError

    def inverse(self, y, params=None):
        if self.y_cache is y and self.state_cache == params.params.state:
            return self.x_cache
        else:
            x = self._inverse(y, params)
            self.x_cache = x
            self.y_cache = y
            self.state_cache = params.params.state
            return x

    def _inverse(self, y, params=None):
        """
        Abstract method to compute inverse transformation.
        """
        raise NotImplementedError

    def log_abs_det_jacobian(self, x, y, params=None):
        """
        Computes the log det jacobian `log |dy/dx|` given input and output.
        By default, assumes a volume preserving bijection.
        """
        if (
            self.x_cache is x
            and self.y_cache is y
            and self.J_cache is not None
            and self.state_cache == params.params.state
        ):
            return self.J_cache
        else:
            J = self._log_abs_det_jacobian(x, y, params)
            self.x_cache = x
            self.y_cache = y
            self.J_cache = J
            self.state_cache = params.params.state
            return J

    def _log_abs_det_jacobian(self, x, y, params=None):
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
        return None

    def __repr__(self):
        return self.__class__.__name__ + "()"
