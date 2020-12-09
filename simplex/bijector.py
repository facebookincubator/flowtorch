# Copyright (c) Simplex Development Team. All Rights Reserved
# SPDX-License-Identifier: MIT

import torch
import torch.distributions
from torch.distributions import constraints

class Bijector(object):
    event_dim = 0
    domain = constraints.real_vector
    codomain = constraints.real_vector

    # TODO: Returning inverse of bijection
    #def __init__(self):
    #    self._inv = None
    #    super(Bijector, self).__init__()

    def __call__(self, x):
        """
        Returns the distribution formed by passing dist through the bijection
        """
        # If the input is a distribution then return transformed distribution
        if isinstance(x, torch.distributions.Distribution):
            raise NotImplementedError('Transforms of distributions not yet implemented!')

        # TODO: Handle other types of inputs such as tensors
        else:
            raise TypeError(f'Bijector called with invalid type: {type(x)}')

    def forward(self, x, params=None):
        """
        Abstract method to compute forward transformation.
        """
        raise NotImplementedError

    def inverse(self, y, params=None):
        """
        Abstract method to compute inverse transformation.
        """
        raise NotImplementedError

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
        return None

    def __repr__(self):
        return self.__class__.__name__ + '()'
