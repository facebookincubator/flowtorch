# Copyright (c) Simplex Development Team. All Rights Reserved
# SPDX-License-Identifier: MIT

import torch

class Bijector(object):
    event_dim = 0

    # TODO: Returning inverse of bijection
    #def __init__(self):
    #    self._inv = None
    #    super(Bijector, self).__init__()

    def __call__(self, dist):
        """
        Returns the distribution formed by passing dist through the bijection
        """
        # TODO: Instantiate a TransformedDistribution class
        
        pass

    def forward(self, x):
        """
        Abstract method to compute forward transformation.
        """
        raise NotImplementedError

    def inverse(self, y):
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

    def __repr__(self):
        return self.__class__.__name__ + '()'
