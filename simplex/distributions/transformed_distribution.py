# Copyright (c) Simplex Development Team. All Rights Reserved
# SPDX-License-Identifier: MIT

import torch
import torch.distributions
from torch.distributions import constraints
from torch.distributions.utils import _sum_rightmost

import simplex
import weakref

class TransformedDistribution(torch.distributions.Distribution):
    def __init__(self, base_distribution, bijectors, params, validate_args=None):
        self.base_dist = base_distribution

        if not isinstance(params, list):
            params = [params, ]
        #self.params = [weakref.ref(p) for p in params]
        self.params = [weakref.proxy(p) for p in params]
        #self.params = [p for p in params]

        if isinstance(bijectors, simplex.Bijector):
            self.bijectors = [bijectors, ]
        elif isinstance(bijectors, list):
            if not all(isinstance(b, simplex.Bijector) for b in bijectors):
                raise ValueError("transforms must be a Transform or a list of Transforms")
            self.bijectors = bijectors
        else:
            raise ValueError(f"transforms must be a Transform or list, but was {bijectors}")
        shape = self.base_dist.batch_shape + self.base_dist.event_shape
        event_dim = max([len(self.base_dist.event_shape)] + [b.event_dim for b in self.bijectors])
        batch_shape = shape[:len(shape) - event_dim]
        event_shape = shape[len(shape) - event_dim:]
        super(TransformedDistribution, self).__init__(batch_shape, event_shape, validate_args=validate_args)
    
    def sample(self, sample_shape=torch.Size()):
        """
        Generates a sample_shape shaped sample or sample_shape shaped batch of
        samples if the distribution parameters are batched. Samples first from
        base distribution and applies `transform()` for every transform in the
        list.
        """
        with torch.no_grad():
            x = self.base_dist.sample(sample_shape)
            for bijector, param in zip(self.bijectors, self.params):
                x = bijector.forward(x, param)
            return x

    def rsample(self, sample_shape=torch.Size()):
        """
        Generates a sample_shape shaped reparameterized sample or sample_shape
        shaped batch of reparameterized samples if the distribution parameters
        are batched. Samples first from base distribution and applies
        `transform()` for every transform in the list.
        """
        x = self.base_dist.rsample(sample_shape)
        for bijector, param in zip(self.bijectors, self.params):
            x = bijector.forward(x, param)
        return x

    def log_prob(self, value):
        """
        Scores the sample by inverting the transform(s) and computing the score
        using the score of the base distribution and the log abs det jacobian.
        """
        event_dim = len(self.event_shape)
        log_prob = 0.0
        y = value
        for bijector, param in zip(reversed(self.bijectors), reversed(self.params)):
            x = bijector.inverse(y, param)
            log_prob = log_prob - _sum_rightmost(bijector.log_abs_det_jacobian(x, y, param),
                                                 event_dim - bijector.event_dim)
            y = x

        log_prob = log_prob + _sum_rightmost(self.base_dist.log_prob(y),
                                             event_dim - len(self.base_dist.event_shape))
        return log_prob
