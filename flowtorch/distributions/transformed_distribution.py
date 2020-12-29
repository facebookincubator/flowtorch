# Copyright (c) FlowTorch Development Team. All Rights Reserved
# SPDX-License-Identifier: MIT

import weakref

import torch
import torch.distributions
from torch import Tensor
from torch.distributions.utils import _sum_rightmost


class TransformedDistribution(torch.distributions.Distribution):
    def __init__(self, base_distribution, bijector, params, validate_args=None):
        self.base_dist = base_distribution

        self.params = weakref.proxy(params)
        self.bijector = bijector
        self.cache = {}

        shape = self.base_dist.batch_shape + self.base_dist.event_shape
        event_dim = max(len(self.base_dist.event_shape), self.bijector.event_dim)
        batch_shape = shape[: len(shape) - event_dim]
        event_shape = shape[len(shape) - event_dim :]
        super(TransformedDistribution, self).__init__(
            batch_shape, event_shape, validate_args=validate_args
        )

    def sample(
        self,
        sample_shape=torch.Size(),  # noqa: B008
    ) -> Tensor:
        """
        Generates a sample_shape shaped sample or sample_shape shaped batch of
        samples if the distribution parameters are batched. Samples first from
        base distribution and applies `transform()` for every transform in the
        list.
        """
        with torch.no_grad():
            x = self.base_dist.sample(sample_shape)
            x = self.bijector.forward(x, self.params)
            return x

    def rsample(
        self,
        sample_shape=torch.Size(),  # noqa: B008
    ) -> Tensor:
        """
        Generates a sample_shape shaped reparameterized sample or sample_shape
        shaped batch of reparameterized samples if the distribution parameters
        are batched. Samples first from base distribution and applies
        `transform()` for every transform in the list.
        """
        x = self.base_dist.rsample(sample_shape)
        x = self.bijector.forward(x, self.params)
        return x

    def log_prob(self, y):
        """
        Scores the sample by inverting the transform(s) and computing the score
        using the score of the base distribution and the log abs det jacobian.
        """
        event_dim = len(self.event_shape)

        x = self.bijector.inverse(y, self.params)
        log_prob = -_sum_rightmost(
            self.bijector.log_abs_det_jacobian(x, y, self.params),
            event_dim - self.bijector.event_dim,
        )
        log_prob = log_prob + _sum_rightmost(
            self.base_dist.log_prob(x),
            event_dim - len(self.base_dist.event_shape),
        )

        # self.cache(y,)

        return log_prob
