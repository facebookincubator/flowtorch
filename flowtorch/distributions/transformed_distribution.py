# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved
# SPDX-License-Identifier: MIT

import weakref
from typing import TYPE_CHECKING, Any, Dict, Optional

import torch
import torch.distributions as dist
from flowtorch.params.base import ParamsModule
from torch import Tensor
from torch.distributions.utils import _sum_rightmost

if TYPE_CHECKING:
    from flowtorch.bijectors.base import Bijector


class TransformedDistribution(dist.Distribution):
    _default_sample_shape = torch.Size()
    arg_constraints: Dict[str, dist.constraints.Constraint] = {}

    def __init__(
        self,
        base_distribution: dist.Distribution,
        bijector: "Bijector",
        params: Optional[ParamsModule],
        validate_args: Any = None,
    ) -> None:
        self.base_dist = base_distribution
        self._context = None

        if params is not None:
            self._params: Optional[weakref.ReferenceType[ParamsModule]] = weakref.ref(
                params
            )
        else:
            self._params = None

        self.bijector = bijector

        shape = self.base_dist.batch_shape + self.base_dist.event_shape
        event_dim = max(len(self.base_dist.event_shape), self.bijector.domain.event_dim)
        batch_shape = shape[: len(shape) - event_dim]
        event_shape = shape[len(shape) - event_dim :]
        super().__init__(batch_shape, event_shape, validate_args=validate_args)

    @property
    def params(self):
        if self._params is not None:
            # De-reference weak reference
            return self._params()
        else:
            return None

    def condition(self, context):
        self._context = context
        return self

    def sample(
        self,
        sample_shape: torch.Size = _default_sample_shape,
        context: Optional[torch.Tensor] = None,
    ) -> Tensor:
        """
        Generates a sample_shape shaped sample or sample_shape shaped batch of
        samples if the distribution parameters are batched. Samples first from
        base distribution and applies `transform()` for every transform in the
        list.
        """
        if context is None:
            context = self._context
        with torch.no_grad():
            x = self.base_dist.sample(sample_shape)
            x = self.bijector.forward(x, self.params, context)
            return x

    def rsample(
        self,
        sample_shape: torch.Size = _default_sample_shape,
        context: Optional[torch.Tensor] = None,
    ) -> Tensor:
        """
        Generates a sample_shape shaped reparameterized sample or sample_shape
        shaped batch of reparameterized samples if the distribution parameters
        are batched. Samples first from base distribution and applies
        `transform()` for every transform in the list.
        """
        if context is None:
            context = self._context
        x = self.base_dist.rsample(sample_shape)
        x = self.bijector.forward(x, self.params, context)
        return x

    def log_prob(
        self, y: torch.Tensor, context: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        """
        Scores the sample by inverting the transform(s) and computing the score
        using the score of the base distribution and the log abs det jacobian.
        """
        if context is None:
            context = self._context
        event_dim = len(self.event_shape)

        x = self.bijector.inverse(y, self.params, context)
        log_prob = -_sum_rightmost(
            self.bijector.log_abs_det_jacobian(x, y, self.params, context),
            event_dim - self.bijector.domain.event_dim,
        )
        log_prob = log_prob + _sum_rightmost(
            self.base_dist.log_prob(x),
            event_dim - len(self.base_dist.event_shape),
        )

        return log_prob
