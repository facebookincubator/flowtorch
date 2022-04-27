# Copyright (c) Meta Platforms, Inc

from typing import Any, Dict, Optional, Union

import flowtorch
import torch
import torch.distributions as dist
from torch import Tensor
from torch.distributions.utils import _sum_rightmost


class Flow(torch.nn.Module, dist.Distribution, metaclass=flowtorch.LazyMeta):
    _default_sample_shape = torch.Size()
    arg_constraints: Dict[str, dist.constraints.Constraint] = {}

    def __init__(
        self,
        base_dist: dist.Distribution,
        bijector: flowtorch.Lazy,
        validate_args: Any = None,
    ) -> None:
        torch.nn.Module.__init__(self)

        self.base_dist = base_dist
        self._context: Optional[torch.Tensor] = None
        self.bijector = bijector(shape=base_dist.event_shape)

        # TODO: Confirm that the following logic works. Shouldn't it use
        # .domain and .codomain?? Infer shape from constructed self.bijector
        shape = (
            self.base_dist.batch_shape + self.base_dist.event_shape  # pyre-ignore[16]
        )
        event_dim = self.bijector.domain.event_dim  # type: ignore
        event_dim = max(event_dim, len(self.base_dist.event_shape))
        batch_shape = shape[: len(shape) - event_dim]
        event_shape = shape[len(shape) - event_dim :]

        dist.Distribution.__init__(
            self, batch_shape, event_shape, validate_args=validate_args
        )

    def condition(self, context: torch.Tensor) -> "Flow":
        self._context = context
        return self

    def sample(
        self,
        sample_shape: Union[Tensor, torch.Size] = _default_sample_shape,
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
            x = self.bijector.forward(x, context)  # type: ignore
            return x

    def rsample(
        self,
        sample_shape: Union[Tensor, torch.Size] = _default_sample_shape,
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
        x = self.bijector.forward(x, context)  # type: ignore
        return x

    def rnormalize(
        self, value: torch.Tensor, context: Optional[torch.Tensor] = None
    ) -> Tensor:
        """
        Push a tensor through the normalizing direction of the flow where
        we can take autodiff gradients on the bijector.
        """
        if context is None:
            context = self._context

        return self.bijector.inverse(value, context)  # type: ignore

    def normalize(
        self, value: torch.Tensor, context: Optional[torch.Tensor] = None
    ) -> Tensor:
        """
        Push a tensor through the normalizing direction of the flow and
        block autodiff gradients on the bijector.
        """
        with torch.no_grad():
            return self.rnormalize(value, context)

    def log_prob(
        self, value: torch.Tensor, context: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        """
        Scores the sample by inverting the transform(s) and computing the score
        using the score of the base distribution and the log abs det jacobian.
        """
        if context is None:
            context = self._context
        event_dim = len(self.event_shape)  # pyre-ignore[16]

        x = self.bijector.inverse(value, context)  # type: ignore
        log_prob = -_sum_rightmost(
            self.bijector.log_abs_det_jacobian(x, value, context),  # type: ignore
            event_dim - self.bijector.domain.event_dim,  # type: ignore
        )
        log_prob = log_prob + _sum_rightmost(
            self.base_dist.log_prob(x),
            event_dim - len(self.base_dist.event_shape),  # pyre-ignore[16]
        )

        return log_prob
