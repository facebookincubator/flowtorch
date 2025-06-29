# Copyright (c) Meta Platforms, Inc

# pyre-unsafe
from typing import Any, Dict, Optional, Union

import torch
import torch.distributions as dist
from torch.distributions import constraints
from torch.distributions.utils import _standard_normal


class NealsFunnel(dist.Distribution):
    """
    Neal's funnel.
    p(x,y) = N(y|0,3) N(x|0,exp(y/2))
    """

    support = constraints.real
    arg_constraints: dict[str, dist.constraints.Constraint] = {}

    def __init__(self, validate_args: Any = None) -> None:
        d = 2
        batch_shape, event_shape = torch.Size([]), (d,)
        # pyre-fixme[6]: For 2nd argument expected `Size` but got `Tuple[int]`.
        super().__init__(batch_shape, event_shape, validate_args=validate_args)

    # pyre-fixme[14]: `rsample` overrides method defined in `Distribution`
    #  inconsistently.
    def rsample(
        self,
        sample_shape: torch.Tensor | torch.Size | None = None,
        context: torch.Tensor | None = None,
    ) -> torch.Tensor:
        if not sample_shape:
            sample_shape = torch.Size()
        eps = _standard_normal(
            # pyre-fixme[6]: For 1st argument expected `Sequence[Union[int,
            #  SymInt]]` but got `Tuple[Union[int, Tensor], int]`.
            (sample_shape[0], 2),
            dtype=torch.float,
            device=torch.device("cpu"),
        )
        z = torch.zeros(eps.shape)
        z[..., 1] = torch.tensor(3.0) * eps[..., 1]
        z[..., 0] = torch.exp(z[..., 1] / 2.0) * eps[..., 0]
        return z

    def log_prob(
        self, value: torch.Tensor, context: torch.Tensor | None = None
    ) -> torch.Tensor:
        if self._validate_args:
            self._validate_sample(value)
        x = value[..., 0]
        y = value[..., 1]

        log_prob = dist.Normal(0, 3).log_prob(y)
        log_prob += dist.Normal(0, torch.exp(y / 2)).log_prob(x)

        return log_prob
