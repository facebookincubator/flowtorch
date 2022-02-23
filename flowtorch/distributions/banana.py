# Copyright (c) Meta Platforms, Inc
from typing import Any, Dict, Optional, Union

import torch
import torch.distributions as dist
from torch.distributions import constraints
from torch.distributions.utils import _standard_normal


class Banana(dist.Distribution):
    """
    Banana shaped distribution.

    Let $$(z_1, z_2)\\sim\\mathcal{N}(0, (100, 0; 0, 1)) $$. Then the "banana
    distribution" is defined as, $$\\theta_1=z_1$$,
    $$\\theta_2=z_2+b\\cdot z^2_1-100b$$ for $$b=0.02$$.

    It can be easily shown that this is a volume-preserving transformation,
    that is, $$\\log(|\\det J|)=0$$.

    """

    support = constraints.real
    arg_constraints: Dict[str, dist.constraints.Constraint] = {}

    def __init__(self, validate_args: Any = None) -> None:
        d = 2
        batch_shape, event_shape = torch.Size([]), (d,)
        super(Banana, self).__init__(
            batch_shape, event_shape, validate_args=validate_args
        )

    def rsample(
        self,
        sample_shape: Union[torch.Tensor, torch.Size] = None,
        context: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        if not sample_shape:
            sample_shape = torch.Size()
        eps = _standard_normal(
            # TODO: Should this really be [0]?
            (sample_shape[0], 2),
            dtype=torch.float,
            device=torch.device("cpu"),
        )
        z = torch.zeros(eps.shape)
        z[..., 0] = 10.0 * eps[..., 0]
        z[..., 1] = eps[..., 1] + 0.02 * torch.square(z[..., 0]) - 100.0 * 0.02
        return z

    def log_prob(
        self, value: torch.Tensor, context: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        if self._validate_args:
            self._validate_sample(value)
        z = torch.zeros(value.shape)
        z[..., 0] = value[..., 0]
        z[..., 1] = value[..., 1] - 0.02 * torch.square(value[..., 0]) + 100.0 * 0.02

        # Since volume-preserving, no need to add log(det(|J|)) term
        log_prob = dist.Normal(0, 1.0).log_prob(z).sum(-1)
        return log_prob
