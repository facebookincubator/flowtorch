# Copyright (c) Meta Platforms, Inc
import math
from typing import Any, Dict, Optional, Union

import torch
import torch.distributions as dist
from torch.distributions import constraints


class Lollipop(dist.Distribution):
    """
    A "lollipop" shaped distribution.

    The lollipop distribution is the mixture of a straight line and a uniform
    distribution over a circle.

    """

    support = constraints.real
    arg_constraints: Dict[str, dist.constraints.Constraint] = {}

    def __init__(self, validate_args: Any = None) -> None:
        d = 2
        batch_shape, event_shape = torch.Size([]), (d,)
        super(Lollipop, self).__init__(
            batch_shape, event_shape, validate_args=validate_args
        )

    def rsample(
        self,
        sample_shape: torch.Size = None,
        context: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        if not sample_shape:
            sample_shape = torch.Size()

        # Approx. 95% of samples belong to lolly and 5% to stick
        count_samples = sum(list(sample_shape))
        count_circle = int(0.95 * count_samples)
        count_stick = count_samples - count_circle

        # Sample uniformly from a circle of radius sqrt(0.5) centered at (2, 2)
        radius = torch.rand(size=(count_circle,))
        radians = 2.0 * math.pi * torch.rand(size=(count_circle,))
        x_circle = 2.0 + radius**0.5 * torch.sin(radians)
        y_circle = 2.0 + radius**0.5 * torch.cos(radians)

        # Sample from a line with slope 1 from (0, 2 - 1 / np.sqrt(2))
        stick = (2.0 - 1.0 / math.sqrt(2.0)) * torch.rand(size=(count_stick,))

        x = torch.cat([x_circle, stick], dim=-1)
        y = torch.cat([y_circle, stick], dim=-1)
        z = torch.reshape(torch.stack([x, y], dim=-1), sample_shape + (2,))
        return z

    def log_prob(
        self, value: torch.Tensor, context: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        # TODO: Implement this class as a mixture distribution to simplify
        # and so we have valid log_prob
        raise NotImplementedError("lollipop.log_prob not implemented")
