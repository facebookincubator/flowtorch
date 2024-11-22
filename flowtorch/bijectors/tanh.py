# Copyright (c) Meta Platforms, Inc

# pyre-unsafe

import math
from collections.abc import Sequence
from typing import Optional, Tuple

import torch
import torch.distributions.constraints as constraints
import torch.nn.functional as F
from flowtorch.bijectors.fixed import Fixed


class Tanh(Fixed):
    r"""
    Transform via the mapping :math:`y = \tanh(x)`.
    """

    codomain = constraints.interval(-1.0, 1.0)

    def _forward(
        self, x: torch.Tensor, params: Sequence[torch.Tensor] | None
    ) -> tuple[torch.Tensor, torch.Tensor | None]:
        y = torch.tanh(x)
        ladj = self._log_abs_det_jacobian(x, y, params)
        return y, ladj

    def _inverse(
        self, y: torch.Tensor, params: Sequence[torch.Tensor] | None
    ) -> tuple[torch.Tensor, torch.Tensor | None]:
        x = torch.atanh(y)
        ladj = self._log_abs_det_jacobian(x, y, params)
        return x, ladj

    def _log_abs_det_jacobian(
        self, x: torch.Tensor, y: torch.Tensor, params: Sequence[torch.Tensor] | None
    ) -> torch.Tensor:
        # pyre-fixme[7]: Expected `Tensor` but got `float`.
        return 2.0 * (math.log(2.0) - x - F.softplus(-2.0 * x))
