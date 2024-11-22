# Copyright (c) Meta Platforms, Inc

# pyre-unsafe

from collections.abc import Sequence
from typing import Optional, Tuple

import flowtorch.ops
import torch
import torch.distributions.constraints as constraints
import torch.nn.functional as F
from flowtorch.bijectors.fixed import Fixed


class Softplus(Fixed):
    r"""
    Elementwise bijector via the mapping :math:`\text{Softplus}(x) = \log(1 + \exp(x))`.
    """

    codomain = constraints.positive

    def _forward(
        self, x: torch.Tensor, params: Sequence[torch.Tensor] | None
    ) -> tuple[torch.Tensor, torch.Tensor | None]:
        y = F.softplus(x)
        ladj = self._log_abs_det_jacobian(x, y, params)
        return y, ladj

    def _inverse(
        self, y: torch.Tensor, params: Sequence[torch.Tensor] | None
    ) -> tuple[torch.Tensor, torch.Tensor | None]:
        x = flowtorch.ops.softplus_inv(y)
        ladj = self._log_abs_det_jacobian(x, y, params)
        return x, ladj

    def _log_abs_det_jacobian(
        self, x: torch.Tensor, y: torch.Tensor, params: Sequence[torch.Tensor] | None
    ) -> torch.Tensor:
        return -F.softplus(-x)
