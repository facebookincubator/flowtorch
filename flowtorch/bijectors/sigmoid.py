# Copyright (c) Meta Platforms, Inc

# pyre-unsafe

from collections.abc import Sequence
from typing import Optional, Tuple

import torch
import torch.distributions.constraints as constraints
import torch.nn.functional as F
from flowtorch.bijectors.fixed import Fixed
from flowtorch.ops import clipped_sigmoid


class Sigmoid(Fixed):
    codomain = constraints.unit_interval

    def _forward(
        self, x: torch.Tensor, params: Sequence[torch.Tensor] | None
    ) -> tuple[torch.Tensor, torch.Tensor | None]:
        y = clipped_sigmoid(x)
        ladj = self._log_abs_det_jacobian(x, y, params)
        return y, ladj

    def _inverse(
        self, y: torch.Tensor, params: Sequence[torch.Tensor] | None
    ) -> tuple[torch.Tensor, torch.Tensor | None]:
        finfo = torch.finfo(y.dtype)
        y = y.clamp(min=finfo.tiny, max=1.0 - finfo.eps)
        x = y.log() - torch.log1p(-y)
        ladj = self._log_abs_det_jacobian(x, y, params)
        return x, ladj

    def _log_abs_det_jacobian(
        self, x: torch.Tensor, y: torch.Tensor, params: Sequence[torch.Tensor] | None
    ) -> torch.Tensor:
        return -F.softplus(-x) - F.softplus(x)
