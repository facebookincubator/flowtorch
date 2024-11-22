# Copyright (c) Meta Platforms, Inc

# pyre-unsafe

from collections.abc import Sequence
from typing import Optional, Tuple

import torch
import torch.distributions.constraints as constraints
import torch.nn.functional as F
from flowtorch.bijectors.fixed import Fixed
from flowtorch.ops import eps


class ELU(Fixed):
    codomain = constraints.greater_than(-1.0)

    # TODO: Setting the alpha value of ELU as __init__ argument

    def _forward(
        self, x: torch.Tensor, params: Sequence[torch.Tensor] | None
    ) -> tuple[torch.Tensor, torch.Tensor | None]:
        y = F.elu(x)
        ladj = self._log_abs_det_jacobian(x, y, params)
        return y, ladj

    def _inverse(
        self, y: torch.Tensor, params: Sequence[torch.Tensor] | None
    ) -> tuple[torch.Tensor, torch.Tensor | None]:
        x = torch.max(y, torch.zeros_like(y)) + torch.min(
            torch.log1p(y + eps), torch.zeros_like(y)
        )
        ladj = self._log_abs_det_jacobian(x, y, params)
        return x, ladj

    def _log_abs_det_jacobian(
        self, x: torch.Tensor, y: torch.Tensor, params: Sequence[torch.Tensor] | None
    ) -> torch.Tensor:
        return -F.relu(-x)
