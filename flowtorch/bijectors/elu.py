# Copyright (c) Meta Platforms, Inc

from typing import Optional, Sequence, Tuple

import torch
import torch.distributions.constraints as constraints
import torch.nn.functional as F
from flowtorch.bijectors.fixed import Fixed
from flowtorch.ops import eps


class ELU(Fixed):
    codomain = constraints.greater_than(-1.0)

    # TODO: Setting the alpha value of ELU as __init__ argument

    def _forward(
        self, x: torch.Tensor, params: Optional[Sequence[torch.Tensor]]
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
        y = F.elu(x)
        ladj = self._log_abs_det_jacobian(x, y, params)
        return y, ladj

    def _inverse(
        self, y: torch.Tensor, params: Optional[Sequence[torch.Tensor]]
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
        x = torch.max(y, torch.zeros_like(y)) + torch.min(
            torch.log1p(y + eps), torch.zeros_like(y)
        )
        ladj = self._log_abs_det_jacobian(x, y, params)
        return x, ladj

    def _log_abs_det_jacobian(
        self, x: torch.Tensor, y: torch.Tensor, params: Optional[Sequence[torch.Tensor]]
    ) -> torch.Tensor:
        return -F.relu(-x)
