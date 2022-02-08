# Copyright (c) Meta Platforms, Inc

from typing import Optional, Sequence

import torch
import torch.distributions
from flowtorch.bijectors.base import Bijector


class VolumePreserving(Bijector):
    def _log_abs_det_jacobian(
        self, x: torch.Tensor, y: torch.Tensor, params: Optional[Sequence[torch.Tensor]]
    ) -> torch.Tensor:
        # TODO: Confirm that this should involve `x`/`self.domain` and not
        # `y`/`self.codomain`
        return torch.zeros(
            x.size()[: -self.domain.event_dim],
            dtype=x.dtype,
            layout=x.layout,  # pyre-ignore[16]
            device=x.device,
        )
