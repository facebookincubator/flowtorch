# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved
# SPDX-License-Identifier: MIT
from typing import Optional

import flowtorch.distributions
import flowtorch.params
import torch
import torch.distributions
from flowtorch.bijectors.base import Bijector


class VolumePreserving(Bijector):
    def _log_abs_det_jacobian(
        self,
        x: torch.Tensor,
        y: torch.Tensor,
        params: Optional[flowtorch.params.ParamsModule] = None,
        context: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        # TODO: Confirm that this should involve `x`/`self.domain` and not
        # `y`/`self.codomain`
        return torch.zeros(
            x.size()[: -self.domain.event_dim],
            dtype=x.dtype,
            layout=x.layout,
            device=x.device,
        )
