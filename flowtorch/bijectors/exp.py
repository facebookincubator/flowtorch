# Copyright (c) FlowTorch Development Team. All Rights Reserved
# SPDX-License-Identifier: MIT

from typing import Optional

import torch
import torch.distributions.constraints as constraints

import flowtorch


class Exp(flowtorch.Bijector):
    r"""
    Elementwise bijector via the mapping :math:`y = \exp(x)`.
    """
    codomain = constraints.positive

    def _forward(
        self,
        x: torch.Tensor,
        params: Optional[flowtorch.ParamsModule] = None,
        context: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        return x.exp()

    def _inverse(
        self,
        y: torch.Tensor,
        params: Optional[flowtorch.ParamsModule] = None,
        context: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        return y.log()

    def _log_abs_det_jacobian(
        self,
        x: torch.Tensor,
        y: torch.Tensor,
        params: Optional[flowtorch.ParamsModule] = None,
        context: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        return x
