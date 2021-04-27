# Copyright (c) FlowTorch Development Team. All Rights Reserved
# SPDX-License-Identifier: MIT

from typing import Optional

import torch
import torch.distributions.constraints as constraints

import flowtorch


class Power(flowtorch.Bijector):
    r"""
    Elementwise bijector via the mapping :math:`y = x^{\text{exponent}}`.
    """
    domain = constraints.positive
    codomain = constraints.positive

    # TODO: Tensor valued exponents and corresponding determination of event_dim
    def __init__(
        self,
        exponent: float = 2.0,
    ) -> None:
        super().__init__(param_fn=None)
        self.exponent = exponent

    def _forward(
        self,
        x: torch.Tensor,
        params: Optional[flowtorch.ParamsModule] = None,
        context: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        return x.pow(self.exponent)

    def _inverse(
        self,
        y: torch.Tensor,
        params: Optional[flowtorch.ParamsModule] = None,
        context: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        return y.pow(1 / self.exponent)

    def _log_abs_det_jacobian(
        self,
        x: torch.Tensor,
        y: torch.Tensor,
        params: Optional[flowtorch.ParamsModule] = None,
        context: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        return (self.exponent * y / x).abs().log()
