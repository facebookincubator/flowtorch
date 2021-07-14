# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved
# SPDX-License-Identifier: MIT

from typing import Optional, Union

from flowtorch.params.base import Params, ParamsModule, ParamsModuleList
import torch
import torch.distributions.constraints as constraints
from flowtorch.bijectors.base import Bijector


class Exp(Bijector):
    r"""
    Elementwise bijector via the mapping :math:`y = \exp(x)`.
    """
    codomain = constraints.positive

    def _forward(
        self,
        x: torch.Tensor,
        params: Optional[Union[ParamsModule, ParamsModuleList]] = None,
        context: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        return x.exp()

    def _inverse(
        self,
        y: torch.Tensor,
        params: Optional[Union[ParamsModule, ParamsModuleList]] = None,
        context: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        return y.log()

    def _log_abs_det_jacobian(
        self,
        x: torch.Tensor,
        y: torch.Tensor,
        params: Optional[Union[ParamsModule, ParamsModuleList]] = None,
        context: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        return x
