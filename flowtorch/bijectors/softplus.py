# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved
# SPDX-License-Identifier: MIT

from typing import Optional

import flowtorch.params
import torch
import torch.distributions.constraints as constraints
import torch.nn.functional as F
from flowtorch.bijectors.base import Bijector


# TODO: Move to flowtorch.ops
def softplus_inv(y):
    return y + y.neg().expm1().neg().log()


class Softplus(Bijector):
    r"""
    Elementwise bijector via the mapping :math:`\text{Softplus}(x) = \log(1 + \exp(x))`.
    """
    codomain = constraints.positive

    def _forward(
        self,
        x: torch.Tensor,
        params: Optional[flowtorch.params.ParamsModule] = None,
        context: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        return F.softplus(x)

    def _inverse(
        self,
        y: torch.Tensor,
        params: Optional[flowtorch.params.ParamsModule] = None,
        context: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        return softplus_inv(y)

    def _log_abs_det_jacobian(
        self,
        x: torch.Tensor,
        y: torch.Tensor,
        params: Optional[flowtorch.params.ParamsModule] = None,
        context: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        return -F.softplus(-x)
