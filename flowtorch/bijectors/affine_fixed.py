# Copyright (c) FlowTorch Development Team. All Rights Reserved
# SPDX-License-Identifier: MIT

import math
from typing import Optional

import torch

import flowtorch


class AffineFixed(flowtorch.Bijector):
    r"""
    Elementwise bijector via the affine mapping :math:`\mathbf{y} = \mu +
    \sigma \otimes \mathbf{x}` where $\mu$ and $\sigma$ are fixed rather than
    learnable.
    """

    # TODO: Handle non-scalar loc and scale with correct broadcasting semantics
    def __init__(self, loc, scale) -> None:
        super().__init__(param_fn=None)
        self.loc = loc
        self.scale = scale

    def _forward(
        self,
        x: torch.Tensor,
        params: Optional[flowtorch.ParamsModule] = None,
        context: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        return self.loc + self.scale * x

    def _inverse(
        self,
        y: torch.Tensor,
        params: Optional[flowtorch.ParamsModule] = None,
        context: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        return (y - self.loc) / self.scale

    def _log_abs_det_jacobian(
        self,
        x: torch.Tensor,
        y: torch.Tensor,
        params: Optional[flowtorch.ParamsModule] = None,
        context: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        return torch.full_like(x, math.log(abs(self.scale)))
