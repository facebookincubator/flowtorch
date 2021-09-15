# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved
# SPDX-License-Identifier: MIT

import math
from typing import Optional

import flowtorch
import torch
from flowtorch.bijectors.base import Bijector


class AffineFixed(Bijector):
    r"""
    Elementwise bijector via the affine mapping :math:`\mathbf{y} = \mu +
    \sigma \otimes \mathbf{x}` where $\mu$ and $\sigma$ are fixed rather than
    learnable.
    """

    # TODO: Handle non-scalar loc and scale with correct broadcasting semantics
    def __init__(
        self,
        shape: torch.Size,
        params: Optional[flowtorch.Lazy] = None,
        context_size: int = 0,
        *,
        loc: float = 0.0,
        scale: float = 1.0
    ) -> None:
        super().__init__(shape, params, context_size)
        self.loc = loc
        self.scale = scale

    def _forward(
        self,
        x: torch.Tensor,
        context: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        return self.loc + self.scale * x

    def _inverse(
        self,
        y: torch.Tensor,
        context: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        return (y - self.loc) / self.scale

    def _log_abs_det_jacobian(
        self,
        x: torch.Tensor,
        y: torch.Tensor,
        context: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        return torch.full_like(x, math.log(abs(self.scale)))
