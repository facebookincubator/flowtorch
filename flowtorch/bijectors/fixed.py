# Copyright (c) FlowTorch Development Team. All Rights Reserved
# SPDX-License-Identifier: MIT
import weakref
from typing import Optional, Sequence, Tuple, Union, cast

import torch
import torch.distributions
from torch.distributions import constraints

import flowtorch
import flowtorch.distributions


class Fixed(flowtorch.Bijector):
    def forward(
        self,
        x: torch.Tensor,
        params: Optional["flowtorch.ParamsModule"],
        context: torch.Tensor,
    ) -> torch.Tensor:
        assert context.shape == (self._context_size,)
        assert params is None
        return self._forward(x, None, context)

    def inverse(
        self,
        y: torch.Tensor,
        params: Optional["flowtorch.ParamsModule"],
        context: torch.Tensor,
    ) -> torch.Tensor:
        assert context.shape == (self._context_size,)
        assert params is None
        return self._inverse(y, None, context)

    def log_abs_det_jacobian(
        self,
        x: torch.Tensor,
        y: torch.Tensor,
        params: Optional["flowtorch.ParamsModule"],
        context: torch.Tensor,
    ) -> torch.Tensor:
        """
        Computes the log det jacobian `log |dy/dx|` given input and output.
        By default, assumes a volume preserving bijection.
        """
        assert params is None
        return self._log_abs_det_jacobian(x, y, None, context)

    def param_shapes(
        self, dist: torch.distributions.Distribution
    ) -> Sequence[torch.Size]:
        """
        Given a base distribution, calculate the parameters for the transformation
        of that distribution under this bijector. By default, no parameters are
        set.
        """
        return (torch.Size([]),)
