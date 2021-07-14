# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved
# SPDX-License-Identifier: MIT
from typing import Optional, Sequence, Union

import flowtorch.distributions
from flowtorch.params.base import Params, ParamsModule, ParamsModuleList
import torch
import torch.distributions
from flowtorch.bijectors.base import Bijector


class Fixed(Bijector):
    def forward(
        self,
        x: torch.Tensor,
        params: Optional[Union[ParamsModule, ParamsModuleList]],
        context: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        assert context is None or context.shape == (self._context_size,)
        assert params is None
        return self._forward(x, None, context)

    def inverse(
        self,
        y: torch.Tensor,
        params: Optional[Union[ParamsModule, ParamsModuleList]],
        context: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        assert context is None or context.shape == (self._context_size,)
        assert params is None
        return self._inverse(y, None, context)

    def log_abs_det_jacobian(
        self,
        x: torch.Tensor,
        y: torch.Tensor,
        params: Optional[Union[ParamsModule, ParamsModuleList]],
        context: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """
        Computes the log det jacobian `log |dy/dx|` given input and output.
        By default, assumes a volume preserving bijection.
        """
        assert context is None or context.shape == (self._context_size,)
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
