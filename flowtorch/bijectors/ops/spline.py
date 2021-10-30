# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved
# SPDX-License-Identifier: MIT

# This implementation is adapted in part from:
# * https://github.com/tonyduan/normalizing-flows/blob/master/nf/flows.py;
# * https://github.com/hmdolatabadi/LRS_NF/blob/master/nde/transforms/
#       nonlinearities.py; and,
# * https://github.com/bayesiains/nsf/blob/master/nde/transforms/splines/
#       rational_quadratic.py
# under the MIT license.

from typing import Any, Optional, Sequence, Tuple

import flowtorch
import torch
import torch.nn.functional as F
from flowtorch.bijectors.base import Bijector
from flowtorch.ops import monotonic_rational_spline
from torch.distributions.utils import _sum_rightmost


class Spline(Bijector):
    def __init__(
        self,
        params: Optional[flowtorch.Lazy] = None,
        *,
        shape: torch.Size,
        context_shape: Optional[torch.Size] = None,
        count_bins: int = 8,
        bound: float = 3.0,
        order: str = "linear"
    ) -> None:
        if order not in ["linear", "quadratic"]:
            raise ValueError(
                "Keyword argument 'order' must be one of ['linear', \
'quadratic'], but '{}' was found!".format(
                    order
                )
            )

        self.count_bins = count_bins
        self.bound = bound
        self.order = order

        super().__init__(params, shape=shape, context_shape=context_shape)

    def _forward(
        self,
        x: torch.Tensor,
        context: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        y, _ = self._op(x, x, context)
        return y

    def _inverse(
        self,
        y: torch.Tensor,
        x: Optional[torch.Tensor] = None,
        context: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        x_new, _ = self._op(y, x, context=context, inverse=True)
        return x_new

    def _log_abs_det_jacobian(
        self,
        x: torch.Tensor,
        y: torch.Tensor,
        context: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        _, log_detJ = self._op(x, x, context)
        return _sum_rightmost(log_detJ, self.domain.event_dim)

    def _op(
        self,
        input: torch.Tensor,
        x: Optional[torch.Tensor] = None,
        context: Optional[torch.Tensor] = None,
        inverse: bool = False,
        **kwargs: Any
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        params = self.params
        assert params is not None

        if self.order == "linear":
            widths, heights, derivatives, lambdas = params(x, context=context)
            lambdas = torch.sigmoid(lambdas)
        else:
            widths, heights, derivatives = params(x, context=context)
            lambdas = None

        # Constrain parameters
        # TODO: Move to flowtorch.ops function?
        widths = F.softmax(widths, dim=-1)
        heights = F.softmax(heights, dim=-1)
        derivatives = F.softplus(derivatives)

        y, log_detJ = monotonic_rational_spline(
            input,
            widths,
            heights,
            derivatives,
            lambdas,
            bound=self.bound,
            inverse=inverse,
            **kwargs
        )
        return y, log_detJ

    def param_shapes(self, shape: torch.Size) -> Sequence[torch.Size]:
        s1 = torch.Size(shape + (self.count_bins,))
        s2 = torch.Size(shape + (self.count_bins - 1,))

        if self.order == "linear":
            return s1, s1, s2, s1
        else:
            return s1, s1, s2
