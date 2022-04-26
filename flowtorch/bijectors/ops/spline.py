# Copyright (c) Meta Platforms, Inc

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
        params_fn: Optional[flowtorch.Lazy] = None,
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

        super().__init__(params_fn, shape=shape, context_shape=context_shape)

    def _forward(
        self, x: torch.Tensor, params: Optional[Sequence[torch.Tensor]]
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        y, log_detJ = self._op(x, params)
        return y, _sum_rightmost(log_detJ, self.domain.event_dim)

    def _inverse(
        self, y: torch.Tensor, params: Optional[Sequence[torch.Tensor]]
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        x_new, log_detJ = self._op(y, params, inverse=True)
        return x_new, _sum_rightmost(-log_detJ, self.domain.event_dim)

    def _log_abs_det_jacobian(
        self,
        x: torch.Tensor,
        y: torch.Tensor,
        params: Optional[Sequence[torch.Tensor]],
    ) -> torch.Tensor:
        _, log_detJ = self._op(x, params)
        return _sum_rightmost(log_detJ, self.domain.event_dim)

    def _op(
        self,
        input: torch.Tensor,
        params: Optional[Sequence[torch.Tensor]],
        inverse: bool = False,
        **kwargs: Any
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        assert params is not None

        lambdas: Optional[torch.Tensor] = None
        if self.order == "linear":
            widths, heights, derivatives, lambdas = params
            lambdas = torch.sigmoid(lambdas)
        else:
            widths, heights, derivatives = params

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
