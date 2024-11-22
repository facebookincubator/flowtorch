# Copyright (c) Meta Platforms, Inc

# pyre-unsafe

from collections.abc import Sequence
from typing import Optional, Tuple

import flowtorch
import torch
import torch.nn.functional as F
from flowtorch.bijectors.base import Bijector
from flowtorch.ops import clamp_preserve_gradients
from torch.distributions.utils import _sum_rightmost


class Affine(Bijector):
    r"""
    Affine mapping :math:`\mathbf{y} = \mu + \sigma \otimes \mathbf{x}` where
    $\mu$ and $\sigma$ are learnable parameters.

    """

    def __init__(
        self,
        params_fn: flowtorch.Lazy | None = None,
        *,
        shape: torch.Size,
        context_shape: torch.Size | None = None,
        clamp_values: bool = False,
        log_scale_min_clip: float = -5.0,
        log_scale_max_clip: float = 3.0,
        scale_fn: str = "softplus",
    ) -> None:
        super().__init__(params_fn, shape=shape, context_shape=context_shape)
        self.clamp_values = clamp_values
        self.log_scale_min_clip = log_scale_min_clip
        self.log_scale_max_clip = log_scale_max_clip
        self.scale_fn = scale_fn

    def _scale_fn(
        self, unbounded_scale: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor]:
        # NOTE: Need to hardcode log(f(x)) for numerical stability
        if self.scale_fn == "softplus":
            scale = F.softplus(unbounded_scale)
            log_scale = torch.log(scale)
        elif self.scale_fn == "exp":
            scale = torch.exp(unbounded_scale)
            log_scale = unbounded_scale
        elif self.scale_fn == "sigmoid":
            scale = torch.sigmoid(unbounded_scale)
            log_scale = F.logsigmoid(unbounded_scale)
        else:
            raise ValueError(f"Unknown scale function: {self.scale_fn}")

        return scale, log_scale

    def _inv_scale_fn(
        self, unbounded_scale: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor]:
        # NOTE: Need to hardcode 1./log(f(x)) for numerical stability
        if self.scale_fn == "softplus":
            scale = F.softplus(unbounded_scale)
            inverse_scale = F.softplus(unbounded_scale).reciprocal()
            log_scale = torch.log(scale)
        elif self.scale_fn == "exp":
            inverse_scale = torch.exp(-unbounded_scale)
            log_scale = unbounded_scale
        elif self.scale_fn == "sigmoid":
            inverse_scale = torch.exp(-unbounded_scale) + 1.0
            log_scale = F.logsigmoid(unbounded_scale)
        else:
            raise ValueError(f"Unknown scale function: {self.scale_fn}")

        return inverse_scale, log_scale

    def _forward(
        self, x: torch.Tensor, params: Sequence[torch.Tensor] | None
    ) -> tuple[torch.Tensor, torch.Tensor]:
        assert params is not None

        mean, unbounded_scale = params
        if self.clamp_values:
            unbounded_scale = clamp_preserve_gradients(
                unbounded_scale, self.log_scale_min_clip, self.log_scale_max_clip
            )

        scale, log_scale = self._scale_fn(unbounded_scale)
        y = scale * x + mean
        return y, _sum_rightmost(log_scale, self.domain.event_dim)

    def _inverse(
        self, y: torch.Tensor, params: Sequence[torch.Tensor] | None
    ) -> tuple[torch.Tensor, torch.Tensor]:
        assert params is not None

        mean, unbounded_scale = params
        if self.clamp_values:
            unbounded_scale = clamp_preserve_gradients(
                unbounded_scale, self.log_scale_min_clip, self.log_scale_max_clip
            )

        inverse_scale, log_scale = self._inv_scale_fn(unbounded_scale)
        x_new = (y - mean) * inverse_scale
        return x_new, _sum_rightmost(log_scale, self.domain.event_dim)

    def _log_abs_det_jacobian(
        self,
        x: torch.Tensor,
        y: torch.Tensor,
        params: Sequence[torch.Tensor] | None,
    ) -> torch.Tensor:
        assert params is not None

        _, unbounded_scale = params
        if self.clamp_values:
            unbounded_scale = clamp_preserve_gradients(
                unbounded_scale, self.log_scale_min_clip, self.log_scale_max_clip
            )
        _, log_scale = self._scale_fn(unbounded_scale)

        return _sum_rightmost(log_scale, self.domain.event_dim)

    def param_shapes(self, shape: torch.Size) -> tuple[torch.Size, torch.Size]:
        # A mean and log variance for every dimension of the event shape
        return shape, shape
