# Copyright (c) Meta Platforms, Inc

from typing import Optional, Sequence, Tuple, Dict, Callable

import flowtorch

import torch
from flowtorch.bijectors.base import Bijector
from flowtorch.ops import clamp_preserve_gradients
from torch.distributions.utils import _sum_rightmost

_DEFAULT_POSITIVE_BIASES = {
    "softplus": torch.expm1(torch.ones(1)).log().item(),
    "exp": 0.0,
}
_POSITIVE_MAPS: Dict[str, Callable[[torch.Tensor], torch.Tensor]] = {
    "softplus": torch.nn.functional.softplus,
    "sigmoid": torch.sigmoid,
    "exp": torch.exp,
}


class Affine(Bijector):
    r"""
    Affine mapping :math:`\mathbf{y} = \mu + \sigma \otimes \mathbf{x}` where
    $\mu$ and $\sigma$ are learnable parameters.

    """

    def __init__(
        self,
        params_fn: Optional[flowtorch.Lazy] = None,
        *,
        shape: torch.Size,
        context_shape: Optional[torch.Size] = None,
        clamp_values: bool = False,
        log_scale_min_clip: float = -5.0,
        log_scale_max_clip: float = 3.0,
        sigmoid_bias: float = 2.0,
        positive_map: str = "softplus",
        positive_bias: Optional[float] = None,
    ) -> None:
        super().__init__(params_fn, shape=shape, context_shape=context_shape)
        self.clamp_values = clamp_values
        self.log_scale_min_clip = log_scale_min_clip
        self.log_scale_max_clip = log_scale_max_clip
        self.sigmoid_bias = sigmoid_bias
        if positive_bias is None:
            positive_bias = _DEFAULT_POSITIVE_BIASES[positive_map]
        self.positive_bias = positive_bias
        if positive_map not in _POSITIVE_MAPS:
            raise RuntimeError(f"Unknwon positive map {positive_map}")
        self._positive_map = _POSITIVE_MAPS[positive_map]
        self._exp_map = self._positive_map is torch.exp and self.positive_bias == 0

    def positive_map(self, x: torch.Tensor) -> torch.Tensor:
        return self._positive_map(x + self.positive_bias)

    def _forward(
        self, x: torch.Tensor, params: Optional[Sequence[torch.Tensor]]
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        assert params is not None

        mean, unbounded_scale = params
        if self.clamp_values:
            unbounded_scale = clamp_preserve_gradients(
                unbounded_scale, self.log_scale_min_clip, self.log_scale_max_clip
            )
        scale = self.positive_map(unbounded_scale)
        log_scale = scale.log() if not self._exp_map else unbounded_scale
        y = scale * x + mean
        return y, _sum_rightmost(log_scale, self.domain.event_dim)

    def _inverse(
        self, y: torch.Tensor, params: Optional[Sequence[torch.Tensor]]
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        assert (
            params is not None
        ), f"{self.__class__.__name__}._inverse got no parameters"

        mean, unbounded_scale = params
        if self.clamp_values:
            unbounded_scale = clamp_preserve_gradients(
                unbounded_scale, self.log_scale_min_clip, self.log_scale_max_clip
            )

        if not self._exp_map:
            inverse_scale = self.positive_map(unbounded_scale).reciprocal()
            log_scale = inverse_scale.log()
        else:
            inverse_scale = torch.exp(-unbounded_scale)
            log_scale = unbounded_scale

        x_new = (y - mean) * inverse_scale
        return x_new, _sum_rightmost(log_scale, self.domain.event_dim)

    def _log_abs_det_jacobian(
        self,
        x: torch.Tensor,
        y: torch.Tensor,
        params: Optional[Sequence[torch.Tensor]],
    ) -> torch.Tensor:
        assert params is not None

        _, unbounded_scale = params
        if self.clamp_values:
            unbounded_scale = clamp_preserve_gradients(
                unbounded_scale, self.log_scale_min_clip, self.log_scale_max_clip
            )
        log_scale = (
            self.positive_map(unbounded_scale).log()
            if not self._exp_map
            else unbounded_scale
        )
        return _sum_rightmost(log_scale, self.domain.event_dim)

    def param_shapes(self, shape: torch.Size) -> Tuple[torch.Size, torch.Size]:
        # A mean and log variance for every dimension of the event shape
        return shape, shape
