# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved
# SPDX-License-Identifier: MIT

from typing import cast, Optional, Tuple

import flowtorch
import flowtorch.parameters
import torch
import torch.distributions.constraints as constraints
from flowtorch.bijectors.base import Bijector
from flowtorch.ops import clamp_preserve_gradients
from torch.distributions.utils import _sum_rightmost


class AffineAutoregressive(Bijector):
    # "Default" event shape is to operate on vectors
    domain = constraints.real_vector
    codomain = constraints.real_vector

    # TODO: Remove when bijector/params type system is implemented
    autoregressive = True

    def __init__(
        self,
        shape: torch.Size,
        params: Optional[flowtorch.Lazy] = None,
        context_size: int = 0,
        *,
        log_scale_min_clip: float = -5.0,
        log_scale_max_clip: float = 3.0,
        sigmoid_bias: float = 2.0,
    ) -> None:
        # Event shape is determined by `shape` argument
        self.domain = constraints.independent(constraints.real, len(shape))
        self.codomain = constraints.independent(constraints.real, len(shape))

        # currently only DenseAutoregressive has a `permutation` buffer
        if not params:
            params = flowtorch.parameters.DenseAutoregressive()  # type: ignore

        super().__init__(shape, params, context_size)
        self.log_scale_min_clip = log_scale_min_clip
        self.log_scale_max_clip = log_scale_max_clip
        self.sigmoid_bias = sigmoid_bias

    def _forward(
        self,
        x: torch.Tensor,
        context: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        params = self.params
        assert params is not None

        mean, log_scale = params(x, context=context)
        log_scale = clamp_preserve_gradients(
            log_scale, self.log_scale_min_clip, self.log_scale_max_clip
        )
        scale = torch.exp(log_scale)
        y = scale * x + mean
        return y

    def _inverse(
        self,
        y: torch.Tensor,
        context: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        params = self.params
        assert params is not None

        x = torch.zeros_like(y)
        # NOTE: Inversion is an expensive operation that scales in the
        # dimension of the input
        permutation = (
            params.permutation
        )  # TODO: type-safe named buffer (e.g. "permutation") access
        for idx in cast(torch.LongTensor, permutation):
            mean, log_scale = params(x.clone(), context=context)
            inverse_scale = torch.exp(
                -clamp_preserve_gradients(
                    log_scale[..., idx],
                    min=self.log_scale_min_clip,
                    max=self.log_scale_max_clip,
                )
            )
            mean = mean[..., idx]
            x[..., idx] = (y[..., idx] - mean) * inverse_scale

        return x

    def _log_abs_det_jacobian(
        self,
        x: torch.Tensor,
        y: torch.Tensor,
        context: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        params = self.params
        assert params is not None

        # Note: params will take care of caching "mean, log_scale, perm = params(x)"
        _, log_scale = params(x, context=context)
        log_scale = clamp_preserve_gradients(
            log_scale, self.log_scale_min_clip, self.log_scale_max_clip
        )
        return _sum_rightmost(log_scale, self.domain.event_dim)

    def param_shapes(self, shape: torch.Size) -> Tuple[torch.Size, torch.Size]:
        # A mean and log variance for every dimension of the event shape
        return shape, shape
