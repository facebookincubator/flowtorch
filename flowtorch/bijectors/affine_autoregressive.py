# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved
# SPDX-License-Identifier: MIT

from typing import cast, Optional, Tuple

import flowtorch.params
import torch
import torch.distributions.constraints as constraints
from flowtorch.bijectors.base import Bijector
from flowtorch.ops import clamp_preserve_gradients


class AffineAutoregressive(Bijector):
    domain = constraints.real_vector
    codomain = constraints.real_vector
    autoregressive = True

    def __init__(
        self,
        param_fn: Optional[flowtorch.params.DenseAutoregressive] = None,
        log_scale_min_clip: float = -5.0,
        log_scale_max_clip: float = 3.0,
        sigmoid_bias: float = 2.0,
        context_size: int = 0,
    ) -> None:
        # currently only DenseAutoregressive has a `permutation` buffer
        if not param_fn:
            param_fn = flowtorch.params.DenseAutoregressive()

        super().__init__(param_fn=param_fn)
        self.log_scale_min_clip = log_scale_min_clip
        self.log_scale_max_clip = log_scale_max_clip
        self.sigmoid_bias = sigmoid_bias
        self._context_size = context_size

    def _forward(
        self,
        x: torch.Tensor,
        context: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        # TODO: lift into type system using thunk, see similar pattern for
        # Param/ParamImpl
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
        # TODO: lift into type system using thunk, see similar pattern for
        # Param/ParamImpl
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
            )  # * 10
            mean = mean[..., idx]
            x[..., idx] = (y[..., idx] - mean) * inverse_scale

        return x

    def _log_abs_det_jacobian(
        self,
        x: torch.Tensor,
        y: torch.Tensor,
        context: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        # TODO: lift into type system using thunk, see similar pattern for
        # Param/ParamImpl
        params = self.params
        assert params is not None

        # Note: params will take care of caching "mean, log_scale, perm = params(x)"
        _, log_scale = params(x, context=context)
        log_scale = clamp_preserve_gradients(
            log_scale, self.log_scale_min_clip, self.log_scale_max_clip
        )
        return log_scale.sum(-1)

    def param_shapes(
        self, dist: torch.distributions.Distribution
    ) -> Tuple[torch.Size, torch.Size]:
        # A mean and log variance for every dimension of base distribution
        # TODO: Change this to reflect base dimension!
        return torch.Size([]), torch.Size([])
