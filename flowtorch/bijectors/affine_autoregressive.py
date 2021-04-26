# Copyright (c) FlowTorch Development Team. All Rights Reserved
# SPDX-License-Identifier: MIT

from typing import Optional, Tuple

import torch
import torch.distributions.constraints as constraints

import flowtorch
import flowtorch.params
from flowtorch.utils import clamp_preserve_gradients


class AffineAutoregressive(flowtorch.Bijector):
    domain = constraints.real_vector
    codomain = constraints.real_vector
    autoregressive = True

    def __init__(
        self,
        param_fn: Optional[flowtorch.Params] = None,
        log_scale_min_clip: float = -5.0,
        log_scale_max_clip: float = 3.0,
        sigmoid_bias: float = 2.0,
        context_size: int = 0,
    ) -> None:
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
        params: Optional[flowtorch.ParamsModule],
        context: Optional[torch.Tensor],
    ) -> torch.Tensor:
        assert isinstance(params, flowtorch.ParamsModule)
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
        params: Optional[flowtorch.ParamsModule],
        context: Optional[torch.Tensor],
    ) -> torch.Tensor:
        assert isinstance(params, flowtorch.ParamsModule)
        x = torch.zeros_like(y)

        # NOTE: Inversion is an expensive operation that scales in the
        # dimension of the input
        for idx in params.permutation:  # type: ignore
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
        params: Optional[flowtorch.ParamsModule],
        context: Optional[torch.Tensor],
    ) -> torch.Tensor:
        assert isinstance(params, flowtorch.ParamsModule)
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
