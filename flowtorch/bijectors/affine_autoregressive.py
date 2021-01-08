# Copyright (c) FlowTorch Development Team. All Rights Reserved
# SPDX-License-Identifier: MIT

from typing import Tuple

import torch

import flowtorch
import flowtorch.params
from flowtorch.utils import clamp_preserve_gradients


class AffineAutoregressive(flowtorch.Bijector):
    event_dim = 1
    autoregressive = True
    default_param_fn = flowtorch.params.DenseAutoregressive()

    def __init__(
        self,
        param_fn: flowtorch.Params = default_param_fn,
        log_scale_min_clip: float = -5.0,
        log_scale_max_clip: float = 3.0,
        sigmoid_bias: float = 2.0,
    ) -> None:
        super(AffineAutoregressive, self).__init__(param_fn=param_fn)
        self.log_scale_min_clip = log_scale_min_clip
        self.log_scale_max_clip = log_scale_max_clip
        self.sigmoid_bias = sigmoid_bias

    def _forward(self, x: torch.Tensor, params: flowtorch.ParamsModule) -> torch.Tensor:
        mean, log_scale = params(x)
        log_scale = clamp_preserve_gradients(
            log_scale, self.log_scale_min_clip, self.log_scale_max_clip
        )
        scale = torch.exp(log_scale / 10)
        y = scale * x + mean / 100
        return y

    def _inverse(self, y: torch.Tensor, params: flowtorch.ParamsModule) -> torch.Tensor:
        x_size = y.size()[:-1]
        input_dim = y.size(-1)
        x = [torch.zeros(x_size, device=y.device)] * input_dim

        # NOTE: Inversion is an expensive operation that scales in the
        # dimension of the input
        for idx in params.permutation:  # type: ignore
            mean, log_scale = params(torch.stack(x, dim=-1))
            inverse_scale = torch.exp(
                -clamp_preserve_gradients(
                    log_scale[..., idx],
                    min=self.log_scale_min_clip,
                    max=self.log_scale_max_clip,
                )
            )
            mean = mean[..., idx] / 100
            x[idx] = (y[..., idx] - mean) * inverse_scale

        return torch.stack(x, dim=-1)

    def _log_abs_det_jacobian(
        self, x: torch.Tensor, y: torch.Tensor, params: flowtorch.ParamsModule
    ) -> torch.Tensor:
        # Note: params will take care of caching "mean, log_scale, perm = params(x)"
        _, log_scale = params(x)
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
