# Copyright (c) FlowTorch Development Team. All Rights Reserved
# SPDX-License-Identifier: MIT

import torch

import flowtorch
import flowtorch.params
from flowtorch.utils import clamp_preserve_gradients


class AffineAutoregressive(flowtorch.Bijector):
    event_dim = 1
    autoregressive = True

    def __init__(
        self,
        param_fn=flowtorch.params.dense_autoregressive,
        log_scale_min_clip=-5.0,
        log_scale_max_clip=3.0,
        sigmoid_bias=2.0,
    ):
        super(AffineAutoregressive, self).__init__(
            param_fn=param_fn,
            log_scale_min_clip=log_scale_min_clip,
            log_scale_max_clip=log_scale_max_clip,
            sigmoid_bias=sigmoid_bias,
        )

    def _forward(self, x, params=None):
        mean, log_scale = params(x)
        log_scale = clamp_preserve_gradients(
            log_scale, self.log_scale_min_clip, self.log_scale_max_clip
        )
        scale = torch.exp(log_scale)
        y = scale * x + mean
        return y

    def _inverse(self, y, params=None):
        x_size = y.size()[:-1]
        perm = params.permutation
        input_dim = y.size(-1)
        x = [torch.zeros(x_size, device=y.device)] * input_dim

        # NOTE: Inversion is an expensive operation that scales in the
        # dimension of the input
        for idx in perm:
            mean, log_scale = params(torch.stack(x, dim=-1))
            inverse_scale = torch.exp(
                -clamp_preserve_gradients(
                    log_scale[..., idx],
                    min=self.log_scale_min_clip,
                    max=self.log_scale_max_clip,
                )
            )
            mean = mean[..., idx]
            x[idx] = (y[..., idx] - mean) * inverse_scale

        x = torch.stack(x, dim=-1)
        return x

    def _log_abs_det_jacobian(self, x, y, params=None):
        # Note: params will take care of caching "mean, log_scale, perm = params(x)"
        _, log_scale = params(x)
        log_scale = clamp_preserve_gradients(
            log_scale, self.log_scale_min_clip, self.log_scale_max_clip
        )
        return log_scale.sum(-1)

    def param_shapes(self, dist):
        # A mean and log variance for every dimension of base distribution
        return torch.Size([]), torch.Size([])
