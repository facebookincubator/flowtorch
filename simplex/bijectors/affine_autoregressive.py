# Copyright (c) Simplex Development Team. All Rights Reserved
# SPDX-License-Identifier: MIT

import torch
import torch.nn as nn

import simplex
import simplex.params
from simplex.utils import clamp_preserve_gradients

class AffineAutoregressive(simplex.Bijector):
    event_dim = 1
    autoregressive = True

    def __init__(
            self,
            param_fn=simplex.Params(simplex.params.DenseAutoregressive),
            log_scale_min_clip=-5.,
            log_scale_max_clip=3.,
            sigmoid_bias=2.0,
    ):
        #super().__init__()
        self.param_fn = param_fn
        self.log_scale_min_clip = log_scale_min_clip
        self.log_scale_max_clip = log_scale_max_clip
        self.sigmoid_bias = sigmoid_bias

        self.sigmoid = nn.Sigmoid()
        self.logsigmoid = nn.LogSigmoid()

    def forward(self, x, params=None):
        mean, log_scale, _ = params(x)
        log_scale = clamp_preserve_gradients(log_scale, self.log_scale_min_clip, self.log_scale_max_clip)
        scale = torch.exp(log_scale)
        y = scale * x + mean
        return y

    def inverse(self, y, params=None):
        x_size = y.size()[:-1]
        _, _, perm = params()
        input_dim = y.size(-1)
        x = [torch.zeros(x_size, device=y.device)] * input_dim

        # NOTE: Inversion is an expensive operation that scales in the dimension of the input
        for idx in perm:
            mean, log_scale, _ = params(torch.stack(x, dim=-1))
            inverse_scale = torch.exp(-clamp_preserve_gradients(
                log_scale[..., idx], min=self.log_scale_min_clip, max=self.log_scale_max_clip))
            mean = mean[..., idx]
            x[idx] = (y[..., idx] - mean) * inverse_scale

        x = torch.stack(x, dim=-1)
        return x

    def log_abs_det_jacobian(self, x, params=None):
        # Note: params will take care of caching "mean, log_scale, perm = params(x)"
        _, log_scale, _ = params(x)
        log_scale = clamp_preserve_gradients(log_scale, self.log_scale_min_clip, self.log_scale_max_clip)
        return log_scale.sum(-1)

    def param_shapes(self, dist):
        # A mean and log variance for every dimension of base distribution
        return torch.Size([]), torch.Size([])
