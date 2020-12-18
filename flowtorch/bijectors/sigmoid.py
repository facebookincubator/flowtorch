# Copyright (c) FlowTorch Development Team. All Rights Reserved
# SPDX-License-Identifier: MIT

import torch
import torch.nn.functional as F

import flowtorch
from flowtorch.utils import clipped_sigmoid


class Sigmoid(flowtorch.Bijector):
    def _forward(self, x, params=None):
        return clipped_sigmoid(x)

    def _inverse(self, y, params=None):
        finfo = torch.finfo(y.dtype)
        y = y.clamp(min=finfo.tiny, max=1.0 - finfo.eps)
        return y.log() - (-y).log1p()

    def _log_abs_det_jacobian(self, x, y, params=None):
        return -F.softplus(-x) - F.softplus(x)
