# Copyright (c) FlowTorch Development Team. All Rights Reserved
# SPDX-License-Identifier: MIT

import torch


def clamp_preserve_gradients(x, min, max):
    # This helper function clamps gradients but still passes through the
    # gradient in clamped regions
    return x + (x.clamp(min, max) - x).detach()


def clipped_sigmoid(x):
    finfo = torch.finfo(x.dtype)
    return torch.clamp(torch.sigmoid(x), min=finfo.tiny, max=1.0 - finfo.eps)
