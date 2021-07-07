# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved
# SPDX-License-Identifier: MIT
import torch

eps = 1e-8


def clamp_preserve_gradients(x: torch.Tensor, min: float, max: float) -> torch.Tensor:
    # This helper function clamps gradients but still passes through the
    # gradient in clamped regions
    return x + (x.clamp(min, max) - x).detach()


def clipped_sigmoid(x: torch.Tensor) -> torch.Tensor:
    finfo = torch.finfo(x.dtype)
    return torch.clamp(torch.sigmoid(x), min=finfo.tiny, max=1.0 - finfo.eps)
