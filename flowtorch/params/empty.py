# Copyright (c) FlowTorch Development Team. All Rights Reserved
# SPDX-License-Identifier: MIT

import warnings
from typing import Any, Dict, Optional, Sequence, Tuple

import torch
import torch.nn as nn
from torch.nn import functional as F

import flowtorch
from flowtorch.param import ParamsModule


class Empty(flowtorch.Params):
    def __call__(
        self,
        input_shape: torch.Size,
        param_shapes: Sequence[torch.Size],
        context_dims: int,
    ) -> Optional[ParamsModule]:
        return None
