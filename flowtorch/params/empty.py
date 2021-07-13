# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved
# SPDX-License-Identifier: MIT

from typing import Optional, Sequence

import torch
from flowtorch.params.base import Params, ParamsModule


class Empty(Params):
    def __call__(
        self,
        input_shape: torch.Size,
        param_shapes: Sequence[torch.Size],
        context_dims: int,
    ) -> Optional[ParamsModule]:
        return None
