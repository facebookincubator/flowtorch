# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved
# SPDX-License-Identifier: MIT
from typing import Sequence

import torch
import torch.distributions
from flowtorch.bijectors.base import Bijector


class Fixed(Bijector):
    def param_shapes(self, shape: torch.Size) -> Sequence[torch.Size]:
        """
        Given a base distribution, calculate the parameters for the transformation
        of that distribution under this bijector. By default, no parameters are
        set.
        """
        return []
