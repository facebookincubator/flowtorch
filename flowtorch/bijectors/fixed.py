# Copyright (c) Meta Platforms, Inc

# pyre-unsafe
from collections.abc import Sequence
from typing import Optional

import flowtorch
import torch
import torch.distributions
from flowtorch.bijectors.base import Bijector


class Fixed(Bijector):
    def __init__(
        self,
        params_fn: flowtorch.Lazy | None = None,
        *,
        shape: torch.Size,
        context_shape: torch.Size | None = None,
    ) -> None:
        # TODO: In the future, make Fixed actually mean that there is no autograd
        # through params
        super().__init__(params_fn, shape=shape, context_shape=context_shape)
        assert params_fn is None

    def param_shapes(self, shape: torch.Size) -> Sequence[torch.Size]:
        """
        Given a base distribution, calculate the parameters for the transformation
        of that distribution under this bijector. By default, no parameters are
        set.
        """
        # TODO: In the future, make Fixed actually mean that there is no autograd
        # through params
        return []
