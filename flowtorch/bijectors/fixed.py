# Copyright (c) Meta Platforms, Inc
from typing import Optional, Sequence

import flowtorch
import torch
import torch.distributions
from flowtorch.bijectors.base import Bijector


class Fixed(Bijector):
    def __init__(
        self,
        params_fn: Optional[flowtorch.Lazy] = None,
        *,
        shape: torch.Size,
        context_shape: Optional[torch.Size] = None,
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
