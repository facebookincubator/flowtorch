# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved
# SPDX-License-Identifier: MIT

from typing import Any, cast, Optional

import flowtorch
import flowtorch.parameters
import torch
import torch.distributions.constraints as constraints
from flowtorch.bijectors.base import Bijector
from flowtorch.parameters.dense_autoregressive import DenseAutoregressive


class Autoregressive(Bijector):
    # "Default" event shape is to operate on vectors
    domain = constraints.real_vector
    codomain = constraints.real_vector

    def __init__(
        self,
        params: Optional[flowtorch.Lazy] = None,
        *,
        shape: torch.Size,
        context_shape: Optional[torch.Size] = None,
        **kwargs: Any
    ) -> None:
        # Event shape is determined by `shape` argument
        self.domain = constraints.independent(constraints.real, len(shape))
        self.codomain = constraints.independent(constraints.real, len(shape))

        # currently only DenseAutoregressive has a `permutation` buffer
        if not params:
            params = DenseAutoregressive()  # type: ignore

        # TODO: Replace P.DenseAutoregressive with P.Autoregressive
        # In the future there will be other autoregressive parameter classes
        assert params is not None and issubclass(params.cls, DenseAutoregressive)

        super().__init__(params, shape=shape, context_shape=context_shape)

    def inverse(
        self,
        y: torch.Tensor,
        x: Optional[torch.Tensor] = None,
        context: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        # TODO: Allow that context can have a batch shape
        assert context is None  # or context.shape == (self._context_size,)
        params = self.params
        assert params is not None

        x_new = torch.zeros_like(y)
        # NOTE: Inversion is an expensive operation that scales in the
        # dimension of the input
        permutation = (
            params.permutation
        )  # TODO: type-safe named buffer (e.g. "permutation") access
        # TODO: Make permutation, inverse work for other event shapes
        for idx in cast(torch.LongTensor, permutation):
            x_new[..., idx] = self._inverse(y, x_new.clone(), context)[..., idx]

        return x_new

    def _log_abs_det_jacobian(
        self,
        x: torch.Tensor,
        y: torch.Tensor,
        context: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        raise NotImplementedError
