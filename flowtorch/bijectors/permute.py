# Copyright (c) Meta Platforms, Inc

# pyre-unsafe

from collections.abc import Sequence
from typing import Optional, Tuple

import flowtorch
import torch
import torch.distributions.constraints as constraints
from flowtorch.bijectors.fixed import Fixed
from flowtorch.bijectors.volume_preserving import VolumePreserving
from torch.distributions.utils import lazy_property


class Permute(Fixed, VolumePreserving):
    domain = constraints.real_vector
    codomain = constraints.real_vector

    # TODO: A new abstraction so can defer construction of permutation
    def __init__(
        self,
        params_fn: flowtorch.Lazy | None = None,
        *,
        shape: torch.Size,
        context_shape: torch.Size | None = None,
        permutation: torch.Tensor | None = None,
    ) -> None:
        super().__init__(params_fn, shape=shape, context_shape=context_shape)
        self.permutation = permutation

    def _forward(
        self, x: torch.Tensor, params: Sequence[torch.Tensor] | None
    ) -> tuple[torch.Tensor, torch.Tensor | None]:
        if self.permutation is None:
            self.permutation = torch.randperm(x.shape[-1])

        # pyre-fixme[6]: For 3rd argument expected `Tensor` but got `Optional[Tensor]`.
        y = torch.index_select(x, -1, self.permutation)
        ladj = self._log_abs_det_jacobian(x, y, params)
        return y, ladj

    def _inverse(
        self, y: torch.Tensor, params: Sequence[torch.Tensor] | None
    ) -> tuple[torch.Tensor, torch.Tensor | None]:
        if self.permutation is None:
            self.permutation = torch.randperm(y.shape[-1])

        x = torch.index_select(y, -1, self.inv_permutation)
        ladj = self._log_abs_det_jacobian(x, y, params)
        return x, ladj

    @lazy_property
    def inv_permutation(self) -> torch.Tensor | None:
        if self.permutation is None:
            return None

        result = torch.empty_like(self.permutation, dtype=torch.long)
        result[self.permutation] = torch.arange(
            # pyre-fixme[16]: `Optional` has no attribute `size`.
            # pyre-fixme[16]: `Optional` has no attribute `device`.
            self.permutation.size(0),
            dtype=torch.long,
            # pyre-fixme[16]: `Optional` has no attribute `device`.
            device=self.permutation.device,
        )
        return result
