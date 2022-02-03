# Copyright (c) Meta Platforms, Inc

from typing import Optional, Sequence, Tuple

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
        params_fn: Optional[flowtorch.Lazy] = None,
        *,
        shape: torch.Size,
        context_shape: Optional[torch.Size] = None,
        permutation: Optional[torch.Tensor] = None
    ) -> None:
        super().__init__(params_fn, shape=shape, context_shape=context_shape)
        self.permutation = permutation

    def _forward(
        self, x: torch.Tensor, params: Optional[Sequence[torch.Tensor]]
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
        if self.permutation is None:
            self.permutation = torch.randperm(x.shape[-1])

        y = torch.index_select(x, -1, self.permutation)
        ladj = self._log_abs_det_jacobian(x, y, params)
        return y, ladj

    def _inverse(
        self, y: torch.Tensor, params: Optional[Sequence[torch.Tensor]]
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
        if self.permutation is None:
            self.permutation = torch.randperm(y.shape[-1])

        x = torch.index_select(y, -1, self.inv_permutation)
        ladj = self._log_abs_det_jacobian(x, y, params)
        return x, ladj

    @lazy_property
    def inv_permutation(self) -> Optional[torch.Tensor]:
        if self.permutation is None:
            return None

        result = torch.empty_like(self.permutation, dtype=torch.long)
        result[self.permutation] = torch.arange(
            self.permutation.size(0), dtype=torch.long, device=self.permutation.device
        )
        return result
