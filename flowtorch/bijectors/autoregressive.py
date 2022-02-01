# Copyright (c) Meta Platforms, Inc

from typing import Any, cast, Optional, Sequence

import flowtorch
import flowtorch.parameters
import torch
import torch.distributions.constraints as constraints
from flowtorch.bijectors.base import Bijector
from flowtorch.bijectors.bijective_tensor import BijectiveTensor, to_bijective_tensor
from flowtorch.bijectors.utils import is_record_flow_graph_enabled
from flowtorch.parameters.dense_autoregressive import DenseAutoregressive


class Autoregressive(Bijector):
    # "Default" event shape is to operate on vectors
    domain = constraints.real_vector
    codomain = constraints.real_vector

    def __init__(
        self,
        params_fn: Optional[flowtorch.Lazy] = None,
        *,
        shape: torch.Size,
        context_shape: Optional[torch.Size] = None,
        **kwargs: Any
    ) -> None:
        # Event shape is determined by `shape` argument
        self.domain = constraints.independent(constraints.real, len(shape))
        self.codomain = constraints.independent(constraints.real, len(shape))

        # currently only DenseAutoregressive has a `permutation` buffer
        if not params_fn:
            params_fn = DenseAutoregressive()  # type: ignore

        # TODO: Replace P.DenseAutoregressive with P.Autoregressive
        # In the future there will be other autoregressive parameter classes
        assert params_fn is not None and issubclass(params_fn.cls, DenseAutoregressive)

        super().__init__(params_fn, shape=shape, context_shape=context_shape)

    def inverse(
        self,
        y: torch.Tensor,
        x: Optional[torch.Tensor] = None,
        context: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        # TODO: Allow that context can have a batch shape
        assert context is None  # or context.shape == (self._context_size,)
        assert self._params_fn is not None
        if self._check_bijective_y(y, context):
            assert isinstance(y, BijectiveTensor)
            return y.get_parent_from_bijector(self)
        x_new = torch.zeros_like(y)
        # NOTE: Inversion is an expensive operation that scales in the
        # dimension of the input
        permutation = (
            self._params_fn.permutation
        )  # TODO: type-safe named buffer (e.g. "permutation") access
        # TODO: Make permutation, inverse work for other event shapes
        log_detJ: Optional[torch.Tensor] = None
        for idx in cast(torch.LongTensor, permutation):
            _params = self._params_fn(x_new.clone(), context=context)
            x_temp, log_detJ = self._inverse(y, params=_params)
            x_new[..., idx] = x_temp[..., idx]
            # _log_detJ = out[1]
            # log_detJ = _log_detJ

        if is_record_flow_graph_enabled():
            x_new = to_bijective_tensor(
                x_new,
                y,
                context=context,
                bijector=self,
                mode="inverse",
                log_detJ=log_detJ,
            )
        return x_new

    def _log_abs_det_jacobian(
        self, x: torch.Tensor, y: torch.Tensor, params: Optional[Sequence[torch.Tensor]]
    ) -> torch.Tensor:
        raise NotImplementedError
