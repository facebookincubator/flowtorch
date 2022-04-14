from typing import Tuple, Optional, Sequence

import torch
from torch import Tensor

import flowtorch
from flowtorch.bijectors import Bijector


class ReshapeBijector(Bijector):
    def __init__(
        self,
        params_fn: Optional[flowtorch.Lazy] = None,
        *,
        shape: torch.Size,
        transform: Bijector,
        context_shape: Optional[torch.Size] = None,
        **kwargs
    ):
        super().__init__(params_fn, shape=shape, context_shape=context_shape)
        self._transform = transform

    def _reshape_in(
        self, x: Tensor,
        mode: str,
        *params: Tensor
    ) -> Tuple[Tensor, ...]:
        raise NotImplementedError

    def _reshape_out(
        self,
        x: Tensor,
        mode: str,
        log_detJ: Tensor,
        *other_and_params: Tensor
    ) -> Tuple[Tensor, ...]:
        raise NotImplementedError

    def _forward(self, x: Tensor, params: Optional[Sequence[Tensor]]) -> Tuple[
        Tensor, Tensor]:
        if params is None:
            params = []
        # reshape
        x_reshaped, *other = self._reshape_in(x, 'forward', *params)

        # transform
        y_reshaped = self._transform.forward(x_reshaped)
        log_detJ = self._transform.log_abs_det_jacobian(x_reshaped, y_reshaped)

        # reshape out
        y, log_detJ = self._reshape_out(y_reshaped, 'forward', log_detJ,
                                        *other, *params)
        return y, log_detJ

    def _inverse(self, y: Tensor, params: Optional[Sequence[Tensor]]) -> Tuple[
        Tensor, Tensor]:
        if params is None:
            params = []
        # reshape
        y_reshaped, *other = self._reshape_in(y, 'inverse', *params)

        # transform
        x_reshaped = self._transform.inverse(y_reshaped)
        log_detJ = self._transform.log_abs_det_jacobian(x_reshaped, y_reshaped)

        # reshape out
        x, log_detJ = self._reshape_out(x_reshaped, 'inverse', log_detJ,
                                        *other, *params)
        return x, log_detJ
