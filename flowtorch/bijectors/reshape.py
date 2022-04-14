from torch import Tensor
from typing import Tuple, Optional, Sequence

from flowtorch.bijectors import Bijector


class ReshapeBijector(Bijector):
    def __init__(
            self,
            *args,
            transform: Bijector,
            **kwargs
            ):
        super().__init__(*args, **kwargs)
        self._transform = transform

    def _reshape_in(self, x: Tensor, *params: Tensor):
        raise NotImplementedError

    def _reshape_out(self, x: Tensor, *other_and_params: Tensor):
        raise NotImplementedError

    def _forward(self, x: Tensor, params: Optional[Sequence[Tensor]]) -> Tuple[Tensor, Tensor]:
        # reshape
        x_reshaped, *other = self._reshape_in(x, *params)

        # transform
        y_reshaped = self._transform.forward(x_reshaped)
        log_detJ = self.transform.log_abs_det_jacobian(x_reshaped, y_reshaped)

        # reshape out
        y = self._reshape_out(y_reshaped, *other, *params)
        return y, log_detJ

    def _inverse(self, y: Tensor, params: Optional[Sequence[Tensor]]) -> Tuple[Tensor, Tensor]:
        # reshape
        y_reshaped, *other = self._reshape_in(y, *params)

        # transform
        x_reshaped = self._transform.inverse(y_reshaped)
        log_detJ = self.transform.log_abs_det_jacobian(x_reshaped, y_reshaped)

        # reshape out
        x = self._reshape_out(x_reshaped, *other, *params)
        return x, log_detJ
