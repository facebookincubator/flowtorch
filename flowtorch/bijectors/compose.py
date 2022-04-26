# Copyright (c) Meta Platforms, Inc
import warnings
from typing import Optional, Sequence

import flowtorch.parameters
import torch
import torch.distributions
from flowtorch.bijectors.base import Bijector
from flowtorch.bijectors.bijective_tensor import BijectiveTensor, to_bijective_tensor
from flowtorch.bijectors.utils import is_record_flow_graph_enabled, requires_log_detJ
from torch.distributions.utils import _sum_rightmost


class Compose(Bijector):
    def __init__(
        self,
        bijectors: Sequence[flowtorch.Lazy],
        *,
        shape: torch.Size,
        context_shape: Optional[torch.Size] = None,
    ):
        assert len(bijectors) > 0
        super().__init__(None, shape=shape, context_shape=context_shape)

        # Instantiate all bijectors, propagating shape information
        self.bijectors = torch.nn.ModuleList()
        for bijector in bijectors:
            assert issubclass(bijector.cls, Bijector)

            self.bijectors.append(bijector(shape=shape))  # type: ignore
            shape = self.bijectors[-1].forward_shape(shape)  # type: ignore

        self.domain = self.bijectors[0].domain  # type: ignore
        self.codomain = self.bijectors[-1].codomain  # type: ignore

        self._context_shape = context_shape

    def forward(
        self,
        x: torch.Tensor,
        context: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        log_detJ: Optional[torch.Tensor] = None
        x_temp = x
        for bijector in self.bijectors:
            y = bijector.forward(x_temp, context)  # type: ignore
            if is_record_flow_graph_enabled() and requires_log_detJ():
                if isinstance(y, BijectiveTensor) and y.from_forward():
                    _log_detJ = y._log_detJ
                elif isinstance(x_temp, BijectiveTensor) and x_temp.from_inverse():
                    _log_detJ = x_temp._log_detJ
                else:
                    raise RuntimeError(
                        "neither of x nor y contains the log-abs-det-jacobian"
                    )
                log_detJ = log_detJ + _log_detJ if log_detJ is not None else _log_detJ
            x_temp = y

        # TODO: Check that this doesn't contain bugs!
        if (
            is_record_flow_graph_enabled()
            and not isinstance(y, BijectiveTensor)
            and not (isinstance(x, BijectiveTensor) and y in set(x.parents()))
        ):
            # we exclude y that are bijective tensors for Compose
            y = to_bijective_tensor(x, x_temp, context, self, log_detJ, mode="forward")
        return y

    def inverse(
        self,
        y: torch.Tensor,
        x: Optional[torch.Tensor] = None,
        context: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        log_detJ: Optional[torch.Tensor] = None
        y_temp = y
        for bijector in reversed(self.bijectors._modules.values()):  # type: ignore
            x = bijector.inverse(y_temp, context)  # type: ignore
            if is_record_flow_graph_enabled() and requires_log_detJ():
                if isinstance(y_temp, BijectiveTensor) and y_temp.from_forward():
                    _log_detJ = y_temp._log_detJ
                elif isinstance(x, BijectiveTensor) and x.from_inverse():
                    _log_detJ = x._log_detJ
                else:
                    raise RuntimeError(
                        "neither of x nor y contains the log-abs-det-jacobian"
                    )
                log_detJ = log_detJ + _log_detJ if log_detJ is not None else _log_detJ
            y_temp = x  # type: ignore

        # TODO: Check that this doesn't contain bugs!
        if (
            is_record_flow_graph_enabled()
            and not isinstance(x, BijectiveTensor)
            and not (isinstance(y, BijectiveTensor) and x in set(y.parents()))
        ):
            x = to_bijective_tensor(y_temp, y, context, self, log_detJ, mode="inverse")
        return x  # type: ignore

    def log_abs_det_jacobian(
        self, x: torch.Tensor, y: torch.Tensor, context: torch.Tensor = None
    ) -> torch.Tensor:
        """
        Computes the log det jacobian `log |dy/dx|` given input and output.
        By default, assumes a volume preserving bijection.
        """
        ldj = _sum_rightmost(
            torch.zeros_like(y),
            self.domain.event_dim,
        )

        if isinstance(x, BijectiveTensor) and x.has_ancestor(y):
            # If x is a BijectiveTensor and has y as ancestor, then the
            # inversion flow.inverse(y) = x has already been computed and
            # we can recover the chain of parents instead of re-computing it.
            _use_cached_inverse = True
            parents = []
            while isinstance(x, BijectiveTensor) and x is not y:
                parents.append(x)
                x = x.parent
        else:
            _use_cached_inverse = False

        if (
            is_record_flow_graph_enabled()
            and not _use_cached_inverse
            and not isinstance(y, BijectiveTensor)
        ):
            warnings.warn(
                "Computing _log_abs_det_jacobian from values and not from cache."
            )

        for bijector in reversed(self.bijectors._modules.values()):  # type: ignore
            if not _use_cached_inverse:
                y_inv = bijector.inverse(y, context)  # type: ignore
            else:
                y_inv = parents.pop()
            ldj += bijector.log_abs_det_jacobian(y_inv, y, context)  # type: ignore
            y = y_inv
        return ldj

    def param_shapes(self, shape: torch.Size) -> Sequence[torch.Size]:
        """
        Given a base distribution, calculate the parameters for the transformation
        of that distribution under this bijector. By default, no parameters are
        set.
        """
        return []
