# Copyright (c) Meta Platforms, Inc
from typing import Optional, Sequence, Tuple, Iterator

import torch
import torch.distributions
from torch.distributions.utils import _sum_rightmost

import flowtorch.parameters
from flowtorch.bijectors.base import Bijector
from flowtorch.bijectors.bijective_tensor import BijectiveTensor
from flowtorch.bijectors.utils import is_record_flow_graph_enabled, requires_log_detJ


class Compose(Bijector):
    def __init__(
            self,
            bijectors: Sequence[flowtorch.Lazy],
            *,
            shape: torch.Size,
            context_shape: Optional[torch.Size] = None,
    ):
        assert len(bijectors) > 0

        # Instantiate all bijectors, propagating shape information
        self.bijectors = []
        for bijector in bijectors:
            assert issubclass(bijector.cls, Bijector)

            self.bijectors.append(bijector(shape=shape))
            shape = self.bijectors[-1].forward_shape(shape)  # type: ignore

        self.domain = self.bijectors[0].domain  # type: ignore
        self.codomain = self.bijectors[-1].codomain  # type: ignore

        self._context_shape = context_shape

    def parameters(self) -> Iterator[torch.Tensor]:
        for b in self.bijectors:
            for param in b.parameters():
                yield param

    def _forward(
            self,
            x: torch.Tensor,
            context: Optional[torch.Tensor] = None
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
        log_detJ = None
        for bijector in self.bijectors:
            y = bijector.forward(x, context)  # type: ignore
            if is_record_flow_graph_enabled() and requires_log_detJ():
                if isinstance(y, BijectiveTensor) and y.from_forward():
                    _log_detJ = y._log_detJ
                elif isinstance(x, BijectiveTensor) and x.from_inverse():
                    _log_detJ = x._log_detJ
                else:
                    raise RuntimeError("neither of x nor y contains the log-abs-det-jacobian")
                log_detJ = log_detJ + _log_detJ if log_detJ is not None else _log_detJ
            x = y

        return x, log_detJ

    def _inverse(
            self,
            y: torch.Tensor,
            context: Optional[torch.Tensor] = None
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
        log_detJ = None
        for bijector in reversed(self.bijectors):
            x = bijector.inverse(y, context)  # type: ignore
            if is_record_flow_graph_enabled() and requires_log_detJ():
                if isinstance(y, BijectiveTensor) and y.from_forward():
                    _log_detJ = y._log_detJ
                elif isinstance(x, BijectiveTensor) and x.from_inverse():
                    _log_detJ = x._log_detJ
                else:
                    raise RuntimeError("neither of x nor y contains the log-abs-det-jacobian")
                log_detJ = log_detJ + _log_detJ if log_detJ is not None else _log_detJ
            y = x
        return y, log_detJ

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
            # If x is a BijectiveTensor and has y as ancestor, then the inversion flow.inverse(y) = x has already
            # been computed and we can recover the chain of parents instead of re-computing it.
            _use_cached_inverse = True
            parents = []
            while isinstance(x, BijectiveTensor) and x is not y:
                parents.append(x)
                x = x.parent
        else:
            _use_cached_inverse = False

        for bijector in reversed(self.bijectors):
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
