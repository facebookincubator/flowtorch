# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved
# SPDX-License-Identifier: MIT
from typing import Optional, Sequence, Union

import flowtorch
import flowtorch.distributions
import flowtorch.parameters
import torch
import torch.distributions
from flowtorch.parameters import Parameters
from torch.distributions import constraints


class Bijector(metaclass=flowtorch.LazyMeta):
    # _inv: Optional[Union[weakref.ReferenceType, "Bijector"]] = None
    codomain: constraints.Constraint = constraints.real
    domain: constraints.Constraint = constraints.real
    identity_initialization: bool = True
    autoregressive: bool = False
    _context_size: int
    event_dim: int = 0
    _params: Optional[Union[Parameters, torch.nn.ModuleList]] = None

    def __init__(
        self,
        shape: torch.Size,
        params: Optional[flowtorch.Lazy] = None,
        context_size: int = 0,
    ) -> None:
        self._context_size = context_size

        # Instantiate parameters (tensor, hypernets, etc.)
        if params is not None:
            shapes = self.param_shapes(shape)
            self._params = params(shape, shapes, self._context_size)  # type: ignore

    @property
    def params(self) -> Optional[Union[Parameters, torch.nn.ModuleList]]:
        return self._params

    @params.setter
    def params(self, value: Optional[Union[Parameters, torch.nn.ModuleList]]) -> None:
        self._params = value

    def forward(
        self,
        x: torch.Tensor,
        context: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        assert context is None or context.shape == (self._context_size,)
        return self._forward(x, context)

    def _forward(
        self,
        x: torch.Tensor,
        context: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """
        Abstract method to compute forward transformation.
        """
        raise NotImplementedError

    def inverse(
        self,
        y: torch.Tensor,
        context: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        assert context is None or context.shape == (self._context_size,)
        return self._inverse(y, context)

    def _inverse(
        self,
        y: torch.Tensor,
        context: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """
        Abstract method to compute inverse transformation.
        """
        raise NotImplementedError

    def log_abs_det_jacobian(
        self,
        x: torch.Tensor,
        y: torch.Tensor,
        context: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """
        Computes the log det jacobian `log |dy/dx|` given input and output.
        By default, assumes a volume preserving bijection.
        """
        return self._log_abs_det_jacobian(x, y, context)

    def _log_abs_det_jacobian(
        self,
        x: torch.Tensor,
        y: torch.Tensor,
        context: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """
        Computes the log det jacobian `log |dy/dx|` given input and output.
        By default, assumes a volume preserving bijection.
        """

        # TODO: Sum out self.event_dim right-most dimensions
        # self.event_dim may be > 0 for derived classes!
        return torch.zeros_like(x)

    def param_shapes(self, shape: torch.Size) -> Sequence[torch.Size]:
        """
        Abstract method to return shapes of parameters
        """
        raise NotImplementedError

    """
    def inv(self) -> "Bijector":
        if self._inv is not None:
            # TODO: remove casting without failing mypy
            inv = cast(_InverseBijector, cast(weakref.ReferenceType, self._inv)())
        else:
            inv = _InverseBijector(self)
            self._inv = weakref.ref(inv)
        return inv
    """

    def __repr__(self) -> str:
        return self.__class__.__name__ + "()"

    def forward_shape(self, shape: torch.Size) -> torch.Size:
        """
        Infers the shape of the forward computation, given the input shape.
        Defaults to preserving shape.
        """
        return shape

    def inverse_shape(self, shape: torch.Size) -> torch.Size:
        """
        Infers the shapes of the inverse computation, given the output shape.
        Defaults to preserving shape.
        """
        return shape
