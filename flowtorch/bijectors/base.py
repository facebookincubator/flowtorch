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
    codomain: constraints.Constraint = constraints.real
    domain: constraints.Constraint = constraints.real
    _shape: torch.Size
    _context_shape: Optional[torch.Size]
    _params: Optional[Union[Parameters, torch.nn.ModuleList]] = None

    def __init__(
        self,
        params: Optional[flowtorch.Lazy] = None,
        *,
        shape: torch.Size,
        context_shape: Optional[torch.Size] = None,
    ) -> None:
        # Prevent "meta bijectors" from being initialized
        # NOTE: We define a "standard bijector" as one that inherits from a
        # subclass of Bijector, hence why we need to test the length of the MRO
        if (
            self.__class__.__module__ == "flowtorch.bijectors.base"
            or len(self.__class__.__mro__) <= 3
        ):
            raise TypeError("Only standard bijectors can be initialized.")

        self._shape = shape
        self._context_shape = context_shape

        # Instantiate parameters (tensor, hypernets, etc.)
        if params is not None:
            param_shapes = self.param_shapes(shape)
            self._params = params(  # type: ignore
                param_shapes, self._shape, self._context_shape
            )

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
        # TODO: Allow that context can have a batch shape
        assert context is None  # or context.shape == (self._context_size,)
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
        x: Optional[torch.Tensor] = None,
        context: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        # TODO: Allow that context can have a batch shape
        assert context is None  # or context.shape == (self._context_size,)
        return self._inverse(y, x, context)

    def _inverse(
        self,
        y: torch.Tensor,
        x: Optional[torch.Tensor] = None,
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
