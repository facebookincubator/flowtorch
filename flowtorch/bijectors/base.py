# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved
# SPDX-License-Identifier: MIT
import weakref
from typing import Optional, Sequence, Union, cast

import flowtorch.distributions
import flowtorch.params
import torch
import torch.distributions
from flowtorch.params import ParamsModule
from torch.distributions import constraints


class Bijector(object):
    _inv: Optional[Union[weakref.ReferenceType, "Bijector"]] = None
    codomain: constraints.Constraint = constraints.real
    domain: constraints.Constraint = constraints.real
    identity_initialization: bool = True
    autoregressive: bool = False
    _context_size: int
    event_dim: int = 0
    _params: Optional[flowtorch.params.ParamsModule] = None

    def __init__(
        self,
        param_fn: Optional[flowtorch.params.Params] = None,
        context_size: int = 0,
    ) -> None:
        super().__init__()
        self.param_fn = param_fn
        self._context_size = context_size

    def __call__(
        self, base_dist: torch.distributions.Distribution
    ) -> flowtorch.distributions.TransformedDistribution:
        """
        Returns the distribution formed by passing dist through the bijection
        """
        # If the input is a distribution then return transformed distribution
        if isinstance(base_dist, torch.distributions.Distribution):
            # Create transformed distribution
            # TODO: Check that if bijector is autoregressive then parameters are as
            # well Possibly do this in simplex.Bijector.__init__ and call from
            # simple.bijectors.*.__init__
            input_shape = (
                base_dist.batch_shape + base_dist.event_shape  # pyre-ignore[16]
            )

            self.params = None
            if self.param_fn is not None:
                self.params = self.param_fn(
                    input_shape, self.param_shapes(base_dist), self._context_size
                )  # <= this is where hypernets etc. are instantiated
            new_dist = flowtorch.distributions.TransformedDistribution(base_dist, self)
            return new_dist

        # TODO: Handle other types of inputs such as tensors
        else:
            raise TypeError(f"Bijector called with invalid type: {type(base_dist)}")

    @property
    def params(self) -> Optional[ParamsModule]:
        return self._params

    @params.setter
    def params(self, value: Optional[ParamsModule]):
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

    def param_shapes(
        self, dist: torch.distributions.Distribution
    ) -> Sequence[torch.Size]:
        """
        Abstract method to return shapes of parameters
        """
        raise NotImplementedError

    def inv(self) -> "Bijector":
        if self._inv is not None:
            # TODO: remove casting without failing mypy
            inv = cast(_InverseBijector, cast(weakref.ReferenceType, self._inv)())
        else:
            inv = _InverseBijector(self)
            self._inv = weakref.ref(inv)
        return inv

    def __repr__(self) -> str:
        return self.__class__.__name__ + "()"

    def forward_shape(self, shape):
        """
        Infers the shape of the forward computation, given the input shape.
        Defaults to preserving shape.
        """
        return shape

    def inverse_shape(self, shape):
        """
        Infers the shapes of the inverse computation, given the output shape.
        Defaults to preserving shape.
        """
        return shape


class _InverseBijector(Bijector):
    _inv: Bijector
    """
    Inverts a single :class:`Bijector`.
    This class is private; please instead use the ``Bijector.inv`` property.
    """

    def __init__(self, bijector: Bijector):
        super(_InverseBijector, self).__init__(param_fn=bijector.param_fn)
        self._inv = bijector
        self.param_fn = bijector.param_fn
        self.domain = bijector.codomain
        self.codomain = bijector.domain
        self._context_size = bijector._context_size

    @property
    def inv(self):
        return self._inv

    @property
    def params(self):
        return self.inv.params

    @params.setter
    def params(self, value):
        self.inv.params = value

    def __eq__(self, other):
        if not isinstance(other, _InverseBijector):
            return False
        assert self._inv is not None
        return self._inv == other._inv

    def _forward(
        self,
        x: torch.Tensor,
        context: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        return self._inv.inverse(x, context)

    def _inverse(
        self,
        y: torch.Tensor,
        context: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        return self._inv.forward(y, context)

    def _log_abs_det_jacobian(
        self,
        x: torch.Tensor,
        y: torch.Tensor,
        context: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        return -self._inv.log_abs_det_jacobian(y, x, context)

    def param_shapes(
        self, dist: torch.distributions.Distribution
    ) -> Sequence[torch.Size]:
        return self._inv.param_shapes(dist)

    def forward_shape(self, shape):
        return self._inv.inverse_shape(shape)

    def inverse_shape(self, shape):
        return self._inv.forward_shape(shape)
