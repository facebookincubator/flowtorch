# Copyright (c) Meta Platforms, Inc
from typing import Optional, Sequence, Tuple, Union, Callable, Iterator

import torch
import torch.distributions
from torch.distributions import constraints

import flowtorch.parameters
from flowtorch.bijectors.bijective_tensor import to_bijective_tensor, BijectiveTensor
from flowtorch.bijectors.utils import is_record_flow_graph_enabled
from flowtorch.parameters import Parameters


class Bijector(metaclass=flowtorch.LazyMeta):
    codomain: constraints.Constraint = constraints.real
    domain: constraints.Constraint = constraints.real
    _shape: torch.Size
    _context_shape: Optional[torch.Size]
    _hypernet: Callable[[Optional[torch.Tensor]], Optional[Union[Parameters, torch.nn.ModuleList]]] = lambda *x: None

    def __init__(
            self,
            params_fn: Optional[flowtorch.Lazy] = None,
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
        if params_fn is not None:
            param_shapes = self.param_shapes(shape)
            self._hypernet = params_fn(  # type: ignore
                param_shapes, self._shape, self._context_shape
            )

    @property
    def params_fn(self) -> Callable[[Optional[torch.Tensor]], Optional[Union[Parameters, torch.nn.ModuleList]]]:
        return self._hypernet

    @params_fn.setter
    def params_fn(self, value: Optional[Union[Parameters, torch.nn.ModuleList]]) -> None:
        self._hypernet = value

    def parameters(self) -> Iterator[torch.Tensor]:
        if hasattr(self._hypernet, 'parameters'):
            for param in self._hypernet.parameters():
                yield param

    def forward(
            self,
            x: torch.Tensor,
            context: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        # TODO: Allow that context can have a batch shape
        assert context is None  # or context.shape == (self._context_size,)
        if isinstance(x, BijectiveTensor) and x.from_inverse() and x.check_bijector(self) and x.check_context(context):
            return x.parent

        params = self.params_fn(x)
        # try:
        y, log_detJ = self._forward(x, params)
        # except:
        #     print(type(self))
        if is_record_flow_graph_enabled():
            y = to_bijective_tensor(x, y, context, self, log_detJ, mode="forward")
        return y

    def _forward(
            self,
            x: torch.Tensor,
            params: Optional[Sequence[torch.Tensor]],
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
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
        if isinstance(y, BijectiveTensor) and y.from_forward() and y.check_bijector(self) and y.check_context(context):
            return y.parent

        # TODO: What to do in this line?
        params = self.params_fn(x)
        try:
            x, log_detJ = self._inverse(y, params)
        except:
            print(type(self))

        if is_record_flow_graph_enabled():
            x = to_bijective_tensor(x, y, context, self, log_detJ, mode="inverse")
        return x

    def _inverse(
            self,
            y: torch.Tensor,
            params: Optional[Sequence[torch.Tensor]],
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
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
        # TODO: Allow that context can have a batch shape
        assert context is None  # or context.shape == (self._context_size,)
        ladj = None
        if isinstance(y, BijectiveTensor) and y.from_forward() and y.check_bijector(self) and y.check_context(context):
            ladj = y.log_detJ
        elif isinstance(x, BijectiveTensor) and x.from_inverse() and x.check_bijector(self) and x.check_context(
                context):
            ladj = x.log_detJ
        if ladj is None:
            return self._log_abs_det_jacobian(x, y, context)
        return ladj

    def _log_abs_det_jacobian(
            self,
            x: torch.Tensor,
            y: torch.Tensor,
            params: Optional[Sequence[torch.Tensor]],
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
