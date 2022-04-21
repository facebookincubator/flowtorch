# Copyright (c) Meta Platforms, Inc
from copy import deepcopy
from typing import Optional, Sequence, Tuple

import flowtorch.parameters
import torch
from flowtorch.bijectors.ops.affine import Affine as AffineOp
from flowtorch.parameters import ConvCoupling, DenseCoupling
from torch.distributions import constraints


_REAL3d = deepcopy(constraints.real)
_REAL3d.event_dim = 3

_REAL1d = deepcopy(constraints.real)
_REAL1d.event_dim = 1


class CouplingBijector(AffineOp):
    """
    Examples:
        >>> params = DenseCoupling()
        >>> bij = CouplingBijector(params)
        >>> bij = bij(shape=torch.Size([32,]))
        >>> for p in bij.parameters():
        ...     p.data += torch.randn_like(p)/10
        >>> x = torch.randn(1, 32,requires_grad=True)
        >>> y = bij.forward(x).detach_from_flow()
        >>> x_bis = bij.inverse(y)
        >>> torch.testing.assert_allclose(x, x_bis)
    """

    domain: constraints.Constraint = _REAL1d
    codomain: constraints.Constraint = _REAL1d

    def __init__(
        self,
        params_fn: Optional[flowtorch.Lazy] = None,
        *,
        shape: torch.Size,
        context_shape: Optional[torch.Size] = None,
        clamp_values: bool = False,
        log_scale_min_clip: float = -5.0,
        log_scale_max_clip: float = 3.0,
        sigmoid_bias: float = 2.0,
        positive_map: str = "softplus",
        positive_bias: Optional[float] = None,
    ) -> None:

        if params_fn is None:
            params_fn = DenseCoupling()  # type: ignore

        AffineOp.__init__(
            self,
            params_fn,
            shape=shape,
            context_shape=context_shape,
            clamp_values=clamp_values,
            log_scale_min_clip=log_scale_min_clip,
            log_scale_max_clip=log_scale_max_clip,
            sigmoid_bias=sigmoid_bias,
            positive_map=positive_map,
            positive_bias=positive_bias,
        )

    def _forward(
        self, x: torch.Tensor, params: Optional[Sequence[torch.Tensor]]
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        assert self._params_fn is not None

        y, ldj = super()._forward(x, params)
        return y, ldj

    def _inverse(
        self, y: torch.Tensor, params: Optional[Sequence[torch.Tensor]]
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        assert self._params_fn is not None

        x, ldj = super()._inverse(y, params)
        return x, ldj


class ConvCouplingBijector(CouplingBijector):
    """
    Examples:
        >>> params = ConvCoupling()
        >>> bij = ConvCouplingBijector(params)
        >>> bij = bij(shape=torch.Size([3,16,16]))
        >>> for p in bij.parameters():
        ...     p.data += torch.randn_like(p)/10
        >>> x = torch.randn(4, 3, 16, 16)
        >>> y = bij.forward(x)
        >>> x_bis = bij.inverse(y.detach_from_flow())
        >>> torch.testing.assert_allclose(x, x_bis)
    """

    domain: constraints.Constraint = _REAL3d
    codomain: constraints.Constraint = _REAL3d

    def __init__(
        self,
        params_fn: Optional[flowtorch.Lazy] = None,
        *,
        shape: torch.Size,
        context_shape: Optional[torch.Size] = None,
        clamp_values: bool = False,
        log_scale_min_clip: float = -5.0,
        log_scale_max_clip: float = 3.0,
        sigmoid_bias: float = 2.0,
        positive_map: str = "softplus",
        positive_bias: Optional[float] = None,
    ) -> None:

        if not len(shape) == 3:
            raise ValueError(f"Expected a 3d-tensor shape, got {shape}")

        if params_fn is None:
            params_fn = ConvCoupling()  # type: ignore

        AffineOp.__init__(
            self,
            params_fn,
            shape=shape,
            context_shape=context_shape,
            clamp_values=clamp_values,
            log_scale_min_clip=log_scale_min_clip,
            log_scale_max_clip=log_scale_max_clip,
            sigmoid_bias=sigmoid_bias,
            positive_map=positive_map,
            positive_bias=positive_bias,
        )
