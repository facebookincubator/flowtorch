# Copyright (c) Meta Platforms, Inc

import warnings
from typing import Callable, Optional, Sequence

import torch
import torch.nn as nn
from flowtorch.nn.made import create_mask, MaskedLinear
from flowtorch.parameters.base import Parameters


class DenseCoupling(Parameters):
    autoregressive = True

    def __init__(
        self,
        param_shapes: Sequence[torch.Size],
        input_shape: torch.Size,
        context_shape: Optional[torch.Size],
        *,
        hidden_dims: Sequence[int] = (256, 256),
        nonlinearity: Callable[[], nn.Module] = nn.ReLU,
        permutation: Optional[torch.LongTensor] = None,
        skip_connections: bool = False,
    ) -> None:
        super().__init__(param_shapes, input_shape, context_shape)

        # Check consistency of input_shape with param_shapes
        # We need each param_shapes to match input_shape in
        # its leftmost dimensions
        for s in param_shapes:
            assert len(s) >= len(input_shape) and s[: len(input_shape)] == input_shape

        self.hidden_dims = hidden_dims
        self.nonlinearity = nonlinearity
        self.skip_connections = skip_connections
        self._build(input_shape, param_shapes, context_shape, permutation)

    def _build(
        self,
        input_shape: torch.Size,
        param_shapes: Sequence[torch.Size],
        context_shape: Optional[torch.Size],
        permutation: Optional[torch.LongTensor],
    ) -> None:
        # Work out flattened input and output shapes
        param_shapes_ = list(param_shapes)
        input_dims = int(torch.sum(torch.tensor(input_shape)).int().item())
        self.input_dims = input_dims
        if input_dims == 0:
            input_dims = 1  # scalars represented by torch.Size([])
        if permutation is None:
            # By default set a random permutation of variables, which is
            # important for performance with multiple steps
            permutation = torch.LongTensor(
                torch.randperm(input_dims, device="cpu").to(
                    torch.LongTensor((1,)).device
                )
            )
        else:
            # The permutation is chosen by the user
            permutation = torch.LongTensor(permutation)

        self.param_dims = [
            int(max(torch.prod(torch.tensor(s[len(input_shape) :])).item(), 1))
            for s in param_shapes_
        ]

        self.output_multiplier = sum(self.param_dims)

        if input_dims == 1:
            warnings.warn(
                "DenseAutoregressive input_dim = 1. "
                "Consider using an affine transformation instead."
            )

        # Calculate the indices on the output corresponding to each parameter
        # TODO: Is this logic correct???
        # ends = torch.cumsum(
        #    torch.tensor(
        #        [max(torch.prod(torch.tensor(s)).item(), 1) for s in param_shapes_]
        #    ),
        #    dim=0,
        # )
        # starts = torch.cat((torch.zeros(1).type_as(ends), ends[:-1]))
        # self.param_slices = [slice(s.item(), e.item()) for s, e in zip(starts, ends)]

        # Hidden dimension must be not less than the input otherwise it isn't
        # possible to connect to the outputs correctly
        for h in self.hidden_dims:
            if h < input_dims:
                raise ValueError(
                    "Hidden dimension must not be less than input dimension."
                )

        # TODO: Check that the permutation is valid for the input dimension!
        # Implement ispermutation() that sorts permutation and checks whether it
        # has all integers from 0, 1, ..., self.input_dims - 1
        self.register_buffer("permutation", permutation)
        self.register_buffer("inv_permutation", permutation.argsort())

        # Create masks
        hidden_dims = self.hidden_dims

        # Create masked layers:
        # input is [x1 ; 0]
        # output is [0 ; mu2], [0 ; sig2]
        mask_input = torch.ones(hidden_dims[0], input_dims)
        self.x1_dim = x1_dim = input_dims // 2
        mask_input[:, x1_dim:] = 0.0

        out_dims = input_dims * self.output_multiplier
        mask_output = torch.ones(self.output_multiplier, input_dims, hidden_dims[-1])
        mask_output[:x1_dim] = 0.0
        mask_output = mask_output.view(-1, hidden_dims[-1])
        self._bias = nn.Parameter(
            torch.zeros(self.output_multiplier, x1_dim, requires_grad=True)
        )

        layers = [
            MaskedLinear(
                input_dims,  # + context_dims,
                hidden_dims[0],
                mask_input,
            ),
            self.nonlinearity(),
        ]
        for i in range(1, len(hidden_dims)):
            layers.extend(
                [
                    nn.Linear(hidden_dims[i - 1], hidden_dims[i]),
                    self.nonlinearity(),
                ]
            )
        layers.append(
            MaskedLinear(
                hidden_dims[-1],
                out_dims,
                mask_output,
                bias=False,
            )
        )

        for l in layers[::2]:
            l.weight.data.normal_(0.0, 1e-3)  # type: ignore
            if l.bias is not None:
                l.bias.data.fill_(0.0)  # type: ignore

        if self.skip_connections:
            mask_skip = torch.ones(out_dims, input_dims)
            mask_skip[:, input_dims // 2 :] = 0.0
            mask_skip[: mask_output // 2] = 0.0
            self.skip_layer = MaskedLinear(
                input_dims,  # + context_dims,
                out_dims,
                mask_skip,
                bias=False,
            )

        self.layers = nn.Sequential(*layers)

    @property
    def bias(self) -> torch.Tensor:
        z = torch.zeros(
            self.output_multiplier,
            self.input_dims - self.x1_dim,
            device=self._bias.device,
            dtype=self._bias.dtype,
        )
        return torch.cat([z, self._bias], -1).view(-1)

    def _forward(
        self,
        x: Optional[torch.Tensor] = None,
        y: Optional[torch.Tensor] = None,
        context: Optional[torch.Tensor] = None,
    ) -> Optional[Sequence[torch.Tensor]]:
        if (x is None) and (y is None):
            raise RuntimeError("Either x or y must be provided.")
        elif x is None:
            x = y[..., self.inv_permutation]  # type: ignore
            inverse = True
        else:
            x = x[..., self.permutation]  # type: ignore
            inverse = False

        if context is not None:
            x_aug = torch.cat([context.expand((*x.shape[:-1], -1)), x], dim=-1)
        else:
            x_aug = x

        h = self.layers(x_aug) + self.bias

        # TODO: Get skip_layers working again!
        if self.skip_connections:
            h = h + self.skip_layer(x_aug)

        # Shape the output
        h = h.view(*x.shape[:-1], self.output_multiplier, -1)

        result = h.unbind(-2)
        perm = self.inv_permutation if inverse else self.permutation
        result = tuple(r[..., perm] for r in result)
        return result
