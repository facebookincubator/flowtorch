# Copyright (c) Meta Platforms, Inc

import warnings
from typing import Callable, Optional, Sequence

import torch
import torch.nn as nn
from flowtorch.nn.made import create_mask, MaskedLinear
from flowtorch.parameters.base import Parameters


class DenseAutoregressive(Parameters):
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

        # Create masks
        hidden_dims = self.hidden_dims
        masks, mask_skip = create_mask(
            input_dim=input_dims,
            context_dim=0,  # context_dims,
            hidden_dims=hidden_dims,
            permutation=permutation,
            output_multiplier=self.output_multiplier,
        )

        # Create masked layers
        layers = [
            MaskedLinear(
                input_dims,  # + context_dims,
                hidden_dims[0],
                masks[0],
            ),
            self.nonlinearity(),
        ]
        for i in range(1, len(hidden_dims)):
            layers.extend(
                [
                    MaskedLinear(hidden_dims[i - 1], hidden_dims[i], masks[i]),
                    self.nonlinearity(),
                ]
            )
        layers.append(
            MaskedLinear(
                hidden_dims[-1],
                input_dims * self.output_multiplier,
                masks[-1],
            )
        )

        if self.skip_connections:
            layers.append(
                MaskedLinear(
                    input_dims,  # + context_dims,
                    input_dims * self.output_multiplier,
                    mask_skip,
                    bias=False,
                )
            )

        self.layers = nn.ModuleList(layers)

    def _forward(
        self,
        x: Optional[torch.Tensor] = None,
        context: Optional[torch.Tensor] = None,
    ) -> Optional[Sequence[torch.Tensor]]:
        assert x is not None

        # Flatten x
        batch_shape = x.shape[: len(x.shape) - len(self.input_shape)]
        if len(batch_shape) > 0:
            x = x.reshape(batch_shape + (-1,))

        if context is not None:
            # TODO: Fix the following!
            h = torch.cat([context.expand((x.shape[0], -1)), x], dim=-1)
        else:
            h = x

        for idx in range(len(self.layers) // 2):
            h = self.layers[2 * idx + 1](self.layers[2 * idx](h))
        h = self.layers[-1](h)

        # TODO: Get skip_layers working again!
        # if self.skip_layer is not None:
        #    h = h + self.skip_layer(x)

        # Shape the output
        # h ~ (batch_dims * input_dims, total_params_per_dim)
        h = h.reshape(-1, self.output_multiplier)

        # result ~ (batch_dims * input_dims, params_per_dim[0]), ...
        result = h.split(list(self.param_dims), dim=-1)

        # results ~ (batch_shape, param_shapes[0]), ...
        result = tuple(
            h_slice.view(batch_shape + p_shape)
            for h_slice, p_shape in zip(result, list(self.param_shapes))
        )
        return result
