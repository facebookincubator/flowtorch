# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved
# SPDX-License-Identifier: MIT

import warnings
from typing import Callable, Optional, Sequence

import torch
import torch.nn as nn
from flowtorch.nn.made import MaskedLinear, create_mask
from flowtorch.parameters.base import Parameters


class DenseAutoregressive(Parameters):
    autoregressive = True

    def __init__(
        self,
        input_shape: torch.Size,
        param_shapes: Sequence[torch.Size],
        context_dims: int,
        *,
        hidden_dims: Sequence[int] = (256, 256),
        nonlinearity: Callable[[], nn.Module] = nn.ReLU,
        permutation: Optional[torch.LongTensor] = None,
        skip_connections: bool = False,
    ) -> None:
        super().__init__(input_shape, param_shapes, context_dims)
        self.hidden_dims = hidden_dims
        self.nonlinearity = nonlinearity
        self.skip_connections = skip_connections
        self._build(input_shape, param_shapes, context_dims, permutation)

    def _build(
        self,
        input_shape: torch.Size,
        param_shapes: Sequence[torch.Size],
        context_dims: int,
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

        self.output_multiplier = int(
            sum(max(torch.sum(torch.tensor(s)).item(), 1) for s in param_shapes_)
        )
        if input_dims == 1:
            warnings.warn(
                "DenseAutoregressive input_dim = 1. "
                "Consider using an affine transformation instead."
            )

        # Calculate the indices on the output corresponding to each parameter
        ends = torch.cumsum(
            torch.tensor(
                [max(torch.sum(torch.tensor(s)).item(), 1) for s in param_shapes_]
            ),
            dim=0,
        )
        starts = torch.cat((torch.zeros(1).type_as(ends), ends[:-1]))
        self.param_slices = [slice(s.item(), e.item()) for s, e in zip(starts, ends)]

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
            context_dim=context_dims,
            hidden_dims=hidden_dims,
            permutation=permutation,
            output_dim_multiplier=self.output_multiplier,
        )

        # Create masked layers
        layers = [
            MaskedLinear(
                input_dims + context_dims,
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
                    input_dims + context_dims,
                    input_dims * self.output_multiplier,
                    mask_skip,
                    bias=False,
                )
            )

        self.layers = nn.ModuleList(layers)

    def _forward(
        self,
        x: torch.Tensor,
        context: Optional[torch.Tensor] = None,
    ) -> Sequence[torch.Tensor]:
        input_dims = int(torch.sum(torch.tensor(self.input_shape)).int().item())

        # TODO: Flatten x. This will fail when len(input_shape) > 0
        # TODO: Get this working again when using skip_layers!
        # NOTE: this assumes x is a 2-tensor (batch_size, event_size)
        if context is not None:
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
        if len(self.input_shape) == 0:
            h = h.reshape(x.size()[:-1] + (self.output_multiplier, input_dims))
            result = tuple(
                h[..., p_slice, :].reshape(
                    torch.Size(h.shape[:-2])
                    + p_shape  # pyre-fixme[58]
                    + torch.Size((1,))  # pyre-fixme[58]
                )
                for p_slice, p_shape in zip(self.param_slices, list(self.param_shapes))
            )
        else:
            h = h.reshape(
                x.size()[: -len(self.input_shape)]
                + (self.output_multiplier, input_dims)
            )
            result = h.split([sl.stop - sl.start for sl in self.param_slices], dim=-2)
            result = tuple(
                h_slice.view(h.shape[:-2] + p_shape + self.input_shape)
                for h_slice, p_shape in zip(result, list(self.param_shapes))
            )
        return result
