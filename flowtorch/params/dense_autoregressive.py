# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved
# SPDX-License-Identifier: MIT

import warnings
from typing import Any, Callable, Dict, Optional, Sequence, Tuple

import torch
import torch.nn as nn
from flowtorch.params.base import Params
from torch.nn import functional as F


def sample_mask_indices(
    input_dim: int, hidden_dim: int, simple: bool = True
) -> torch.Tensor:
    """
    Samples the indices assigned to hidden units during the construction of MADE masks
    :param input_dim: the dimensionality of the input variable
    :param hidden_dim: the dimensionality of the hidden layer
    :param simple: True to space fractional indices by rounding to nearest
    int, false round randomly
    """
    indices = torch.linspace(1, input_dim, steps=hidden_dim, device="cpu").to(
        torch.Tensor().device
    )
    if simple:
        # Simple procedure tries to space fractional indices evenly by rounding
        # to nearest int
        return torch.round(indices)
    else:
        # "Non-simple" procedure creates fractional indices evenly then rounds
        # at random
        ints = indices.floor()
        ints += torch.bernoulli(indices - ints)
        return ints


def create_mask(
    input_dim: int,
    context_dim: int,
    hidden_dims: Sequence[int],
    permutation: torch.LongTensor,
    output_dim_multiplier: int,
) -> Tuple[Sequence[torch.Tensor], torch.Tensor]:
    """
    Creates MADE masks for a conditional distribution
    :param input_dim: the dimensionality of the input variable
    :param context_dim: the dimensionality of the variable that is
    conditioned on (for conditional densities)
    :param hidden_dims: the dimensionality of the hidden layers(s)
    :param permutation: the order of the input variables
    :param output_dim_multiplier: tiles the output (e.g. for when a separate
    mean and scale parameter are desired)
    """
    # Create mask indices for input, hidden layers, and final layer
    # We use 0 to refer to the elements of the variable being conditioned on,
    # and range(1:(D_latent+1)) for the input variable
    var_index = torch.empty(permutation.shape, dtype=torch.get_default_dtype())
    var_index[permutation] = torch.arange(input_dim, dtype=torch.get_default_dtype())

    # Create the indices that are assigned to the neurons
    input_indices = torch.cat((torch.zeros(context_dim), 1 + var_index))

    # For conditional MADE, introduce a 0 index that all the conditioned
    # variables are connected to as per Paige and Wood (2016) (see below)
    if context_dim > 0:
        hidden_indices = [sample_mask_indices(input_dim, h) - 1 for h in hidden_dims]
    else:
        hidden_indices = [sample_mask_indices(input_dim - 1, h) for h in hidden_dims]

    output_indices = (var_index + 1).repeat(output_dim_multiplier)

    # Create mask from input to output for the skips connections
    mask_skip = (output_indices.unsqueeze(-1) > input_indices.unsqueeze(0)).type_as(
        var_index
    )

    # Create mask from input to first hidden layer, and between subsequent
    # hidden layers
    masks = [
        (hidden_indices[0].unsqueeze(-1) >= input_indices.unsqueeze(0)).type_as(
            var_index
        )
    ]
    for i in range(1, len(hidden_dims)):
        masks.append(
            (
                hidden_indices[i].unsqueeze(-1) >= hidden_indices[i - 1].unsqueeze(0)
            ).type_as(var_index)
        )

    # Create mask from last hidden layer to output layer
    masks.append(
        (output_indices.unsqueeze(-1) > hidden_indices[-1].unsqueeze(0)).type_as(
            var_index
        )
    )

    return masks, mask_skip


class MaskedLinear(nn.Linear):
    """
    A linear mapping with a given mask on the weights (arbitrary bias)
    :param in_features: the number of input features
    :param out_features: the number of output features
    :param mask: the mask to apply to the in_features x out_features weight matrix
    :param bias: whether or not `MaskedLinear` should include a bias term.
    defaults to `True`
    """

    def __init__(
        self, in_features: int, out_features: int, mask: torch.Tensor, bias: bool = True
    ) -> None:
        super().__init__(in_features, out_features, bias)
        self.register_buffer("mask", mask.data)

    def forward(self, _input: torch.Tensor) -> torch.Tensor:
        masked_weight = self.weight * self.mask
        return F.linear(_input, masked_weight, self.bias)


class DenseAutoregressive(Params):
    autoregressive = True

    def __init__(
        self,
        hidden_dims: Sequence[int] = (256, 256),
        nonlinearity: Callable[[], nn.Module] = nn.ReLU,
        permutation: Optional[torch.LongTensor] = None,
        skip_connections: bool = False,
    ) -> None:
        super().__init__()
        self.hidden_dims = hidden_dims
        self.nonlinearity = nonlinearity
        self.permutation = permutation
        self.skip_connections = skip_connections

    # Continue from here!
    def _build(
        self,
        input_shape: torch.Size,
        param_shapes: Sequence[torch.Size],
        context_dims: int,
    ) -> Tuple[nn.ModuleList, Dict[str, Any]]:
        # TODO: Implement conditional version!
        self.context_dims = context_dims

        # Work out flattened input and output shapes
        param_shapes_ = list(param_shapes)
        self.input_dims = int(torch.sum(torch.tensor(input_shape)).int().item())
        if self.input_dims == 0:
            self.input_dims = 1  # scalars represented by torch.Size([])
        self.output_multiplier = int(
            sum(max(torch.sum(torch.tensor(s)).item(), 1) for s in param_shapes_)
        )
        if self.input_dims == 1:
            warnings.warn(
                "DenseAutoregressive input_dim = 1. "
                "Consider using an affine transformation instead."
            )
        self.count_params = len(param_shapes_)

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
            if h < self.input_dims:
                raise ValueError(
                    "Hidden dimension must not be less than input dimension."
                )

        if self.permutation is None:
            # By default set a random permutation of variables, which is
            # important for performance with multiple steps
            self.permutation = torch.LongTensor(
                torch.randperm(self.input_dims, device="cpu").to(
                    torch.LongTensor().device
                )
            )
        else:
            # The permutation is chosen by the user
            self.permutation = torch.LongTensor(self.permutation)

        # TODO: Check that the permutation is valid for the input dimension!
        # Implement ispermutation() that sorts permutation and checks whether it
        # has all integers from 0, 1, ..., self.input_dims - 1

        buffers = {"permutation": self.permutation}

        # Create masks
        hidden_dims = self.hidden_dims
        self.masks, self.mask_skip = create_mask(
            input_dim=self.input_dims,
            context_dim=self.context_dims,
            hidden_dims=hidden_dims,
            permutation=self.permutation,
            output_dim_multiplier=self.output_multiplier,
        )

        # Create masked layers
        layers = [
            MaskedLinear(
                self.input_dims + self.context_dims,
                hidden_dims[0],
                self.masks[0],
            ),
            self.nonlinearity(),
        ]
        for i in range(1, len(hidden_dims)):
            layers.extend(
                [
                    MaskedLinear(hidden_dims[i - 1], hidden_dims[i], self.masks[i]),
                    self.nonlinearity(),
                ]
            )
        layers.append(
            MaskedLinear(
                hidden_dims[-1],
                self.input_dims * self.output_multiplier,
                self.masks[-1],
            )
        )

        if self.skip_connections:
            layers.append(
                MaskedLinear(
                    self.input_dims + self.context_dims,
                    self.input_dims * self.output_multiplier,
                    self.mask_skip,
                    bias=False,
                )
            )

        return nn.ModuleList(layers), buffers

    def _forward(
        self,
        x: torch.Tensor,
        context: Optional[torch.Tensor],
        modules: nn.ModuleList,
    ) -> Sequence[torch.Tensor]:
        # TODO: Flatten x. This will fail when len(input_shape) > 0
        # TODO: Get this working again when using skip_layers!
        # NOTE: this assumes x is a 2-tensor (batch_size, event_size)
        if context is not None:
            h = torch.cat([context.expand((x.shape[0], -1)), x], dim=-1)
        else:
            h = x

        for idx in range(len(modules) // 2):
            h = modules[2 * idx + 1](modules[2 * idx](h))
        h = modules[-1](h)

        # TODO: Get skip_layers working again!
        # if self.skip_layer is not None:
        #    h = h + self.skip_layer(x)

        # Shape the output
        if len(self.input_shape) == 0:
            h = h.reshape(x.size()[:-1] + (self.output_multiplier, self.input_dims))
            result = tuple(
                h[..., p_slice, :].reshape(
                    torch.Size(h.shape[:-2]) + p_shape + torch.Size((1,))
                )
                for p_slice, p_shape in zip(self.param_slices, list(self.param_shapes))
            )
        else:
            h = h.reshape(
                x.size()[: -len(self.input_shape)]
                + (self.output_multiplier, self.input_dims)
            )
            result = h.split([sl.stop - sl.start for sl in self.param_slices], dim=-2)
            result = tuple(
                h_slice.view(h.shape[:-2] + p_shape + self.input_shape)
                for h_slice, p_shape in zip(result, list(self.param_shapes))
            )
        return result
