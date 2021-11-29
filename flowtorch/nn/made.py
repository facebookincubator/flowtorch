# Copyright (c) Meta Platforms, Inc

from typing import Sequence, Tuple

import torch
import torch.nn as nn
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
    output_multiplier: int,
) -> Tuple[Sequence[torch.Tensor], torch.Tensor]:
    """
    Creates MADE masks for a conditional distribution
    :param input_dim: the dimensionality of the input variable
    :param context_dim: the dimensionality of the variable that is
    conditioned on (for conditional densities)
    :param hidden_dims: the dimensionality of the hidden layers(s)
    :param permutation: the order of the input variables
    :param output_multipliers: tiles the output (e.g. for when a separate
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

    # *** TODO: Fix this line ***
    output_indices = (
        (var_index + 1).unsqueeze(-1).repeat(1, output_multiplier).reshape(-1)
    )

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
