# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved
# SPDX-License-Identifier: MIT

from typing import Any, Dict, Sequence, Tuple

import flowtorch.params as params
import torch
import torch.nn as nn


class DenseAutoregressive(params.DenseAutoregressive):
    def _build(
        self,
        input_shape: torch.Size,
        param_shapes: Sequence[torch.Size],
        context_dims: int,
    ) -> Tuple[nn.ModuleList, Dict[str, Any]]:
        # Construct network in standard way for DenseAutoregressive
        layers, buffers = super()._build(input_shape, param_shapes, context_dims)

        # Perform experimental initialization
        self._init_weights(layers)

        return nn.ModuleList(layers), buffers

    def _init_weights(self, layers: nn.ModuleList) -> None:
        """
        EXPERIMENTAL: initialize weights such that transforming a standard Normal yields
        a Normal with zero mean and variance less than one.

        NOTE: may have stability issues for reasonable (1e-3) learning rates,
        see https://github.com/stefanwebb/flowtorch/issues/43
        """
        input_dim = self.input_dims + self.context_dims
        weight_product = torch.eye(input_dim, input_dim)

        for idx in range(0, len(layers), 2):
            # Required for type checking
            layer = layers[idx]
            assert (
                isinstance(layer.bias, torch.Tensor)
                and isinstance(layer.weight, torch.Tensor)
                and isinstance(layer.mask, torch.Tensor)
            )

            # Initialize biases to 0
            torch.nn.init.zeros_(layer.bias)

            # Initialize product of weights up until this point so each column has
            # l_2 norm = 1 * scaling constant
            # layer.weight ~ input_dims x output_dims
            weight_product = torch.matmul(layer.weight * layer.mask, weight_product)
            l2_norm = torch.sum(weight_product.pow(2), dim=1, keepdim=True).sqrt()
            layer.weight.data.div_(l2_norm * 16.0 + 1e-8)
            weight_product.data.div_(l2_norm + 1e-8)
