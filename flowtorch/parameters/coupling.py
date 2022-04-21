# Copyright (c) Meta Platforms, Inc

from typing import Callable, Iterable, Optional, Sequence

import torch
import torch.nn as nn
from flowtorch.nn.made import MaskedLinear
from flowtorch.parameters.base import Parameters


def _make_mask(shape: torch.Size, mask_type: str) -> torch.Tensor:
    if mask_type.startswith("neg_"):
        return _make_mask(shape, mask_type[4:])
    elif mask_type == "chessboard":
        z = torch.zeros(shape, dtype=torch.bool)
        z[:, ::2, ::2] = 1
        z[:, 1::2, 1::2] = 1
        return z
    elif mask_type == "quadrant":
        z = torch.zeros(shape, dtype=torch.bool)
        z[:, shape[1] // 2 :, : shape[2] // 2] = 1
        z[:, : shape[1] // 2, shape[2] // 2 :] = 1
        return z
    else:
        raise NotImplementedError(shape)


class DenseCoupling(Parameters):
    autoregressive = False

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
        input_dims = sum(input_shape)
        self.input_dims = input_dims
        if input_dims == 0:
            input_dims = 1  # scalars represented by torch.Size([])
        if permutation is None:
            # permutation will define the split of the input
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
            raise ValueError(
                "Coupling input_dim = 1. Coupling transforms require at least two features."
            )

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
        mask_input = mask_input[:, self.permutation]

        out_dims = input_dims * self.output_multiplier
        mask_output = torch.ones(
            self.output_multiplier, input_dims, hidden_dims[-1], dtype=torch.bool
        )
        mask_output[:, :x1_dim] = 0.0
        mask_output = mask_output[:, self.permutation]
        mask_output_reg = mask_output[0, :, 0]
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

        if self.skip_connections:
            self.skip_layer = MaskedLinear(
                input_dims,  # + context_dims,
                out_dims,
                mask_output,
                bias=False,
            )

        self.layers = nn.Sequential(*layers)
        self.register_buffer("mask_output", mask_output_reg.to(torch.bool))
        self._init_weights()

    def _init_weights(self) -> None:
        for layer in self.modules():
            if hasattr(layer, "weight"):
                layer.weight.data.normal_(0.0, 1e-3)  # type: ignore
            if hasattr(layer, "bias") and layer.bias is not None:
                layer.bias.data.fill_(0.0)  # type: ignore

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
        *input: torch.Tensor,
        inverse: bool,
        context: Optional[torch.Tensor] = None,
    ) -> Optional[Sequence[torch.Tensor]]:

        input = input[0]
        input_masked = input.masked_fill(self.mask_output, 0.0)  # type: ignore
        if context is not None:
            input_aug = torch.cat(
                [context.expand((*input.shape[:-1], -1)), input_masked], dim=-1
            )
        else:
            input_aug = input_masked

        h = self.layers(input_aug) + self.bias

        # TODO: Get skip_layers working again!
        if self.skip_connections:
            h = h + self.skip_layer(input_aug)

        # Shape the output
        h = h.view(*input.shape[:-1], self.output_multiplier, -1)

        result = h.unbind(-2)
        result = tuple(
            r.masked_fill(~self.mask_output.expand_as(r), 0.0) for r in result  # type: ignore
        )
        return result


class ConvCoupling(Parameters):
    autoregressive = False
    _mask_types = ["chessboard", "quadrants", "inv_chessboard", "inv_quadrants"]

    def __init__(
        self,
        param_shapes: Sequence[torch.Size],
        input_shape: torch.Size,
        context_shape: Optional[torch.Size],
        *,
        cnn_activate_input: bool = True,
        cnn_channels: int = 256,
        cnn_kernel: Sequence[int] = None,
        cnn_padding: Sequence[int] = None,
        cnn_stride: Sequence[int] = None,
        nonlinearity: Callable[[], nn.Module] = nn.ReLU,
        skip_connections: bool = False,
        mask_type: str = "chessboard",
    ) -> None:
        super().__init__(param_shapes, input_shape, context_shape)

        # Check consistency of input_shape with param_shapes
        # We need each param_shapes to match input_shape in
        # its leftmost dimensions
        for s in param_shapes:
            assert len(s) >= len(input_shape) and s[: len(input_shape)] == input_shape

        if cnn_kernel is None:
            cnn_kernel = [3, 1, 3]
        if cnn_padding is None:
            cnn_padding = [1, 0, 1]
        if cnn_stride is None:
            cnn_stride = [1, 1, 1]

        self.cnn_channels = cnn_channels
        self.cnn_activate_input = cnn_activate_input
        self.cnn_kernel = cnn_kernel
        self.cnn_padding = cnn_padding
        self.cnn_stride = cnn_stride

        self.nonlinearity = nonlinearity
        self.skip_connections = skip_connections
        self._build(input_shape, param_shapes, context_shape, mask_type)

    def _build(
        self,
        input_shape: torch.Size,  # something like [C, W, H]
        param_shapes: Sequence[torch.Size],  # something like [[C, W, H], [C, W, H]]
        context_shape: Optional[torch.Size],
        mask_type: str,
    ) -> None:

        mask = _make_mask(input_shape, mask_type)
        self.register_buffer("mask", mask)
        self.output_multiplier = len(param_shapes)

        out_channels, width, height = input_shape

        layers = []
        if self.cnn_activate_input:
            layers.append(self.nonlinearity())
        layers.append(
            nn.LazyConv2d(
                out_channels=self.cnn_channels,
                kernel_size=self.cnn_kernel[0],
                padding=self.cnn_padding[0],
                stride=self.cnn_stride[0],
            )
        )
        layers.append(self.nonlinearity())
        layers.append(
            nn.Conv2d(
                in_channels=self.cnn_channels,
                out_channels=self.cnn_channels,
                kernel_size=self.cnn_kernel[1],
                padding=self.cnn_padding[1],
                stride=self.cnn_stride[1],
            )
        )
        layers.append(self.nonlinearity())
        layers.append(
            nn.Conv2d(
                in_channels=self.cnn_channels,
                out_channels=out_channels * self.output_multiplier,
                kernel_size=self.cnn_kernel[2],
                padding=self.cnn_padding[2],
                stride=self.cnn_stride[2],
            )
        )

        self.layers = nn.Sequential(*layers)
        self._init_weights()

    def _init_weights(self) -> None:
        for layer in self.modules():
            if hasattr(layer, "weight"):
                layer.weight.data.normal_(0.0, 1e-3)  # type: ignore
            if hasattr(layer, "bias") and layer.bias is not None:
                layer.bias.data.fill_(0.0)  # type: ignore

    def _forward(
        self,
        *input: torch.Tensor,
        inverse: bool,
        context: Optional[torch.Tensor] = None,
    ) -> Optional[Sequence[torch.Tensor]]:

        input = input[0]
        unsqueeze = False
        if input.ndimension() == 3:
            # mostly for initialization
            unsqueeze = True
            input = input.unsqueeze(0)

        input_masked = input.masked_fill(self.mask, 0.0)  # type: ignore
        if context is not None:
            context_shape = [shape for shape in input_masked.shape]
            context_shape[-3] = context.shape[-3]
            input_aug = torch.cat(
                [context.expand(*context_shape), input_masked], dim=-1
            )
        else:
            input_aug = input_masked

        print(self.layers)
        h = self.layers(input_aug)

        if self.skip_connections:
            h = h + input_masked

        # Shape the output

        if unsqueeze:
            h = h.squeeze(0)
        result = h.chunk(2, -3)

        result = tuple(
            r.masked_fill(~self.mask.expand_as(r), 0.0) for r in result  # type: ignore
        )

        return result
