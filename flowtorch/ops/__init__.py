# Copyright (c) Meta Platforms, Inc

from typing import Optional, Tuple

import torch
import torch.nn.functional as F

eps = 1e-8


def clamp_preserve_gradients(x: torch.Tensor, min: float, max: float) -> torch.Tensor:
    """
    This helper function clamps gradients but still passes through the
    gradient in clamped regions
    """
    return x + (x.clamp(min, max) - x).detach()


def clipped_sigmoid(x: torch.Tensor) -> torch.Tensor:
    finfo = torch.finfo(x.dtype)
    return torch.clamp(torch.sigmoid(x), min=finfo.tiny, max=1.0 - finfo.eps)


def softplus_inv(y: torch.Tensor) -> torch.Tensor:
    return y + y.neg().expm1().neg().log()


def _searchsorted(sorted_sequence: torch.Tensor, values: torch.Tensor) -> torch.Tensor:
    """
    Searches for which bin an input belongs to (in a way that is parallelizable and
    amenable to autodiff)
    TODO: Replace with torch.searchsorted once it is released
    """
    return torch.sum(values[..., None] >= sorted_sequence, dim=-1) - 1


def _select_bins(x: torch.Tensor, idx: torch.Tensor) -> torch.Tensor:
    """
    Performs gather to select the bin in the correct way on batched inputs
    """
    idx = idx.clamp(min=0, max=x.size(-1) - 1)

    # Broadcast dimensions of idx over x
    # idx ~ (batch_dims, input_dim, 1)
    # x ~ (context_batch_dims, input_dim, count_bins)
    # Note that by convention, the context variable batch dimensions must broadcast
    # over the input batch dimensions.
    if len(idx.shape) >= len(x.shape):
        x = x.reshape((1,) * (len(idx.shape) - len(x.shape)) + x.shape)
        x = x.expand(idx.shape[:-2] + (-1,) * 2)

    return x.gather(-1, idx).squeeze(-1)


def _calculate_knots(
    lengths: torch.Tensor, lower: float, upper: float
) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Given a tensor of unscaled bin lengths that sum to 1, plus the lower and upper
    limits, returns the shifted and scaled lengths plus knot positions
    """

    # Cumulative widths gives x (y for inverse) position of knots
    knots = torch.cumsum(lengths, dim=-1)

    # Pad left of last dimension with 1 zero to compensate for dim lost to cumsum
    knots = F.pad(knots, pad=(1, 0), mode="constant", value=0.0)

    # Translate [0,1] knot points to [-B, B]
    knots = (upper - lower) * knots + lower

    # Convert the knot points back to lengths
    # NOTE: Are following two lines a necessary fix for accumulation (round-off) error?
    knots[..., 0] = lower
    knots[..., -1] = upper
    lengths = knots[..., 1:] - knots[..., :-1]

    return lengths, knots


def monotonic_rational_spline(
    inputs: torch.Tensor,
    widths: torch.Tensor,
    heights: torch.Tensor,
    derivatives: torch.Tensor,
    lambdas: Optional[torch.Tensor] = None,
    inverse: bool = False,
    bound: float = 3.0,
    min_bin_width: float = 1e-3,
    min_bin_height: float = 1e-3,
    min_derivative: float = 1e-3,
    min_lambda: float = 0.025,
    eps: float = 1e-6,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Calculating a monotonic rational spline (linear or quadratic) or its inverse,
    plus the log(abs(detJ)) required for normalizing flows.
    NOTE: I omit the docstring with parameter descriptions for this method since it
    is not considered "public" yet!
    """

    # Ensure bound is positive
    # NOTE: For simplicity, we apply the identity function outside [-B, B] X [-B, B]
    # rather than allowing arbitrary corners to the bounding box. If you want a
    # different bounding box you can apply an affine transform before and after the
    # input.
    assert bound > 0.0

    num_bins = widths.shape[-1]
    if min_bin_width * num_bins > 1.0:
        raise ValueError("Minimal bin width too large for the number of bins")
    if min_bin_height * num_bins > 1.0:
        raise ValueError("Minimal bin height too large for the number of bins")

    # inputs, inside_interval_mask, outside_interval_mask ~ (batch_dim, input_dim)
    left, right = -bound, bound
    bottom, top = -bound, bound
    inside_interval_mask = (inputs >= left) & (inputs <= right)
    outside_interval_mask = ~inside_interval_mask

    # outputs, logabsdet ~ (batch_dim, input_dim)
    outputs = torch.zeros_like(inputs)
    logabsdet = torch.zeros_like(inputs)

    # For numerical stability, put lower/upper limits on parameters. E.g. give
    # every bin min_bin_width, then add width fraction of remaining length.
    # NOTE: Do this here rather than higher up because we want everything to
    # ensure numerical stability within this function.
    widths = min_bin_width + (1.0 - min_bin_width * num_bins) * widths
    heights = min_bin_height + (1.0 - min_bin_height * num_bins) * heights
    derivatives = min_derivative + derivatives

    # Cumulative widths are x (y for inverse) position of knots
    # Similarly, cumulative heights are y (x for inverse) position of knots
    widths, cumwidths = _calculate_knots(widths, left, right)
    heights, cumheights = _calculate_knots(heights, bottom, top)

    # Pad left and right derivatives with fixed values at first and last knots
    # These are 1 since the function is the identity outside the bounding box
    # and the derivative is continuous.
    # NOTE: Not sure why this is 1.0 - min_derivative rather than 1.0. I've
    # copied this from original implementation
    derivatives = F.pad(
        derivatives, pad=(1, 1), mode="constant", value=1.0 - min_derivative
    )

    # Get the index of the bin that each input is in
    # bin_idx ~ (batch_dim, input_dim, 1)
    bin_idx = _searchsorted(
        cumheights + eps if inverse else cumwidths + eps, inputs
    ).unsqueeze(-1)

    # Select the value for the relevant bin for the variables
    # used in the main calculation
    input_widths = _select_bins(widths, bin_idx)
    input_cumwidths = _select_bins(cumwidths, bin_idx)
    input_cumheights = _select_bins(cumheights, bin_idx)
    input_delta = _select_bins(heights / widths, bin_idx)
    input_derivatives = _select_bins(derivatives, bin_idx)
    input_derivatives_plus_one = _select_bins(derivatives[..., 1:], bin_idx)
    input_heights = _select_bins(heights, bin_idx)

    # Calculate monotonic *linear* rational spline
    if lambdas is not None:
        lambdas = (1 - 2 * min_lambda) * lambdas + min_lambda
        input_lambdas = _select_bins(lambdas, bin_idx)

        # The weight, w_a, at the left-hand-side of each bin
        # We are free to choose w_a, so set it to 1
        wa = 1.0

        # The weight, w_b, at the right-hand-side of each bin
        # This turns out to be a multiple of the w_a
        # TODO: Should this be done in log space for numerical stability?
        wb = torch.sqrt(input_derivatives / input_derivatives_plus_one) * wa

        # The weight, w_c, at the division point of each bin
        # Recall that each bin is divided into two parts so we have enough
        # d.o.f. to fit spline
        wc = (
            input_lambdas * wa * input_derivatives
            + (1 - input_lambdas) * wb * input_derivatives_plus_one
        ) / input_delta

        # Calculate y coords of bins
        ya = input_cumheights
        yb = input_heights + input_cumheights
        yc = ((1.0 - input_lambdas) * wa * ya + input_lambdas * wb * yb) / (
            (1.0 - input_lambdas) * wa + input_lambdas * wb
        )

        if inverse:
            numerator = (input_lambdas * wa * (ya - inputs)) * (
                inputs <= yc
            ).float() + (
                (wc - input_lambdas * wb) * inputs + input_lambdas * wb * yb - wc * yc
            ) * (
                inputs > yc
            ).float()

            denominator = ((wc - wa) * inputs + wa * ya - wc * yc) * (
                inputs <= yc
            ).float() + ((wc - wb) * inputs + wb * yb - wc * yc) * (inputs > yc).float()

            theta = numerator / denominator

            outputs = theta * input_widths + input_cumwidths

            derivative_numerator = (
                wa * wc * input_lambdas * (yc - ya) * (inputs <= yc).float()
                + wb * wc * (1 - input_lambdas) * (yb - yc) * (inputs > yc).float()
            ) * input_widths

            logabsdet = torch.log(derivative_numerator) - 2 * torch.log(
                torch.abs(denominator)
            )

        else:
            theta = (inputs - input_cumwidths) / input_widths

            numerator = (wa * ya * (input_lambdas - theta) + wc * yc * theta) * (
                theta <= input_lambdas
            ).float() + (wc * yc * (1 - theta) + wb * yb * (theta - input_lambdas)) * (
                theta > input_lambdas
            ).float()

            denominator = (wa * (input_lambdas - theta) + wc * theta) * (
                theta <= input_lambdas
            ).float() + (wc * (1 - theta) + wb * (theta - input_lambdas)) * (
                theta > input_lambdas
            ).float()

            outputs = numerator / denominator

            derivative_numerator = (
                wa * wc * input_lambdas * (yc - ya) * (theta <= input_lambdas).float()
                + wb
                * wc
                * (1 - input_lambdas)
                * (yb - yc)
                * (theta > input_lambdas).float()
            ) / input_widths

            logabsdet = torch.log(derivative_numerator) - 2 * torch.log(
                torch.abs(denominator)
            )

    # Calculate monotonic *quadratic* rational spline
    else:
        if inverse:
            a = (inputs - input_cumheights) * (
                input_derivatives + input_derivatives_plus_one - 2 * input_delta
            ) + input_heights * (input_delta - input_derivatives)
            b = input_heights * input_derivatives - (inputs - input_cumheights) * (
                input_derivatives + input_derivatives_plus_one - 2 * input_delta
            )
            c = -input_delta * (inputs - input_cumheights)

            discriminant = b.pow(2) - 4 * a * c
            assert (discriminant >= 0).all()

            root = (2 * c) / (-b - torch.sqrt(discriminant))
            outputs = root * input_widths + input_cumwidths

            theta_one_minus_theta = root * (1 - root)
            denominator = input_delta + (
                (input_derivatives + input_derivatives_plus_one - 2 * input_delta)
                * theta_one_minus_theta
            )
            derivative_numerator = input_delta.pow(2) * (
                input_derivatives_plus_one * root.pow(2)
                + 2 * input_delta * theta_one_minus_theta
                + input_derivatives * (1 - root).pow(2)
            )
            logabsdet = -(torch.log(derivative_numerator) - 2 * torch.log(denominator))

        else:
            theta = (inputs - input_cumwidths) / input_widths
            theta_one_minus_theta = theta * (1 - theta)

            numerator = input_heights * (
                input_delta * theta.pow(2) + input_derivatives * theta_one_minus_theta
            )
            denominator = input_delta + (
                (input_derivatives + input_derivatives_plus_one - 2 * input_delta)
                * theta_one_minus_theta
            )
            outputs = input_cumheights + numerator / denominator

            derivative_numerator = input_delta.pow(2) * (
                input_derivatives_plus_one * theta.pow(2)
                + 2 * input_delta * theta_one_minus_theta
                + input_derivatives * (1 - theta).pow(2)
            )
            logabsdet = torch.log(derivative_numerator) - 2 * torch.log(denominator)

    # Apply the identity function outside the bounding box
    outputs[outside_interval_mask] = inputs[outside_interval_mask]
    logabsdet[outside_interval_mask] = 0.0
    return outputs, logabsdet
