# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved
# SPDX-License-Identifier: MIT
import flowtorch
import flowtorch.bijectors as bijectors
import flowtorch.params
import numpy as np
import pytest
import torch
import torch.distributions as dist
import torch.optim


def test_bijector_constructor():
    param_fn = flowtorch.params.DenseAutoregressive()
    b = flowtorch.bijectors.AffineAutoregressive(param_fn=param_fn)
    assert b is not None


@pytest.fixture(params=[bij_name for _, bij_name in bijectors.standard_bijectors])
def flow(request):
    flow = request.param()
    return flow


def test_jacobian(flow, epsilon=1e-3):
    # TODO: The following pattern to produce the standard bijector should
    # be factored out
    # Define plan for flow
    event_dim = max(flow.domain.event_dim, 1)
    event_shape = event_dim * [4]
    base_dist = dist.Normal(torch.zeros(event_shape), torch.ones(event_shape))

    # Instantiate transformed distribution and parameters
    _ = flow(base_dist)
    params = flow.params

    # Calculate auto-diff Jacobian
    x = torch.randn(*event_shape)
    x = torch.distributions.transform_to(flow.domain)(x)
    y = flow.forward(x)
    if flow.domain.event_dim == 1:
        analytic_ldt = flow.log_abs_det_jacobian(x, y).data
    else:
        analytic_ldt = flow.log_abs_det_jacobian(x, y).sum(-1).data

    # Calculate numerical Jacobian
    # TODO: Better way to get all indices of array/tensor?
    jacobian = torch.zeros(event_shape * 2)
    idxs = np.nonzero(np.ones(event_shape))

    # Have to permute elements for MADE
    count_vars = len(idxs[0])
    if hasattr(params, "permutation"):
        inv_permutation = np.zeros(count_vars, dtype=int)
        inv_permutation[params.permutation] = np.arange(count_vars)

    # TODO: Vectorize numerical calculation of Jacobian with PyTorch
    # TODO: Break this out into flowtorch.numerical.derivatives.jacobian
    for var_idx in range(count_vars):
        idx = [dim_idx[var_idx] for dim_idx in idxs]
        epsilon_vector = torch.zeros(event_shape)
        epsilon_vector[(*idx,)] = epsilon
        # TODO: Use scipy.misc.derivative or another library's function?
        delta = (
            flow.forward(x + 0.5 * epsilon_vector)
            - flow.forward(x - 0.5 * epsilon_vector)
        ) / epsilon

        for var_jdx in range(count_vars):
            jdx = [dim_jdx[var_jdx] for dim_jdx in idxs]

            # Have to account for permutation potentially introduced by MADE network
            # TODO: Make this more general with structure abstraction
            if hasattr(params, "permutation"):
                jacobian[(inv_permutation[idx[0]], inv_permutation[jdx[0]])] = float(
                    delta[(Ellipsis, *jdx)].data.sum()
                )
            else:
                jacobian[(*idx, *jdx)] = float(delta[(Ellipsis, *jdx)].data.sum())

    # For autoregressive flow, Jacobian is sum of diagonal, otherwise need full
    # determinate
    if hasattr(params, "permutation"):
        numeric_ldt = torch.sum(torch.log(torch.diag(jacobian)))
    else:
        numeric_ldt = torch.log(torch.abs(jacobian.det()))

    ldt_discrepancy = (analytic_ldt - numeric_ldt).abs()
    assert ldt_discrepancy < epsilon

    # Test that lower triangular with non-zero diagonal for autoregressive flows
    if hasattr(params, "permutation"):

        def nonzero(x):
            return torch.sign(torch.abs(x))

        diag_sum = torch.sum(torch.diag(nonzero(jacobian)))
        lower_sum = torch.sum(torch.tril(nonzero(jacobian), diagonal=-1))
        assert diag_sum == float(count_vars)
        assert lower_sum == float(0.0)


def test_inverse(flow, epsilon=1e-6):
    # Define plan for flow
    event_dim = max(flow.domain.event_dim, 1)
    event_shape = event_dim * [4]
    base_dist = dist.Normal(torch.zeros(event_shape), torch.ones(event_shape))

    # Instantiate transformed distribution and parameters
    _ = flow(base_dist)

    # Test g^{-1}(g(x)) = x
    x_true = base_dist.sample(torch.Size([10]))
    x_true = torch.distributions.transform_to(flow.domain)(x_true)

    y = flow._forward(x_true)
    x_calculated = flow._inverse(y)
    assert (x_true - x_calculated).abs().max().item() < epsilon

    # Test that Jacobian after inverse op is same as after forward
    J_1 = flow.log_abs_det_jacobian(x_true, y)
    J_2 = flow.log_abs_det_jacobian(x_calculated, y)
    assert (J_1 - J_2).abs().max().item() < epsilon


# TODO
def _test_shape(self, base_shape, transform):
    pass


# TODO: This tests whether can take autodiff gradient without exception
def _test_autodiff(self, input_dim, transform, inverse=False):
    pass
