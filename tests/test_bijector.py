# Copyright (c) Meta Platforms, Inc
import warnings

import flowtorch.bijectors as bijectors
import flowtorch.parameters as params

import numpy as np
import pytest
import torch
import torch.distributions as dist
import torch.optim
from flowtorch.bijectors import AffineAutoregressive, Compose
from flowtorch.distributions import Flow

"""
def test_bijector_constructor():
    param_fn = flowtorch.params.DenseAutoregressive()
    b = flowtorch.bijectors.AffineAutoregressive(param_fn=param_fn)
    assert b is not None
"""


@pytest.fixture(params=[bij_name for _, bij_name in bijectors.standard_bijectors])
def flow(request):
    bij = request.param
    event_dim = max(bij.domain.event_dim, 1)
    event_shape = event_dim * [3]
    base_dist = dist.Independent(
        dist.Normal(torch.zeros(event_shape), torch.ones(event_shape)), event_dim
    )

    flow = Flow(base_dist, bij)
    return flow


def test_jacobian(flow, epsilon=1e-2):
    # Instantiate transformed distribution and parameters
    bij = flow.bijector
    params = bij._params_fn

    # Calculate auto-diff Jacobian
    x = torch.randn(*flow.event_shape)
    x = torch.distributions.transform_to(bij.domain)(x)
    y = bij.forward(x)
    if bij.domain.event_dim == 1:
        analytic_ldt = bij.log_abs_det_jacobian(x, y).data
    else:
        analytic_ldt = bij.log_abs_det_jacobian(x, y).sum(-1).data

    # Calculate numerical Jacobian
    # TODO: Better way to get all indices of array/tensor?
    jacobian = torch.zeros(flow.event_shape * 2)
    idxs = np.nonzero(np.ones(flow.event_shape))

    # Have to permute elements for MADE
    count_vars = len(idxs[0])
    if hasattr(params, "permutation"):
        inv_permutation = np.zeros(count_vars, dtype=int)
        inv_permutation[params.permutation] = np.arange(count_vars)

    # TODO: Vectorize numerical calculation of Jacobian with PyTorch
    # TODO: Break this out into flowtorch.numerical.derivatives.jacobian
    for var_idx in range(count_vars):
        idx = [dim_idx[var_idx] for dim_idx in idxs]
        epsilon_vector = torch.zeros(flow.event_shape)
        epsilon_vector[(*idx,)] = epsilon
        # TODO: Use scipy.misc.derivative or another library's function?
        delta = (
                    bij.forward(x + 0.5 * epsilon_vector)
                    - bij.forward(x - 0.5 * epsilon_vector)
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


def test_inverse(flow, epsilon=1e-5):
    bij = flow.bijector
    base_dist = flow.base_dist

    # Test g^{-1}(g(x)) = x
    x_true = base_dist.sample(torch.Size([10]))
    x_true = torch.distributions.transform_to(bij.domain)(x_true)

    y = bij.forward(x_true)
    x_calculated = bij.inverse(y)
    assert (x_true - x_calculated).abs().max().item() < epsilon

    # Test that Jacobian after inverse op is same as after forward
    J_1 = bij.log_abs_det_jacobian(x_true, y)
    J_2 = bij.log_abs_det_jacobian(x_calculated, y)
    assert (J_1 - J_2).abs().max().item() < epsilon


def get_net() -> AffineAutoregressive:
    ar = Compose(
        [
            AffineAutoregressive(params.DenseAutoregressive()),
            AffineAutoregressive(params.DenseAutoregressive()),
            AffineAutoregressive(params.DenseAutoregressive()),
        ]
    )
    ar = ar(
        shape=torch.Size(
            [
                100,
            ]
        )
    )
    return ar


def test_invert():
    net = get_net()

    assert net.invert().invert() is net

    x = torch.randn(50, 100, requires_grad=True)
    inv_net = net.invert()
    torch.testing.assert_allclose(inv_net.forward(x), net.inverse(x))

    y = inv_net.forward(x)

    # checks that no warning is displayed, which can happen if no cache is used
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        inv_net.log_abs_det_jacobian(x, y)

    with pytest.warns(UserWarning):
        y_det = y.detach_from_flow()
        inv_net.log_abs_det_jacobian(x, y_det)

    y = y.detach_from_flow()
    torch.testing.assert_allclose(
        inv_net.log_abs_det_jacobian(x, y), -net.log_abs_det_jacobian(y, x)
    )


"""
# TODO
def _test_shape(self, base_shape, transform):
    pass


# TODO: This tests whether can take autodiff gradient without exception
def _test_autodiff(self, input_dim, transform, inverse=False):
    pass
"""
