# Copyright (c) FlowTorch Development Team. All Rights Reserved
# SPDX-License-Identifier: MIT

import numpy as np
import scipy.misc
import scipy.stats
import torch
import torch.distributions as dist
import torch.optim
from torch.distributions import constraints
from torch.distributions.utils import _standard_normal

import flowtorch
import flowtorch.bijectors as bijectors
import flowtorch.params


class NealsFunnel(dist.Distribution):
    """
    Neal's funnel.

    p(x,y) = N(y|0,3) N(x|0,exp(y/2))
    """

    support = constraints.real
    arg_constraints = {}

    def __init__(self, validate_args=None):
        d = 2
        batch_shape, event_shape = torch.Size([]), (d,)
        super(NealsFunnel, self).__init__(
            batch_shape, event_shape, validate_args=validate_args
        )

    def rsample(self, sample_shape=None):
        if not sample_shape:
            sample_shape = torch.Size()
        eps = _standard_normal(
            (sample_shape[0], 2), dtype=torch.float, device=torch.device("cpu")
        )
        z = torch.zeros(eps.shape)
        z[..., 1] = torch.tensor(3.0) * eps[..., 1]
        z[..., 0] = torch.exp(z[..., 1] / 2.0) * eps[..., 0]
        return z

    def log_prob(self, value):
        if self._validate_args:
            self._validate_sample(value)
        x = value[..., 0]
        y = value[..., 1]

        log_prob = dist.Normal(0, 3).log_prob(y)
        log_prob += dist.Normal(0, torch.exp(y / 2)).log_prob(x)

        return log_prob


def test_bijector_constructor():
    param_fn = flowtorch.params.DenseAutoregressive()
    b = flowtorch.bijectors.AffineAutoregressive(param_fn=param_fn)
    assert b is not None


def test_neals_funnel_vi():
    torch.manual_seed(42)
    nf = NealsFunnel()
    flow = flowtorch.bijectors.AffineAutoregressive(
        flowtorch.params.DenseAutoregressive()
    )
    tdist, params = flow(
        dist.Independent(dist.Normal(torch.zeros(2), torch.ones(2)), 1)
    )
    opt = torch.optim.Adam(params.parameters(), lr=1e-3)
    num_elbo_mc_samples = 100
    for _ in range(400):
        z0 = tdist.base_dist.rsample(sample_shape=(num_elbo_mc_samples,))
        zk = flow._forward(z0, params, context=torch.empty(0))
        ldj = flow._log_abs_det_jacobian(z0, zk, params, context=torch.empty(0))

        neg_elbo = -nf.log_prob(zk).sum()
        neg_elbo += tdist.base_dist.log_prob(z0).sum() - ldj.sum()
        neg_elbo /= num_elbo_mc_samples

        if not torch.isnan(neg_elbo):
            neg_elbo.backward()
            opt.step()
            opt.zero_grad()

    nf_samples = NealsFunnel().sample((20,)).squeeze().numpy()
    vi_samples = tdist.sample((20,)).detach().numpy()

    assert scipy.stats.ks_2samp(nf_samples[:, 0], vi_samples[:, 0]).pvalue >= 0.05
    assert scipy.stats.ks_2samp(nf_samples[:, 1], vi_samples[:, 1]).pvalue >= 0.05


def test_inv():
    flow = flowtorch.bijectors.AffineAutoregressive(
        flowtorch.params.DenseAutoregressive()
    )
    inv_flow = flow.inv()
    assert inv_flow.inv == flow

    base_dist = dist.Independent(dist.Normal(torch.zeros(2), torch.ones(2)), 1)
    tdist, params = flow(base_dist)
    inv_tdist, inv_params = inv_flow(base_dist)
    x = torch.zeros(1, 2)
    y = flow.forward(x, params, context=torch.empty(0))
    assert tdist.bijector.log_abs_det_jacobian(
        x, y, params, context=torch.empty(0)
    ) == -inv_tdist.bijector.log_abs_det_jacobian(
        y, x, inv_params, context=torch.empty(0)
    )


class TestBijectors:
    def test_affine_autoregressive_jacobian(self):
        # Define plan for flow
        flow = bijectors.AffineAutoregressive()
        event_dim = min(flow.domain.event_dim, 1)
        event_shape = event_dim * [4]
        base_dist = dist.Normal(torch.zeros(event_shape), torch.ones(event_shape))

        # Instantiate transformed distribution and parameters
        _, params = flow(base_dist)

        # Calculate auto-diff Jacobian
        x = torch.randn(1, *event_shape)
        y = flow.forward(x, params)
        if event_dim == 1:
            analytic_ldt = flow.log_abs_det_jacobian(x, y, params).data
        else:
            analytic_ldt = flow.log_abs_det_jacobian(x, y, params).sum(-1).data

        # Calculate numerical Jacobian
        # TODO: Better way to get all indices of array/tensor?
        jacobian = torch.zeros(event_shape * 2)
        idxs = np.nonzero(np.ones(event_shape * 2))

        print('indices', idxs)

        # TODO: Vectorize numerical calculation of Jacobian with PyTorch
        # TODO: Break this out into flowtorch.numerical.derivatives.jacobian
        epsilon = 1e-4
        for idx in idxs:
            for jdx in idxs:
                epsilon_vector = torch.zeros(event_shape)
                epsilon_vector[idx] = epsilon
                # TODO: Use scipy.misc.derivative or another library's function?
                delta = (flow.forward(x + 0.5 * epsilon_vector, params) - flow.forward(x - 0.5 * epsilon_vector, params)) / epsilon
                print(idx, jdx, jacobian.shape)
                jacobian[idx + jdx] = float(delta[jdx].data.sum())
                

        print('analytic ldt', analytic_ldt)
        print('numerical ldt', jacobian)


    def _test_jacobian(self, input_dim, transform):
        """jacobian = torch.zeros(input_dim, input_dim)

        def nonzero(x):
            return torch.sign(torch.abs(x))

        x = torch.randn(1, input_dim)
        y = transform(x)
        if transform.event_dim == 1:
            analytic_ldt = transform.log_abs_det_jacobian(x, y).data
        else:
            analytic_ldt = transform.log_abs_det_jacobian(x, y).sum(-1).data

        for j in range(input_dim):
            for k in range(input_dim):
                epsilon_vector = torch.zeros(1, input_dim)
                epsilon_vector[0, j] = self.epsilon
                delta = (transform(x + 0.5 * epsilon_vector) - transform(x - 0.5 * epsilon_vector)) / self.epsilon
                jacobian[j, k] = float(delta[0, k].data.sum())

        # Apply permutation for autoregressive flows with a network
        if hasattr(transform, 'arn') and 'get_permutation' in dir(transform.arn):
            permutation = transform.arn.get_permutation()
            permuted_jacobian = jacobian.clone()
            for j in range(input_dim):
                for k in range(input_dim):
                    permuted_jacobian[j, k] = jacobian[permutation[j], permutation[k]]
            jacobian = permuted_jacobian

        # For autoregressive flow, Jacobian is sum of diagonal, otherwise need full determinate
        if hasattr(transform, 'autoregressive') and transform.autoregressive:
            numeric_ldt = torch.sum(torch.log(torch.diag(jacobian)))
        else:
            numeric_ldt = torch.log(torch.abs(jacobian.det()))

        ldt_discrepancy = (analytic_ldt - numeric_ldt).abs()
        assert ldt_discrepancy < self.epsilon

        # Test that lower triangular with unit diagonal for autoregressive flows
        if hasattr(transform, 'autoregressive'):
            diag_sum = torch.sum(torch.diag(nonzero(jacobian)))
            lower_sum = torch.sum(torch.tril(nonzero(jacobian), diagonal=-1))
            assert diag_sum == float(input_dim)
            assert lower_sum == float(0.0)"""

    # TODO: Only run test inverse when not an abstract method (auto-detect this)
    def _test_inverse(self, shape, transform):
        pass

    def _test_shape(self, base_shape, transform):
        pass

    def _test_autodiff(self, input_dim, transform, inverse=False):
        pass

tb = TestBijectors()
tb.test_affine_autoregressive_jacobian()