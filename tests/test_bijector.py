# Copyright (c) FlowTorch Development Team. All Rights Reserved
# SPDX-License-Identifier: MIT

import scipy.stats
import torch
import torch.distributions as dist
import torch.optim
from torch.distributions import constraints
from torch.distributions.utils import _standard_normal

import flowtorch
import flowtorch.bijectors
import flowtorch.params


class NealsFunnel(dist.Distribution):
    """
    Neal's funnel.

    p(x,y) = N(y|0,3) N(x|0,exp(y/2))
    """

    support = constraints.real

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
    num_elbo_mc_samples = 1000
    for _ in range(300):
        z0 = tdist.base_dist.rsample(sample_shape=(num_elbo_mc_samples,))
        zk = flow._forward(z0, params)
        ldj = flow._log_abs_det_jacobian(z0, zk, params)

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


# class TestClass:
#     def test_shapes(self):
#         """
#         Tests output shapes of bijector
#         """

#         assert "h" in x
