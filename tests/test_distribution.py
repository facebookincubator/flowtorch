# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved
# SPDX-License-Identifier: MIT

import flowtorch.bijectors
import flowtorch.params
import scipy.stats
import torch
import torch.distributions as dist
import torch.optim
from torch.distributions import constraints
from torch.distributions.utils import _standard_normal


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


def test_conditional_2gmm():
    context_size = 2

    flow = flowtorch.bijectors.Compose(
        [
            flowtorch.bijectors.AffineAutoregressive(context_size=context_size)
            for _ in range(2)
        ],
        context_size=context_size,
    ).inv()

    base_dist = dist.Normal(torch.zeros(2), torch.ones(2))
    new_cond_dist, params_module = flow(base_dist)

    target_dist_0 = dist.Independent(
        dist.Normal(torch.zeros(2) + 5, torch.ones(2) * 0.5), 1
    )
    target_dist_1 = dist.Independent(
        dist.Normal(torch.zeros(2) - 5, torch.ones(2) * 0.5), 1
    )

    opt = torch.optim.Adam(params_module.parameters(), lr=1e-3)

    for idx in range(101):
        opt.zero_grad()

        if idx % 2 == 0:
            target_dist = target_dist_0
            context = torch.ones(context_size)
        else:
            target_dist = target_dist_1
            context = -1 * torch.ones(context_size)

        marginal = new_cond_dist.condition(context)
        y = marginal.rsample((100,))
        loss = -target_dist.log_prob(y) + marginal.log_prob(y)
        loss = loss.mean()

        if idx % 100 == 0:
            print("epoch", idx, "loss", loss)

        loss.backward()
        opt.step()

    assert (
        new_cond_dist.condition(torch.ones(context_size)).sample((1000,)).mean() - 5.0
    ).norm().item() < 1.0
    assert (
        new_cond_dist.condition(-1 * torch.ones(context_size)).sample((1000,)).mean()
        + 5.0
    ).norm().item() < 1.0
