# Copyright (c) Meta Platforms, Inc

# pyre-unsafe

import flowtorch.bijectors as bijs
import flowtorch.distributions as dist
import flowtorch.parameters as params
import scipy.stats
import torch
import torch.distributions
import torch.optim


def test_tdist_standalone():
    input_dim = 3

    def make_tdist():
        # train a flow here
        base_dist = torch.distributions.Independent(
            torch.distributions.Normal(torch.zeros(input_dim), torch.ones(input_dim)), 1
        )
        bijector = bijs.AffineAutoregressive()
        tdist = dist.Flow(base_dist, bijector)
        return tdist

    tdist = make_tdist()
    tdist.log_prob(torch.randn(input_dim))  # should run without error
    assert True


def test_neals_funnel_vi():
    torch.manual_seed(42)
    nf = dist.NealsFunnel()
    bijector = bijs.AffineAutoregressive(params_fn=params.DenseAutoregressive())

    base_dist = torch.distributions.Independent(
        torch.distributions.Normal(torch.zeros(2), torch.ones(2)), 1
    )
    flow = dist.Flow(base_dist, bijector)
    bijector = flow.bijector

    opt = torch.optim.Adam(flow.parameters(), lr=2e-3)
    num_elbo_mc_samples = 200
    for _ in range(100):
        z0 = flow.base_dist.rsample(sample_shape=(num_elbo_mc_samples,))
        zk = bijector.forward(z0)
        ldj = zk._log_detJ

        neg_elbo = -nf.log_prob(zk).sum()
        neg_elbo += flow.base_dist.log_prob(z0).sum() - ldj.sum()
        neg_elbo /= num_elbo_mc_samples

        if not torch.isnan(neg_elbo):
            neg_elbo.backward()
            opt.step()
            opt.zero_grad()

    nf_samples = dist.NealsFunnel().sample((20,)).squeeze().numpy()
    vi_samples = flow.sample((20,)).detach().numpy()

    assert scipy.stats.ks_2samp(nf_samples[:, 0], vi_samples[:, 0]).pvalue >= 0.05
    assert scipy.stats.ks_2samp(nf_samples[:, 1], vi_samples[:, 1]).pvalue >= 0.05


"""
def test_conditional_2gmm():
    context_size = 2

    flow = bijs.Compose(
        bijectors=[
            bijs.AffineAutoregressive(context_size=context_size)
            for _ in range(2)
        ],
        context_size=context_size,
    ).inv()

    base_dist = dist.Normal(torch.zeros(2), torch.ones(2))
    new_cond_dist = flow(base_dist)
    flow = new_cond_dist.bijector

    target_dist_0 = dist.Independent(
        dist.Normal(torch.zeros(2) + 5, torch.ones(2) * 0.5), 1
    )
    target_dist_1 = dist.Independent(
        dist.Normal(torch.zeros(2) - 5, torch.ones(2) * 0.5), 1
    )

    opt = torch.optim.Adam(flow.params.parameters(), lr=1e-3)

    for idx in range(100):
        opt.zero_grad()

        if idx % 2 == 0:
            target_dist = target_dist_0
            context = torch.ones(context_size)
        else:
            target_dist = target_dist_1
            context = -1 * torch.ones(context_size)

        marginal = new_cond_dist.condition(context)
        y = marginal.rsample((50,))
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
"""
