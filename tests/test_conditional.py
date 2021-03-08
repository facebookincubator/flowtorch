# Copyright (c) FlowTorch Development Team. All Rights Reserved
# SPDX-License-Identifier: MIT

import torch
import torch.distributions as dist
import torch.optim

import flowtorch.bijectors
import flowtorch.params


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

    opt = torch.optim.Adam(params_module.parameters(), lr=5e-3)

    for idx in range(501):
        opt.zero_grad()

        if idx % 2 == 0:
            target_dist = target_dist_0
            context = torch.ones(context_size)
        else:
            target_dist = target_dist_1
            context = -1 * torch.ones(context_size)

        marginal = new_cond_dist.condition(context)
        y = marginal.rsample((1000,))
        loss = -target_dist.log_prob(y) + marginal.log_prob(y)
        loss = loss.mean()

        if idx % 100 == 0:
            print("epoch", idx, "loss", loss)

        loss.backward()
        opt.step()

    assert (
        new_cond_dist.condition(torch.ones(context_size)).sample((1000,)).mean() - 5.0
    ).norm().item() < 0.1
    assert (
        new_cond_dist.condition(-1 * torch.ones(context_size)).sample((1000,)).mean()
        + 5.0
    ).norm().item() < 0.1
