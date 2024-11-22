# Copyright (c) Meta Platforms, Inc

# pyre-unsafe

import flowtorch.bijectors as bijs
import flowtorch.distributions as dist
import flowtorch.parameters as params
import torch
import torch.distributions
import torch.optim


def test_compose():
    transforms = bijs.Compose(
        bijectors=[
            bijs.AffineAutoregressive(
                params.DenseAutoregressive(),
            ),
            bijs.AffineAutoregressive(
                params.DenseAutoregressive(),
            ),
            bijs.AffineAutoregressive(
                params.DenseAutoregressive(),
            ),
        ]
    )

    event_shape = (5,)
    base_dist = torch.distributions.Independent(
        torch.distributions.Normal(
            loc=torch.zeros(event_shape), scale=torch.ones(event_shape)
        ),
        len(event_shape),
    )
    flow = dist.Flow(base_dist, transforms)

    optimizer = torch.optim.Adam(flow.parameters())
    assert optimizer.param_groups[0]["params"][0].grad is None
    flow.log_prob(torch.randn((100,) + event_shape)).sum().backward()
    assert optimizer.param_groups[0]["params"][0].grad.abs().sum().item() > 1e-3
    optimizer.zero_grad()
