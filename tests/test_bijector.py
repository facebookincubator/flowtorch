# Copyright (c) FlowTorch Development Team. All Rights Reserved
# SPDX-License-Identifier: MIT


import torch
import torch.distributions as dist
import torch.optim

import flowtorch
import flowtorch.bijectors
import flowtorch.params


def test_bijector_constructor():
    param_fn = flowtorch.params.DenseAutoregressive()
    b = flowtorch.bijectors.AffineAutoregressive(param_fn=param_fn)
    assert b is not None


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
