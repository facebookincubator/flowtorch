# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved
# SPDX-License-Identifier: MIT

import flowtorch
import flowtorch.bijectors
import flowtorch.params
import torch
import torch.distributions as dist
import torch.optim


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
    inv_tdist, _ = inv_flow(base_dist)
    x = torch.zeros(1, 2)
    y = flow.forward(x, params, context=torch.empty(0))

    # TODO: Check that f^-1(f(x)) approx x

    # Note that the second argment from calling the flow will be a new NN each time
    # so we need to pass `params` to both methods
    assert tdist.bijector.log_abs_det_jacobian(
        x, y, params, context=torch.empty(0)
    ) == -inv_tdist.bijector.log_abs_det_jacobian(y, x, params, context=torch.empty(0))
