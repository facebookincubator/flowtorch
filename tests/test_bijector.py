# Copyright (c) FlowTorch Development Team. All Rights Reserved
# SPDX-License-Identifier: MIT

import flowtorch
import flowtorch.bijectors
import flowtorch.params


def test_bijector_constructor():
    param_fn = flowtorch.params.DenseAutoregressive()
    b = flowtorch.bijectors.AffineAutoregressive(param_fn=param_fn)
    assert b is not None
