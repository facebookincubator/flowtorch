# Copyright (c) FlowTorch Development Team. All Rights Reserved
# SPDX-License-Identifier: MIT

"""
My intention is that this module will contain functions and classes for constructing
parameters for bijections like

* Constant
* Dense
* DenseAutoregressive
* LstmAutoregressive

etc.

These can be constant parameters that do not vary for
different inputs, as well as parameters that are the output of "hypernets" such
as autoregressive neural networks being a function of the input random variable.

In all cases, they should be callables that you can pass to classes deriving from
Bijector. They encapsulate *all* of the state for a bijection.

Parameters should take care of caching when necessary.

"""

from flowtorch.params.dense_autoregressive import DenseAutoregressive

__all__ = [
    "DenseAutoregressive",
]
