# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved
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

from flowtorch.params.base import Params, ParamsModule, ParamsModuleList
from flowtorch.params.dense_autoregressive import DenseAutoregressive
from flowtorch.params.empty import Empty
from flowtorch.params.tensor import Tensor

__all__ = [
    "DenseAutoregressive",
    "Empty",
    "Params",
    "ParamsModule",
    "ParamsModuleList",
    "Tensor",
]
