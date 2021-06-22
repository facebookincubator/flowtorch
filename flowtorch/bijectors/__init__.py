# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved
# SPDX-License-Identifier: MIT

import inspect
from typing import List, Tuple, Type

from flowtorch import Bijector

# TODO: Autogenerate this from script!
from flowtorch.bijectors.affine_autoregressive import AffineAutoregressive
from flowtorch.bijectors.affine_fixed import AffineFixed
from flowtorch.bijectors.compose import Compose
from flowtorch.bijectors.elu import ELU
from flowtorch.bijectors.exp import Exp
from flowtorch.bijectors.fixed import Fixed
from flowtorch.bijectors.leaky_relu import LeakyReLU
from flowtorch.bijectors.permute import Permute
from flowtorch.bijectors.power import Power
from flowtorch.bijectors.sigmoid import Sigmoid
from flowtorch.bijectors.softplus import Softplus
from flowtorch.bijectors.tanh import Tanh
from flowtorch.bijectors.volume_preserving import VolumePreserving

# "Meta bijectors" are classes that descend from flowtorch.Bijector but either
# are not used directly by the user, or are used to operate on other bijectors.
# We have to write special units tests for these.
meta_bijectors: List[Tuple[str, Type[Bijector]]] = [
    ("Compose", Compose),
    ("Fixed", Fixed),
    ("VolumePreserving", VolumePreserving),
]


def isbijector(cls):
    # A class must inherit from flowtorch.Bijector to be considered a valid bijector
    return issubclass(cls, Bijector)


def standard_bijector(cls):
    # "Standard bijectors" are the ones we can perform standard automated tests upon
    return (
        inspect.isclass(cls)
        and isbijector(cls)
        and cls.__name__ not in [clx for clx, _ in meta_bijectors]
    )


# TODO: Autogenerate this from script!
standard_bijectors: List[Tuple[str, Type[Bijector]]] = [
    ("AffineAutoregressive", AffineAutoregressive),
    ("AffineFixed", AffineFixed),
    ("ELU", ELU),
    ("Exp", Exp),
    ("LeakyReLU", LeakyReLU),
    ("Permute", Permute),
    ("Power", Power),
    ("Sigmoid", Sigmoid),
    ("Softplus", Softplus),
    ("Tanh", Tanh),
]

__all__ = ["standard_bijectors", "meta_bijectors"] + [
    cls for cls, _ in meta_bijectors + standard_bijectors
]
