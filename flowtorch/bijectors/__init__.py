# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved
# SPDX-License-Identifier: MIT

import importlib
import inspect
import pkgutil
import sys

import torch
import torch.distributions as dist

from flowtorch import Bijector
from flowtorch.bijectors.affine_autoregressive import AffineAutoregressive
from flowtorch.bijectors.affine_fixed import AffineFixed
from flowtorch.bijectors.compose import Compose
from flowtorch.bijectors.elu import ELU
from flowtorch.bijectors.exp import Exp
from flowtorch.bijectors.fixed import Fixed
from flowtorch.bijectors.volume_preserving import VolumePreserving

if __name__ == "__main__":
    raise RuntimeError("Cannot run flowtorch.bijectors as a script")

this_module = sys.modules[__name__]

# "Meta bijectors" are classes that descend from flowtorch.Bijector but either
# are not used directly by the user, or are used to operate on other bijectors.
# We have to write special units tests for these.
meta_bijectors = [
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


# Programatically import all standard bijectors
# This way, don't have to modify __init__.py when a new bijector is implemented!
# Is a list of (<class name>, <class ref>) tuples for all standard bijectors
standard_bijectors = []
# The following line uncovered a bug that hasn't been fixed in mypy:
# https://github.com/python/mypy/issues/1422
for importer, modname, _ in pkgutil.walk_packages(
    path=this_module.__path__,  # type: ignore  # mypy issue #1422
    prefix=this_module.__name__ + ".",
    onerror=lambda x: None,
):
    # Conditions required for mypy
    if importer is not None:
        if isinstance(importer, importlib.abc.MetaPathFinder):
            finder = importer.find_module(modname, None)
        elif isinstance(importer, importlib.abc.PathEntryFinder):
            finder = importer.find_module(modname)
    else:
        finder = None

    if finder is not None:
        module = finder.load_module(modname)
    if module is not None:
        this_bijectors = inspect.getmembers(module, standard_bijector)
        standard_bijectors.extend(this_bijectors)

for cls_name, cls in standard_bijectors:
    globals()[cls_name] = cls

# Determine invertible bijectors
invertible_bijectors = []
for bij_name, cls in standard_bijectors:
    # TODO: Use factored out version of the following
    # Define plan for flow
    bij = cls()
    event_dim = max(bij.domain.event_dim, 1)
    event_shape = event_dim * [4]
    base_dist = dist.Normal(torch.zeros(event_shape), torch.ones(event_shape))
    _, params = bij(base_dist)

    try:
        y = torch.randn(*bij.forward_shape(event_shape))
        bij.inverse(y, params)
    except NotImplementedError:
        pass
    else:
        invertible_bijectors.append((bij_name, cls))


__all__ = ["standard_bijectors", "meta_bijectors", "invertible_bijectors"] + [
    cls for cls, _ in meta_bijectors + standard_bijectors
]
