# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved
# SPDX-License-Identifier: MIT

import importlib
import inspect
import os
import pkgutil
from functools import partial
from typing import Sequence, Tuple

import flowtorch
import torch


def isderivedclass(cls, base_cls):
    return inspect.isclass(cls) and issubclass(cls, base_cls)


def list_bijectors() -> Sequence[Tuple[str, flowtorch.Bijector]]:
    return _walk_packages(
        "bijectors", partial(isderivedclass, base_cls=flowtorch.Bijector)
    )


def list_params() -> Sequence[Tuple[str, flowtorch.Params]]:
    return _walk_packages("params", partial(isderivedclass, base_cls=flowtorch.Params))


def _walk_packages(modname, filter):
    classes = []

    # NOTE: I use path of flowtorch rather than e.g. flowtorch.bijectors
    # to avoid circular imports
    path = [os.path.join(flowtorch.__path__[0], modname)]

    # The followings line uncovered a bug that hasn't been fixed in mypy:
    # https://github.com/python/mypy/issues/1422
    for importer, this_modname, _ in pkgutil.walk_packages(
        path=path,  # type: ignore  # mypy issue #1422
        prefix=f"{flowtorch.__name__}.{modname}.",
        onerror=lambda x: None,
    ):
        # Conditions required for mypy
        if importer is not None:
            if isinstance(importer, importlib.abc.MetaPathFinder):
                finder = importer.find_module(this_modname, None)
            elif isinstance(importer, importlib.abc.PathEntryFinder):
                finder = importer.find_module(this_modname)
        else:
            finder = None

        if finder is not None:
            module = finder.load_module(this_modname)
        if module is not None:
            this_classes = inspect.getmembers(module, filter)
            classes.extend(this_classes)

    return classes


eps = 1e-8


class InterfaceError(Exception):
    pass


def clamp_preserve_gradients(x: torch.Tensor, min: float, max: float) -> torch.Tensor:
    # This helper function clamps gradients but still passes through the
    # gradient in clamped regions
    return x + (x.clamp(min, max) - x).detach()


def clipped_sigmoid(x: torch.Tensor) -> torch.Tensor:
    finfo = torch.finfo(x.dtype)
    return torch.clamp(torch.sigmoid(x), min=finfo.tiny, max=1.0 - finfo.eps)
