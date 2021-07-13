# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved
# SPDX-License-Identifier: MIT

import importlib
import inspect
import os
import pkgutil
from functools import partial
from typing import Sequence, Tuple

import flowtorch
from flowtorch.bijectors.base import Bijector
from flowtorch.params.base import Params


def isderivedclass(cls, base_cls):
    return inspect.isclass(cls) and issubclass(cls, base_cls)


def list_bijectors() -> Sequence[Tuple[str, Bijector]]:
    return _walk_packages("bijectors", partial(isderivedclass, base_cls=Bijector))


def list_params() -> Sequence[Tuple[str, Params]]:
    return _walk_packages("params", partial(isderivedclass, base_cls=Params))


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


class InterfaceError(Exception):
    pass
