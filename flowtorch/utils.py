# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved
# SPDX-License-Identifier: MIT

import importlib
import inspect
import pkgutil
from typing import Sequence, Tuple

import flowtorch
import flowtorch.bijectors


def isbijector(cls):
    # A class must inherit from flowtorch.Bijector to be considered a valid bijector
    return inspect.isclass(cls) and issubclass(cls, flowtorch.Bijector)


def list_bijectors() -> Sequence[Tuple[str, flowtorch.Bijector]]:
    bijectors = []

    # The followings line uncovered a bug that hasn't been fixed in mypy:
    # https://github.com/python/mypy/issues/1422
    for importer, modname, _ in pkgutil.walk_packages(
        path=flowtorch.bijectors.__path__,  # type: ignore  # mypy issue #1422
        prefix=flowtorch.bijectors.__name__ + ".",
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
            this_bijectors = inspect.getmembers(module, isbijector)
            bijectors.extend(this_bijectors)

    return bijectors
