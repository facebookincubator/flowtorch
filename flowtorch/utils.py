# Copyright (c) Meta Platforms, Inc

# pyre-unsafe

import importlib
import inspect
import os
import pkgutil
from collections.abc import Callable, Sequence
from functools import partial
from typing import Any

import flowtorch
from flowtorch.bijectors.base import Bijector
from flowtorch.parameters.base import Parameters
from torch.distributions import Distribution


copyright_header = """Copyright (c) Meta Platforms, Inc"""


def classname(cls: type) -> str:
    return ".".join([cls.__module__, cls.__name__])


def issubclass_byname(cls: type, test_cls: type) -> bool:
    """
    Test whether a class is a subclass of another by class names, in contrast
    to the built-in issubclass that does it by instance.
    """
    return classname(test_cls) in [classname(c) for c in cls.__mro__]


def isderivedclass(cls: type, base_cls: type) -> bool:
    # NOTE issubclass won't always do what we want here if base_cls is imported
    # inside the module of cls. I.e. issubclass returns False if cls inherits
    # from a base_cls with a different instance.
    return inspect.isclass(cls) and issubclass_byname(cls, base_cls)


def list_bijectors() -> Sequence[tuple[str, Bijector]]:
    ans = _walk_packages("bijectors", partial(isderivedclass, base_cls=Bijector))
    ans = [a for a in ans if ".ops." not in a[1].__module__]
    return list({classname(cls[1]): cls for cls in ans}.values())


def list_parameters() -> Sequence[tuple[str, Parameters]]:
    ans = _walk_packages("parameters", partial(isderivedclass, base_cls=Parameters))
    return list({classname(cls[1]): cls for cls in ans}.values())


def list_distributions() -> Sequence[tuple[str, Parameters]]:
    ans = _walk_packages(
        "distributions", partial(isderivedclass, base_cls=Distribution)
    )
    return list({classname(cls[1]): cls for cls in ans}.values())


def _walk_packages(
    modname: str, filter: Callable[[Any], bool] | None
) -> Sequence[tuple[str, Any]]:
    classes = []

    # NOTE: I use path of flowtorch rather than e.g. flowtorch.bijectors
    # to avoid circular imports
    path = [os.path.join(flowtorch.__path__[0], modname)]  # type: ignore

    # The followings line uncovered a bug that hasn't been fixed in mypy:
    # https://github.com/python/mypy/issues/1422
    for _, this_modname, _ in pkgutil.walk_packages(
        path=path,  # type: ignore  # mypy issue #1422
        prefix=f"{flowtorch.__name__}.{modname}.",
        onerror=lambda x: None,
    ):
        module = importlib.import_module(this_modname)

        if module is not None:
            this_classes = inspect.getmembers(module, filter)
            classes.extend(this_classes)

            del module

        else:
            raise Exception("Module is none")

    return classes


class InterfaceError(Exception):
    pass
