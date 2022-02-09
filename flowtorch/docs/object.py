# Copyright (c) Meta Platforms, Inc
import importlib
import inspect
import pkgutil
from inspect import isclass, isfunction
from types import ModuleType
from typing import Sequence, Tuple, Callable, Any


def module_members(module: ModuleType) -> Sequence[Tuple[str, Any]]:
    """
    Given a module object, returns a list of object name and values for documentable
    objects (functions and classes defined in this module or a subclass).
    """
    return [
        (n, m)
        for n, m in inspect.getmembers(module, None)
        if isfunction(m) or (isclass(m) and m.__module__.startswith(module.__name__))
    ]


def class_members(cls: Any) -> Sequence[Tuple[str, Any]]:
    """
    Given a class object, returns a list of documentable objects (currently
    just methods).
    """
    assert isclass(cls)

    members = inspect.getmembers(cls, predicate=inspect.isroutine)
    members = [(n, obj) for n, obj in members if type(obj) not in ["method_descriptor"]]
    return members


def modules(modname: str) -> Sequence[Tuple[str, Any]]:
    """
    Given the name of a module, returns a list of submodules under this one.
    """
    module = importlib.import_module(modname)
    mods = [(modname, module)]

    # NOTE: I use path of flowtorch rather than e.g. flowtorch.bijectors
    # to avoid circular imports
    path = module.__path__  # type: ignore

    # The followings line uncovered a bug that hasn't been fixed in mypy:
    # https://github.com/python/mypy/issues/1422
    for importer, this_modname, _ in pkgutil.walk_packages(
        path=path,  # type: ignore  # mypy issue #1422
        prefix=f"{module.__name__}.",
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
        else:
            raise Exception("Finder is none")

        if module is not None:
            mods.append((this_modname, module))
            del finder
        else:
            raise Exception("Module is none")

    return mods


def decorators(function: Callable) -> Sequence[str]:
    """Returns list of decorators names

    Args:
        function (Callable): decorated method/function

    Return:
        List of decorators as strings

    Example:
        Given:

        @my_decorator
        @another_decorator
        def decorated_function():
            pass

        >>> get_decorators(decorated_function)
        ['@my_decorator', '@another_decorator']

    """
    source = inspect.getsource(function)
    index = source.find("def ")
    return [
        line.strip().split()[0]
        for line in source[:index].strip().splitlines()
        if line.strip()[0] == "@"
    ]
