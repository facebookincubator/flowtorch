# Copyright (c) Meta Platforms, Inc
from enum import Enum, unique
import inspect
from typing import Any, Optional

@unique
class SymbolType(Enum):
    UNDEFINED = 0
    MODULE = 1
    CLASS = 2
    METHOD = 3
    ATTRIBUTE = 4
    FUNCTION = 5
    VARIABLE = 6
    BUILTIN = 7

class Symbol:
    """
    Represents a documentable entity.
    """
    def __init__(self, module: Optional[Any], name:str, object: Any) -> None:
        """
        Initialize a 

        Params:
            module: an object corresponding to where the symbol is bound.
            name: the name of the variable to which the symbol is bound.
            object: an object corresponding to the symbol.
        """
        assert inspect.ismodule(module) is True or module is None

        # Test for built-in
        if inspect.isbuiltin(object):
            self._type = SymbolType.BUILTIN
            return

        # Determine what type of symbol this is
        elif inspect.ismodule(object):
            self._type = SymbolType.MODULE
        elif inspect.isclass(object):
            self._type = SymbolType.CLASS
        elif inspect.ismethod(object) or inspect.isfunction(object):
            if object.__name__ != object.__qualname__:
                self._type = SymbolType.METHOD
            else:
                self._type = SymbolType.FUNCTION
        else:
            # TODO: How to detect a variable/attribute?
            self._type = SymbolType.UNDEFINED
            return

        # Remove indentation from docstring and store
        self._docstring = inspect.getdoc(object)

        # TODO: Check for decorators

        # Get source file locations
        if module is not None:
            self._file = inspect.getfile(module)
        else:
            self._file = inspect.getfile(object)
        self._canonical_file = inspect.getfile(object)

        # Get signature
        if self._type in [SymbolType.METHOD, SymbolType.FUNCTION]:
            self._signature = inspect.signature(object)

        # Get module/name and canonical modulename
        if self._type is SymbolType.MODULE:
            self._canonical_name = object.__name__
            #inspect.getmodulename(object.__file__)
            if module is not None:
                self._module = self._canonical_module = module.__name__
            else:
                self._module = self._canonical_module = ""

        elif self._type in [SymbolType.CLASS, SymbolType.FUNCTION, SymbolType.METHOD]:
            self._canonical_name = object.__qualname__
            self._module = module.__name__
            self._canonical_module = object.__module__

        self._name = name
        got_here = True


if __name__ == "__main__":
    # DEBUG
    #import flowtorch
    #import flowtorch.bijectors

    # Example module
    #s = Symbol(None, "flowtorch", flowtorch)

    # Example class
    #s = Symbol(flowtorch, "Lazy", flowtorch.Lazy)

    # Example method
    #s = Symbol(flowtorch, "Lazy.__init__", flowtorch.Lazy.__init__)

    # Example function
    #s = Symbol(flowtorch.bijectors, "isbijector", flowtorch.bijectors.isbijector)

    # TODO: Enumerate through all modules and symbols in those modules
    # TODO: At the same time, convert self._docstring's to self.Docstring's
    # and raise exception if error in processing...

    # *** CONTINUE FROM HERE TOMORROW ***
    # Copy code to walk packages

    # TODO: Build mapping from qualified symbol name to Symbol

    # TODO: Build hierarchy of modules and symbols in those modules

    # TODO: Check for duplicates according to canonical names

    # TODO: Convert Symbol's to Page's

    # TODO: Save Page's as MDX
