# Copyright (c) Meta Platforms, Inc

import importlib
import inspect
import pkgutil
from collections import OrderedDict

from types import ModuleType
from typing import Dict, Mapping, Sequence, Tuple, Callable, Optional, Any

from flowtorch.docs.docstring import Docstring
from flowtorch.docs.symbol import Symbol, SymbolType
from flowtorch.docs.object import modules, module_members, class_members
from flowtorch.docs.filters import regexs


def search_module(modname: str, filters: Mapping[str, Callable]) -> Sequence[Symbol]:
    symbols = {}

    # Enumerate and filter submodules
    mods = modules(modname)
    mods = [(name, obj) for name, obj in mods if filters['module'](name)]

    # Search items in submodules
    for module_name, module_obj in mods:
        symbols[module_name] = Symbol(None, module_name, module_obj)

        # Enumerate and filter objects
        members = module_members(module_obj)
        members = [(member_name, member_obj) for member_name, member_obj in members if filters['symbol'](module_name + '.' + member_name)]

        for member_name, member_obj in members:
            qualified_name = module_name + '.' + member_name
            this_symbol = Symbol(module_obj, qualified_name, member_obj)
            if this_symbol._type.name not in ["UNDEFINED", "BUILTIN"] and this_symbol._canonical_module.startswith(modname):
                symbols[qualified_name] = this_symbol

            # Get methods
            if this_symbol._type.name == "CLASS":
                methods = class_members(member_obj)
                methods = [(method_name, method_obj) for method_name, method_obj in methods if filters['symbol'](module_name + '.' + member_name + '.' + method_name)]

                for method_name, method_obj in methods:
                    qualified_name = module_name + '.' + member_name + '.' + method_name
                    this_symbol = Symbol(module_obj, qualified_name, method_obj)

                    if this_symbol._type.name not in ["UNDEFINED", "BUILTIN"] and this_symbol._canonical_module.startswith(modname):
                        symbols[qualified_name] = this_symbol

    return symbols


def generate_symbols(config: Any) -> Sequence[Symbol]:
    """
    Given a base module name, return a mapping from the name of all modules
    accessible under the base to a tuple of module and symbol objects.

    A symbol is represented by a tuple of the object name and value, and is
    either a function or a class accessible when the module is imported.

    """
    # Validate module name to document
    assert (
        "settings" in config
        and "search" in config["settings"]
        and (
            type(config["settings"]["search"]) is str
            or type(config["settings"]["search"]) is list
        )
    )

    # Construct module and object level filters
    filters = regexs(config)

     # Read in all modules and symbols
    search = config["settings"]["search"]
    search = set([search] if type(search) is str else search)
    symbols = {}
    for modname in search:
        symbols = {**symbols, **search_module(modname, filters)}

    return symbols


"""
def sparse_module_hierarchy(mod_names: Sequence[str]) -> Mapping[str, Any]:
    # Make list of modules to search and their hierarchy, pruning entries that
    # aren't in mod_names
    results: Dict[str, Any] = OrderedDict()
    this_dict = results

    for module in sorted(mod_names):
        submodules = module.split(".")

        # Navigate to the correct insertion place for this module
        for idx in range(0, len(submodules)):
            submodule = ".".join(submodules[0 : (idx + 1)])
            if submodule in this_dict:
                this_dict = this_dict[submodule]

        # Insert module if it doesn't exist already
        this_dict.setdefault(module, {})

    return results
"""
