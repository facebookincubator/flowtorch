# Copyright (c) Meta Platforms, Inc

from typing import Any, Mapping, Sequence, Tuple, Callable

from flowtorch.docs.filters import regexs
from flowtorch.docs.object import modules, module_members, class_members
from flowtorch.docs.symbol import Symbol


def search_module(
    modname: str, filters: Mapping[str, Callable]
) -> Mapping[str, Symbol]:
    symbols = {}

    # Enumerate and filter submodules
    mods = modules(modname)
    mods = [(name, obj) for name, obj in mods if filters["module"](name)]

    # Search items in submodules
    for module_name, module_obj in mods:
        symbols[module_name] = Symbol(None, module_name, module_obj)

        # Enumerate and filter objects
        members = module_members(module_obj)
        members = [
            (member_name, member_obj)
            for member_name, member_obj in members
            if filters["symbol"](module_name + "." + member_name)
        ]

        for member_name, member_obj in members:
            qualified_name = module_name + "." + member_name
            this_symbol = Symbol(module_obj, qualified_name, member_obj)
            if (
                this_symbol._type.name
                not in [
                    "UNDEFINED",
                    "BUILTIN",
                ]
                and this_symbol._canonical_module.startswith(modname)
            ):
                symbols[qualified_name] = this_symbol

            # Get methods
            if this_symbol._type.name == "CLASS":
                methods = class_members(member_obj)
                methods = [
                    (method_name, method_obj)
                    for method_name, method_obj in methods
                    if filters["symbol"](
                        module_name + "." + member_name + "." + method_name
                    )
                ]

                for method_name, method_obj in methods:
                    qualified_name = module_name + "." + member_name + "." + method_name
                    this_symbol = Symbol(module_obj, qualified_name, method_obj)

                    if (
                        this_symbol._type.name
                        not in [
                            "UNDEFINED",
                            "BUILTIN",
                        ]
                        and this_symbol._canonical_module.startswith(modname)
                    ):
                        symbols[qualified_name] = this_symbol

    return symbols


def generate_symbols(config: Any) -> Mapping[str, Symbol]:
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


def construct_article_list(
    symbols: Mapping[str, Symbol]
) -> Tuple[Mapping[str, Symbol], Mapping[str, str]]:
    # Construct list of articles (converting symbols to lower-case and collating)
    # NOTE: Webservers and Windows machines can't seem to distinguish addresses by
    # case...
    articles: Mapping[str, Symbol] = {}
    symbol_to_article: Mapping[str, str] = {}

    for _, symbol in symbols.items():
        if symbol._type.name in ["MODULE", "CLASS", "FUNCTION"]:
            article_name = symbol._name.lower()
            # Find a unique name
            if article_name in articles:
                suffix = 1
                while article_name + str(suffix) in articles:
                    suffix += 1
                article_name = article_name + str(suffix)

            articles[article_name] = symbol
            symbol_to_article[symbol._name] = article_name

    return articles, symbol_to_article


def module_sidebar(
    symbols: Mapping[str, Symbol],
    hierarchy: Mapping[str, Sequence[str]],
    symbol_to_article: Mapping[str, str],
    name: str = "",
) -> Sequence[str]:
    # Create .js sidebar string for Docusaurus v2

    # Next elements in hierarchy
    # TODO: Sort so that goes classes, functions, modules
    if name in hierarchy:
        new = hierarchy[name]
        new = list(sorted(new))
    else:
        new = []

    # Base condition
    if name == "":
        items = []
        for item_name in new:
            items = items + module_sidebar(
                symbols, hierarchy, symbol_to_article, item_name
            )
        return ["module.exports = [\n'api/overview',", *items, "];"]
    else:
        if symbols[name]._type.name in ["CLASS", "FUNCTION"]:
            return [f'"api/{symbol_to_article[name]}", ']
        elif symbols[name]._type.name in ["MODULE"]:
            # TODO: Fill collapsed from a filter in config
            items = []
            for item_name in new:
                items = items + module_sidebar(
                    symbols, hierarchy, symbol_to_article, item_name
                )

            return [
                f"""{{
  type: 'category',
  label: '{name}',
  collapsed: true,
  items: ["api/{symbol_to_article[name]}",""",
                *items,
                "],\n},",
            ]
