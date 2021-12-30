# Copyright (c) Meta Platforms, Inc

"""
Generates MDX (Markdown + JSX, see https://mdxjs.com/) files and sidebar
information for the Docusaurus v2 website from the library components'
docstrings.

We have chosen to take this approach to integrate our API documentation
with Docusaurus because there is no pre-existing robust solution to use
Sphinx output with Docusaurus.

This script will be run by the "documentation" GitHub workflow on pushes
and pull requests to the main branch. It will function corrrectly from
any working directory.

"""

import errno
import importlib
import inspect
import os
import re
from inspect import isclass, isfunction, ismodule
from typing import Any

import toml
from flowtorch.docs import (
    generate_class_markdown,
    generate_function_markdown,
    generate_module_markdown,
    sparse_module_hierarchy,
    walk_packages,
)


def module_sidebar(mod_name, items):
    return f"{{\n  type: 'category',\n  label: '{mod_name}',\n  \
collapsed: {'true'},\
  items: [{', '.join(items)}],\n}}"


def fullname(key, item):
    return key + "." + item


def dfs(dict):
    sidebar_items = []
    for key, val in dict.items():
        if len(modules_and_symbols[key][1]) > 0:
            items = [f'"api/{symbol_to_article[key]}"'] + [
                f'"api/{symbol_to_article[fullname(key, item)]}"'
                for item, _ in modules_and_symbols[key][1]
            ]
        else:
            items = []

        if val != {}:
            items.extend(dfs(val))

        sidebar_items.append(module_sidebar(key, items))

    return sidebar_items


# Generate article markdown files
def generate_markdown(article_name: str, symbol_name: str, entity: Any) -> str:
    """
    TODO: Method that inputs an object, extracts signature/docstring,
    and formats as markdown
    TODO: Method that build index markdown for overview files
    The overview for the entire API is a special case
    """

    if symbol_name == "":
        header = """---
id: overview
sidebar_label: "Overview"
slug: "/api"
---

:::info

These API stubs are generated from Python via a custom script and will filled
out in the future.

:::

"""
        return header

    # Regular modules/functions
    item = {
        "id": article_name,
        "sidebar_label": "Overview" if ismodule(entity) else symbol_name.split(".")[-1],
        "slug": f"/api/{article_name}",
        "ref": entity,
    }

    header = f"""---
id: {item['id']}
sidebar_label: {item['sidebar_label']}
---"""

    # Convert symbol to MDX

    # Imports for custom styling components
    markdown = [
        """import { FontAwesomeIcon } from '@fortawesome/react-fontawesome'
import { faAngleDoubleRight } from '@fortawesome/free-solid-svg-icons'
import PythonClass from "@theme/PythonClass";
import PythonFunction from "@theme/PythonFunction";
import PythonMethod from "@theme/PythonMethod";
import PythonModule from "@theme/PythonModule";
import PythonNavbar from "@theme/PythonNavbar";
"""
    ]

    # Make URL
    entity_file = (
        entity.__file__ if ismodule(entity) else inspect.getmodule(entity).__file__
    )
    url = (
        config["settings"]["github"]
        + "flowtorch/"
        + entity_file[(len(main_path) + 1) :].replace("\\", "/")
    )

    # Make navigation bar
    markdown.append(f"<PythonNavbar url='{url}'>\n")
    navigation = []
    symbol_splits = symbol_name.split(".")
    for idx in range(len(symbol_splits)):
        partial_symbol_name = ".".join(symbol_splits[0 : (idx + 1)])
        if idx == len(symbol_splits) - 1:
            navigation.append(f"*{symbol_splits[idx]}*")
        elif partial_symbol_name in symbol_to_article:
            navigation.append(
                f"[{symbol_splits[idx]}](/api/{symbol_to_article[partial_symbol_name]})"
            )
        else:
            navigation.append(f"{symbol_splits[idx]}")

    markdown.append(
        ' <FontAwesomeIcon icon={faAngleDoubleRight} size="sm" /> '.join(navigation)
    )
    markdown.append("\n</PythonNavbar>\n")

    # Handle known symbol types
    if isclass(entity):
        markdown.append(generate_class_markdown(symbol_name, entity))
        return "\n".join([header] + markdown)

    elif ismodule(entity):
        markdown.append(generate_module_markdown(symbol_name, entity))
        return "\n".join([header] + markdown)

    # Signature for function
    elif isfunction(entity):
        markdown.append(generate_function_markdown(symbol_name, entity))
        return "\n".join([header] + markdown)

    # Unknown symbol type
    else:
        raise ValueError(f"Symbol {symbol_name} has unknown type {type(symbol_object)}")


def search_symbols(config):
    # Validate module name to document
    assert (
        "settings" in config
        and "search" in config["settings"]
        and (
            type(config["settings"]["search"]) is str
            or type(config["settings"]["search"]) is list
        )
    )

    # TODO: Try to import module, more validation, etc.

    # Construct regular expressions for includes and excludes
    # Default include/exclude rules
    patterns = {
        "include": {"modules": re.compile(r".+"), "symbols": re.compile(r".+")},
        "exclude": {"modules": re.compile(r""), "symbols": re.compile(r"")},
    }

    # Override rules based on configuration file
    if "filters" in config:
        filters = config["filters"]
        for clude, rules in filters.items():
            for rule, pattern in rules.items():
                if type(pattern) is list:
                    pattern = "|".join(pattern)
                patterns[clude][rule] = re.compile(pattern)

    # Read in all modules and symbols
    search = config["settings"]["search"]
    search = [search] if type(search) is str else search
    modules_and_symbols = {}
    for modname in set(search):
        modules_and_symbols = {**modules_and_symbols, **walk_packages(modname)}

    # Apply filtering
    # TODO: Would be slightly faster if we applied module filtering inside walk_packages
    tmp = {}
    for x, y in modules_and_symbols.items():
        if (
            patterns["include"]["modules"].fullmatch(x) is not None
            and patterns["exclude"]["modules"].fullmatch(x) is None
        ):

            new_y1 = [
                (a, b)
                for a, b in y[1]
                if patterns["include"]["symbols"].fullmatch(x + "." + a) is not None
                and patterns["exclude"]["symbols"].fullmatch(x + "." + a) is None
            ]

            tmp[x] = (y[0], new_y1)

    return tmp


def construct_article_list(modules_and_symbols):
    # Construct list of articles (converting symbols to lower-case and collating)
    # NOTE: Webservers and Windows machines can't seem to distinguish addresses by
    # case...
    articles = {}
    symbol_to_article = {}

    for mod_name, (module, symbols) in modules_and_symbols.items():
        if len(symbols):
            article_name = mod_name.lower()
            # Find a unique name
            if article_name in articles:
                suffix = 1
                while article_name + str(suffix) in articles:
                    suffix += 1
                article_name = article_name + str(suffix)

            articles[article_name] = (mod_name, module)
            symbol_to_article[mod_name] = article_name

            suffix = 0
            for symbol_name, symbol in symbols:
                full_name = mod_name + "." + symbol_name
                article_name = full_name.lower()

                # Find a unique name
                if article_name in articles:
                    suffix += 1
                    while article_name + str(suffix) in articles:
                        suffix += 1
                    article_name = article_name + str(suffix)

                articles[article_name] = (full_name, symbol)
                symbol_to_article[full_name] = article_name

    return articles, symbol_to_article


if __name__ == "__main__":
    # Load and validate configuration file
    import flowtorch

    config_path = os.path.join(flowtorch.__path__[0], "../website/documentation.toml")
    config = toml.load(config_path)

    modules_and_symbols = search_symbols(config)
    articles, symbol_to_article = construct_article_list(modules_and_symbols)

    # Generate sidebar
    # Build hierarchy of modules
    hierarchy = sparse_module_hierarchy(modules_and_symbols.keys())

    # Create directories if they don't exist
    search = config["settings"]["search"]
    search = [search] if type(search) is str else search
    main_module = importlib.import_module(search[0])
    main_path = main_module.__path__[0]
    sidebar_path = os.path.join(main_path, config["paths"]["sidebar"])
    markdown_path = os.path.join(main_path, config["paths"]["markdown"])

    def create_paths(path: str) -> None:
        try:
            os.makedirs(path)
        except OSError as e:
            if e.errno != errno.EEXIST:
                raise

    create_paths(sidebar_path)
    create_paths(markdown_path)

    with open(
        os.path.join(
            os.path.join(
                main_path,
                config["paths"]["sidebar"],
                config["paths"]["sidebar_filename"],
            )
        ),
        "w",
    ) as file:
        print("module.exports = [\n'api/overview',", file=file)
        print(",".join(dfs(hierarchy)), file=file)
        print("];", file=file)

    # For each article, convert the symbols to markdown, etc.

    # TODO: How to handle when there is a symbol called overview?
    # Maybe add a key for None instead of "overview"?
    articles["overview"] = ("", None)

    for article_name, (symbol_name, symbol_object) in articles.items():
        with open(
            os.path.join(
                os.path.join(
                    main_path, config["paths"]["markdown"], article_name + ".mdx"
                )
            ),
            "w",
        ) as file:
            print(
                generate_markdown(article_name, symbol_name, symbol_object), file=file
            )
