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
from inspect import ismodule, isclass, isfunction
from typing import Any, Mapping, Sequence

import toml
from flowtorch.docs import (
    #sparse_module_hierarchy,
    generate_symbols,
    #generate_class_markdown,
    #generate_module_markdown,
    #generate_function_markdown,
)
from flowtorch.docs.page import Page


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


# hierarchy: Mapping[str, Sequence[str]]
def construct_article_list(symbols):
    # Construct list of articles (converting symbols to lower-case and collating)
    # NOTE: Webservers and Windows machines can't seem to distinguish addresses by
    # case...
    articles = {}
    symbol_to_article = {}

    for name, symbol in symbols.items():
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


def create_paths(path: str) -> None:
        try:
            os.makedirs(path)
        except OSError as e:
            if e.errno != errno.EEXIST:
                raise


if __name__ == "__main__":
    # Load and validate configuration file
    import flowtorch
    config_path = os.path.join(flowtorch.__path__[0], "../website/documentation.toml")
    config = toml.load(config_path)

    # Create directories if they don't exist
    search = config["settings"]["search"]
    search = [search] if type(search) is str else search
    main_module = importlib.import_module(search[0])
    main_path = main_module.__path__[0]
    sidebar_path = os.path.join(main_path, config["paths"]["sidebar"])
    markdown_path = os.path.join(main_path, config["paths"]["markdown"])
    create_paths(sidebar_path)
    create_paths(markdown_path)

    # Produce mappings from symbol name and canonical name to Symbol object
    symbols = generate_symbols(config)
    canonical_symbols = {}
    for _, s in symbols.items():
        canonical_symbols.setdefault(s._canonical_name, []).append(s)
    
    # Check for symbol duplicates (excepting inherited methods)
    duplicates = []
    for name, syms in canonical_symbols.items():
        if len(syms) > 1 and syms[0]._type.name != 'METHOD':
            duplicates.append((name, [s._name for s in syms]))

    if len(duplicates):
        duplicates_str = '\n  '.join([f'{n}: {s}' for n,s in duplicates])
        error_msg = 'Duplicate symbols found:\n  ' + duplicates_str
        raise Exception(error_msg)

    # Build implicit hierarchy (mapping from symbol name to other names under it)
    hierarchy = {}
    for name, symbol in symbols.items():
        if symbol._type.name == 'METHOD':
            class_name = '.'.join(symbol._name.split('.')[:-1])
            hierarchy.setdefault(class_name, []).append(symbol._name)
        else:
            hierarchy.setdefault(symbol._module, []).append(symbol._name)

    # Build article list
    articles, symbol_to_article = construct_article_list(symbols)

    # DEBUG
    #for x, y in symbol_to_article.items():
    #    print(x, '->', y)

    # Convert symbols to MDX and save
    for page_name, symbol in articles.items():
        github = config["settings"]["github"]
        page = Page(page_name, symbol, symbols, hierarchy, symbol_to_article, github)

        with open(
                os.path.join(
                    main_path, config["paths"]["markdown"], page_name + ".mdx"
                )
            ,
            "w",
        ) as file:
            print(
                page, file=file
            )

    # At this point, (symbols, hierarchy, articles) defines everything we need
    # to construct the MDX files and navigation sidebar

    # Build 

    # DEBUG
    """
    for name, symbol in symbols.items():
        print('Symbol', name)
        print('  name:', symbol._name)
        print('  canonical name:', symbol._canonical_name)
        print('  type:', symbol._type)
        print('  module:', symbol._module)
        print('  canonical module:', symbol._canonical_module)
        print('  signature:', str(symbol._signature) if symbol._signature is not None else None)
        print('  bases:', symbol._bases)
        print('  file:', symbol._file)
        print('  canonical file:', symbol._canonical_file)
        print('')
    """

    """
    # Create .js sidebar
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
    """
