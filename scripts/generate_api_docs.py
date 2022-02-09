# Copyright (c) Meta Platforms, Inc

"""
Generates MDX (Markdown + JSX, see https://mdxjs.com/) files and sidebar
information for the Docusaurus v2 website from the library components'
docstrings.

We have chosen to take this approach to integrate our API documentation
with Docusaurus because there is no pre-existing robust solution to use
Sphinx output with Docusaurus.

This script will be run by the "documentation" GitHub workflow on pushes
and pull requests to the main branch. It will function correctly from
any working directory.

"""

import errno
import importlib
import os
from typing import Mapping, Sequence, Tuple

import toml
from flowtorch.docs import generate_symbols, construct_article_list, module_sidebar
from flowtorch.docs.page import Page
from flowtorch.docs.symbol import Symbol


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
    symbols: Mapping[str, Symbol] = generate_symbols(config)
    canonical_symbols: Mapping[str, Sequence[Symbol]] = {}
    for _, s in symbols.items():
        canonical_symbols.setdefault(s._canonical_name, []).append(s)

    # Check for symbol duplicates (excepting inherited methods)
    duplicates: Sequence[Tuple[str, Sequence[str]]] = []
    for name, syms in canonical_symbols.items():
        if len(syms) > 1 and syms[0]._type.name != "METHOD":
            duplicates.append((name, [s._name for s in syms]))

    if len(duplicates):
        duplicates_str = "\n  ".join([f"{n}: {s}" for n, s in duplicates])
        error_msg = "Duplicate symbols found:\n  " + duplicates_str
        raise Exception(error_msg)

    # Build implicit hierarchy (mapping from symbol name to other names under it)
    hierarchy: Mapping[str, Sequence[str]] = {}
    for _, symbol in symbols.items():
        if symbol._type.name == "METHOD":
            class_name = ".".join(symbol._name.split(".")[:-1])
            hierarchy.setdefault(class_name, []).append(symbol._name)
        else:
            hierarchy.setdefault(symbol._module, []).append(symbol._name)

    # Build article list, i.e. mapping from article name to symbol object
    articles, symbol_to_article = construct_article_list(symbols)

    # At this point, (symbols, hierarchy, articles) defines everything we need
    # to construct the MDX files and navigation sidebar

    # Convert symbols to MDX and save
    # The magic happens inside Page object
    for page_name, symbol in articles.items():
        github = config["settings"]["github"]
        page = Page(page_name, symbol, symbols, hierarchy, symbol_to_article, github)

        with open(
            os.path.join(main_path, config["paths"]["markdown"], page_name + ".mdx"),
            "w",
        ) as file:
            print(page, file=file)

    # Save Doc v2 sidebar file
    with open(
        os.path.join(
            main_path,
            config["paths"]["sidebar"],
            config["paths"]["sidebar_filename"],
        ),
        "w",
    ) as file:
        sidebar_str = "\n".join(module_sidebar(symbols, hierarchy, symbol_to_article))
        print(sidebar_str, file=file)
