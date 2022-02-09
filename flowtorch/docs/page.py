# Copyright (c) Meta Platforms, Inc
from typing import Mapping, Sequence

from flowtorch.docs.markdown import generate_class_markdown
from flowtorch.docs.symbol import Symbol

# TODO: Specify imports from config file
imports = """import { FontAwesomeIcon } from '@fortawesome/react-fontawesome'
import { faAngleDoubleRight } from '@fortawesome/free-solid-svg-icons'
import PythonClass from "@theme/PythonClass";
import PythonFunction from "@theme/PythonFunction";
import PythonMethod from "@theme/PythonMethod";
import PythonModule from "@theme/PythonModule";
import PythonNavbar from "@theme/PythonNavbar";
"""

index_header = """---
id: overview
sidebar_label: "Overview"
slug: "/api"
---

:::info

These API stubs are generated from Python via a custom script and will filled
out in the future.

:::

"""


class Page:
    """
    Represents a page of MDX markdown for a module, class, or function
    """

    def __init__(
        self,
        page_name: str,
        symbol: Symbol,
        symbols: Mapping[str, Symbol],
        hierarchy: Mapping[str, Sequence[str]],
        symbol_to_article: Mapping[str, str],
        github: str,
    ):
        # MDX header for Docusaurus v2
        if symbol._type.name == "MODULE":
            label = "Overview"
        else:
            label = symbol._name.split(".")[-1]
        header = f"""---
id: {page_name}
sidebar_label: {label}
---"""

        markdown = [header, imports]

        # Make URL
        # TODO: Make this generalizable across projects
        main_path = symbols["flowtorch"]._canonical_file[: -len("__init__.py")]
        symbol_path = symbol._canonical_file
        url = github + "flowtorch/" + symbol_path[len(main_path) :].replace("\\", "/")

        # Make navigation bar
        # TODO: Factor this out?
        markdown.append(f"<PythonNavbar url='{url}'>\n")
        navigation = []
        symbol_splits = symbol._name.split(".")
        for idx in range(len(symbol_splits)):
            partial_symbol_name = ".".join(symbol_splits[0 : (idx + 1)])
            if idx == len(symbol_splits) - 1:
                navigation.append(f"*{symbol_splits[idx]}*")
            elif partial_symbol_name in symbol_to_article:
                navigation.append(
                    f"""[{symbol_splits[idx]}](/api/\
{symbol_to_article[partial_symbol_name]})"""
                )
            else:
                navigation.append(f"{symbol_splits[idx]}")

        markdown.append(
            ' <FontAwesomeIcon icon={faAngleDoubleRight} size="sm" /> '.join(navigation)
        )
        markdown.append("\n</PythonNavbar>\n")

        if symbol._type.name == "CLASS":
            markdown.append(generate_class_markdown(symbol, symbols, hierarchy))

        elif symbol._type.name == "MODULE":
            markdown.append("(module)")

        elif symbol._type.name == "FUNCTION":
            markdown.append("(function)")

        else:
            raise Exception("Invalid symbol type for Page object")

        self._mdx = "\n".join(markdown)

    def __repr__(self) -> str:
        return self._mdx
