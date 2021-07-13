# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved
# SPDX-License-Identifier: MIT

import importlib
from collections import OrderedDict
from functools import lru_cache
from inspect import isclass, isfunction, ismodule
from typing import Any, Dict, Sequence

# We don't want to include, e.g. both flowtorch.bijectors.Affine and
# flowtorch.bijectors.affine.Affine. Hence, we specify a list of modules
# to explicitly include in the API docs (and don't recurse on them).
# TODO: Include flowtorch.ops and flowtorch.numerical

include_modules = [
    "flowtorch",
    "flowtorch.bijectors",
    "flowtorch.distributions",
    "flowtorch.experimental.params",
    "flowtorch.params",
    "flowtorch.utils",
]


def ispublic(name):
    return not name.startswith("_")


@lru_cache(maxsize=1)
def _documentable_modules() -> Dict[Any, Sequence[Any]]:
    """
    Returns a list of (module, [(name, entity), ...]) pairs for modules
    that are documentable
    """

    # TODO: Self document flowtorch.docs module
    results = {}

    def dfs(dict):
        for key, val in dict.items():
            module = importlib.import_module(key)
            entities = [
                (n, getattr(module, n))
                for n in sorted(
                    [
                        n
                        for n in dir(module)
                        if ispublic(n)
                        and (
                            isclass(getattr(module, n))
                            or isfunction(getattr(module, n))
                        )
                    ]
                )
            ]
            results[module] = entities

            dfs(val)

    # Depth first search over module hierarchy, loading modules and extracting entities
    dfs(_module_hierarchy())
    return results


@lru_cache(maxsize=1)
def _documentable_entities():
    """
    Returns a list of (str, entity) pairs for entities that are documentable
    """

    name_entity_mapping = {}
    documentable_modules = _documentable_modules()
    for module, entities in documentable_modules.items():
        if len(entities) > 0:
            name_entity_mapping[module.__name__] = module

        for name, entity in entities:
            qualified_name = f"{module.__name__}.{name}"
            name_entity_mapping[qualified_name] = entity

    sorted_entity_names = sorted(name_entity_mapping.keys())
    return sorted_entity_names, name_entity_mapping


@lru_cache(maxsize=1)
def _module_hierarchy():
    # Make list of modules to search and their hierarchy
    results = OrderedDict()
    for module in sorted(include_modules):
        submodules = module.split(".")
        this_dict = results.setdefault(submodules[0], {})

        for idx in range(1, len(submodules)):
            submodule = ".".join(submodules[0 : (idx + 1)])
            this_dict.setdefault(submodule, {})
            this_dict = this_dict[submodule]

    return results


def generate_markdown(name, entity):
    """
    TODO: Method that inputs an object, extracts signature/docstring,
    and formats as markdown
    TODO: Method that build index markdown for overview files
    The overview for the entire API is a special case
    """

    if name == "":
        header = """---
id: overview
sidebar_label: "Overview"
slug: "/api"
---"""
        filename = "../website/docs/api/overview.mdx"
        return filename, header

    # Regular modules/functions
    item = {
        "id": name,
        "sidebar_label": "Overview" if ismodule(entity) else name.split(".")[-1],
        "slug": f"/api/{name}",
        "ref": entity,
        "filename": f"../website/docs/api/{name}.mdx",
    }

    header = f"""---
id: {item['id']}
sidebar_label: {item['sidebar_label']}
slug: {item['slug']}
---"""

    markdown = header
    return item["filename"], markdown


module_hierarchy = _module_hierarchy()
documentable_modules = _documentable_modules()
sorted_entity_names, name_entity_mapping = _documentable_entities()

__all__ = [
    "documentable_modules",
    "generate_markdown",
    "module_hierarchy",
    "name_entity_mapping",
    "sorted_entity_names",
]
