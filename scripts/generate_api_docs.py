# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved
# SPDX-License-Identifier: MIT

"""
Generates MDX (Markdown + JSX, see https://mdxjs.com/) files and sidebar
information for the Docusaurus v2 website from the library components'
docstrings.

We have chosen to take this approach to integrate our API documentation
with Docusaurus because there is no pre-existing robust solution to use
Sphinx output with Docusaurus.

"""

import importlib
import os
from inspect import isclass, isfunction

import flowtorch

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


if __name__ == "__main__":
    # Make list of modules to search and their hierarchy
    # TODO: Ordered dictionary in alphabetical order
    module_hierarchy = {}
    for module in include_modules:
        submodules = module.split(".")
        this_dict = module_hierarchy.setdefault(submodules[0], {})

        for idx in range(1, len(submodules)):
            submodule = ".".join(submodules[0 : (idx + 1)])
            this_dict.setdefault(submodule, {})
            this_dict = this_dict[submodule]

    # Make a list of classes and functions to document from each module
    modules = {}

    def dfs(dict):
        for key, val in dict.items():
            modules[key] = importlib.import_module(key)
            dfs(val)

    dfs(module_hierarchy)

    # Build mapping from module to list of APIable entities inside
    module_items = {
        m.__name__: list(
            sorted(
                [
                    n
                    for n in dir(m)
                    if ispublic(n)
                    and (isclass(getattr(m, n)) or isfunction(getattr(m, n)))
                ]
            )
        )
        for m in modules.values()
    }

    # Build list of APIable entities relevant info to build markdown
    # Add overview for entire API
    item = {
        "id": "overview",
        "sidebar_label": "Overview",
        "slug": "/api",
        "ref": None,
        "filename": "../website/docs/api/overview.mdx",
    }
    api_items = [item]

    for m in modules.values():
        # Add main page for a module
        item = {
            "id": m.__name__,
            "sidebar_label": "Overview",
            "slug": f"/api/{m.__name__}",
            "ref": m,
            "filename": f"../website/docs/api/{m.__name__}.mdx",
        }
        api_items.append(item)

        # Add pages for items inside module
        api_items.extend(
            [
                {
                    "id": f"{m.__name__}.{n}",
                    "sidebar_label": n,
                    "slug": f"/api/{m.__name__}.{n}",
                    "ref": getattr(m, n),
                    "filename": f"../website/docs/api/{m.__name__}.{n}.mdx",
                }
                for n in dir(m)
                if ispublic(n) and (isclass(getattr(m, n)) or isfunction(getattr(m, n)))
            ]
        )

    # Build sidebar JSON based on module hierarchy and save to 'website/api.sidebar.js'
    all_sidebar_items = ["api/overview"]

    def module_sidebar(mod_name, items):
        return f"{{\n  type: 'category',\n  label: '{mod_name}',\n  \
collapsed: {'false' if mod_name in module_hierarchy.keys() else 'true'},\
  items: [{', '.join(items)}],\n}}"

    def dfs2(dict):
        sidebar_items = []
        for key, val in dict.items():
            items = [f'"api/{key}"'] + [
                f'"api/{key}.{item}"' for item in module_items[key]
            ]

            if val != {}:
                items.extend(dfs2(val))

            sidebar_items.append(module_sidebar(key, items))

        return sidebar_items

    # Convert class hierarchy into API sidebar
    with open(
        os.path.join(flowtorch.__path__[0], "../website/api.sidebar.js"), "w"
    ) as file:
        print("module.exports = [\n'api/overview',", file=file)
        print(",".join(dfs2(module_hierarchy)), file=file)
        print("];", file=file)

    # TODO: Unit test for API items that are indistinguishable by case

    # Save stubs
    # TODO: Method that inputs an object, extracts signature/docstring,
    # and formats as markdown
    # TODO: Method that build index markdown for overview files
    for item in api_items:
        header = f"""---
id: {item['id']}
sidebar_label: {item['sidebar_label']}
slug: {item['slug']}
---"""
        with open(os.path.join(flowtorch.__path__[0], item["filename"]), "w") as file:
            print(header, file=file)
