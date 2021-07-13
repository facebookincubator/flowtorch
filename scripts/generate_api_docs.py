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

import os

import flowtorch
from flowtorch.docs import (
    documentable_modules,
    generate_markdown,
    module_hierarchy,
    name_entity_mapping,
)

if __name__ == "__main__":
    # Build sidebar JSON based on module hierarchy and save to 'website/api.sidebar.js'
    all_sidebar_items = []

    documentable_module_names = {m.__name__: v for m, v in documentable_modules.items()}

    def module_sidebar(mod_name, items):
        return f"{{\n  type: 'category',\n  label: '{mod_name}',\n  \
collapsed: {'false' if mod_name in module_hierarchy.keys() else 'true'},\
  items: [{', '.join(items)}],\n}}"

    def dfs(dict):
        sidebar_items = []
        for key, val in dict.items():
            items = (
                [f'"api/{key}"']
                + [f'"api/{key}.{item[0]}"' for item in documentable_module_names[key]]
                if len(documentable_module_names[key]) > 0
                else []
            )

            if val != {}:
                items.extend(dfs(val))

            sidebar_items.append(module_sidebar(key, items))

        return sidebar_items

    # Convert class hierarchy into API sidebar
    with open(
        os.path.join(flowtorch.__path__[0], "../website/api.sidebar.js"), "w"
    ) as file:
        print("module.exports = [\n'api/overview',", file=file)
        print(",".join(dfs(module_hierarchy)), file=file)
        print("];", file=file)

    # Generate markdown files for documentable entities
    name_entity_mapping = name_entity_mapping.copy()
    name_entity_mapping[""] = None
    for name, entity in name_entity_mapping.items():
        filename, markdown = generate_markdown(name, entity)

        with open(os.path.join(flowtorch.__path__[0], filename), "w") as file:
            print(markdown, file=file)
