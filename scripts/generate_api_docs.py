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
from inspect import isclass, isfunction

# We don't want to include, e.g. both flowtorch.bijectors.Affine and
# flowtorch.bijectors.affine.Affine. Hence, we specify a list of modules
# to explicitly include in the API docs (and don't recurse on them).
# TODO: Include flowtorch.ops and flowtorch.numerical
include_modules = [
    'flowtorch',
    'flowtorch.bijectors',
    'flowtorch.distributions',
    'flowtorch.experimental.params',
    'flowtorch.params',
    'flowtorch.utils']

def ispublic(name):
    return not name.startswith('_')

if __name__ == "__main__":
    # Make list of modules to search and their hierarchy
    module_hierarchy = {}
    for module in include_modules:
        #print(module)

        submodules = module.split('.')
        this_dict = module_hierarchy.setdefault(submodules[0], {})

        for idx in range(1, len(submodules)):
            submodule = '.'.join(submodules[0:(idx+1)])
            this_dict.setdefault(submodule, {})
            this_dict = this_dict[submodule]
            
    print(module_hierarchy)

    # Make a list of classes and functions to document from each module
    modules = {}
    def dfs(dict):
        for key, val in dict.items():
            modules[key] = importlib.import_module(key)
            dfs(val)

    dfs(module_hierarchy)

    for m in modules.values():
        print(m.__name__, [n for n in dir(m) if ispublic(n) and (isclass(getattr(m, n)) or isfunction(getattr(m, n)))])

    # TODO: Build sidebar JSON based on module hierarchy and save to /website

    # TODO: Extract signatures and save API file stubs

    # TODO: Format docstring and save as MDX

    pass
