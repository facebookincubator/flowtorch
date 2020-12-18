[![](https://github.com/stefanwebb/flowtorch/workflows/Python%20package/badge.svg)](https://github.com/stefanwebb/flowtorch/actions?query=workflow%3A%22Python+package%22)

Copyright (c) FlowTorch Development Team.

This source code is licensed under the MIT license found in the
LICENSE.txt file in the root directory of this source tree.

# Overview

FlowTorch is a PyTorch library for the flexible representation of random distributions with Normalizing Flows.

# Installing FlowTorch

    git clone https://github.com/stefanwebb/flowtorch.git
    cd flowtorch
    pip install -e .

# Developing

To build documentation

    cd docs
    sphinx-apidoc -o source ../flowtorch/
    make html

To preview built HTML documentation

    cd docs/_build/html
    python -m http.server  # python 3 

See `.github/workflows/python-package.yml` for how we build, lint, and test.

# Developing

Install dev dependencies

    pip install -e .[dev]
    
Running tests

    pytest
