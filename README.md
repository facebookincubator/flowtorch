[![](https://github.com/stefanwebb/simplex/workflows/Python%20package/badge.svg)](https://github.com/stefanwebb/simplex/actions?query=workflow%3A%22Python+package%22)

Copyright (c) Simplex Development Team.

This source code is licensed under the MIT license found in the
LICENSE.txt file in the root directory of this source tree.

# Overview

Simplex is a library for deep probabilistic modelling in PyTorch.

# Installing Simplex

    git clone https://github.com/stefanwebb/simplex.git
    cd simplex
    pip install -e .

# Developing

To build documentation

    cd docs
    sphinx-apidoc -o source ../simplex/
    make html

To preview built HTML documentation

    cd docs/_build/html
    python -m http.server  # python 3 
    python -m SimpleHTTPServer  # python 2

See `.github/workflows/python-package.yml` for how we build, lint, and test.

# Developing

Install dev dependencies

    pip install -e .[dev]
    
Running tests

    pytest
