Copyright (c) Simplex Development Team.

This source code is licensed under the MIT license found in the
LICENSE.txt file in the root directory of this source tree.

# Overview

Simplex is a library for deep probabilistic modelling in PyTorch.

# Installing Simplex

    git clone https://github.com/stefanwebb/simplex.git
    cd simplex
    pip install -e .

# Building Docs

To build documentation

    cd docs
    sphinx-apidoc -o source ../simplex/
    make html

To preview built HTML documentation

    cd docs/_build/html
    python -m http.server  # python 3 
    python -m SimpleHTTPServer  # python 2

# Developing

Install dev dependencies

    pip install -e .[dev]
    
Running tests

    pytest
