---
id: installation
title: Installation
sidebar_label: Installation
---

[FlowTorch](https://flowtorch.ai) can be installed as a package or directly from source.

## Requirements

Python 3.6 or later is required. Other requirements will be downloaded by `pip` according to [setup.py](https://github.com/stefanwebb/flowtorch/blob/master/setup.py).

## Pre-release

As [FlowTorch](https://flowtorch.ai) is currently under rapid development, we recommend installing the [latest commit](https://github.com/stefanwebb/flowtorch/commits/master) from GitHub:

    git clone https://github.com/stefanwebb/flowtorch.git
    cd flowtorch
    pip install -e .

Updates can then be performed by navigating to the directory where you cloned [FlowTorch](https://flowtorch.ai) and running:

    git pull

## Latest Release

Alternatively, the [latest release](https://github.com/stefanwebb/flowtorch/releases) is installed from [PyPI](https://pypi.org/project/flowtorch/):

    pip install flowtorch

## Developers

[Additional libraries](https://github.com/stefanwebb/flowtorch/blob/master/setup.py#L14) required for development are installed by replacing the above `pip` command with:

    pip install -e .[dev]