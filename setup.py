# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved
# SPDX-License-Identifier: MIT

import os
import sys

from setuptools import find_packages, setup

REQUIRED_MAJOR = 3
REQUIRED_MINOR = 6


TEST_REQUIRES = ["numpy", "pytest", "pytest-cov", "scipy"]
DEV_REQUIRES = TEST_REQUIRES + [
    "black",
    "flake8",
    "flake8-bugbear",
    "mypy",
    "usort",
]


# Check for python version
if sys.version_info < (REQUIRED_MAJOR, REQUIRED_MINOR):
    error = (
        "Your version of python ({major}.{minor}) is too old. You need "
        "python >= {required_major}.{required_minor}."
    ).format(
        major=sys.version_info.major,
        minor=sys.version_info.minor,
        required_minor=REQUIRED_MINOR,
        required_major=REQUIRED_MAJOR,
    )
    sys.exit(error)


# read in README.md as the long description
with open("README.md", "r") as fh:
    long_description = fh.read()

setup(
    name="flowtorch",
    description="Normalizing Flows for PyTorch",
    author="FlowTorch Development Team",
    author_email="info@stefanwebb.me",
    license="MIT",
    url="https://flowtorch.ai/users",
    project_urls={
        "Documentation": "https://flowtorch.ai/users",
        "Source": "https://www.github.com/facebookincubator/flowtorch",
    },
    keywords=[
        "Deep Learning",
        "Bayesian Inference",
        "Statistical Modeling",
        "Variational Inference",
        "PyTorch",
    ],
    classifiers=[
        "Development Status :: 3 - Alpha",
        "Programming Language :: Python :: 3 :: Only",
        "License :: OSI Approved :: MIT License",
        "Topic :: Scientific/Engineering",
        "Intended Audience :: Science/Research",
        "Intended Audience :: Developers",
    ],
    long_description=long_description,
    long_description_content_type="text/markdown",
    python_requires=">={}.{}".format(REQUIRED_MAJOR, REQUIRED_MINOR),
    install_requires=[
        "torch>=1.8.1",
    ],
    setup_requires=["setuptools_scm"],
    use_scm_version={
        "root": ".",
        "relative_to": __file__,
        "write_to": os.path.join("flowtorch", "version.py"),
    },
    packages=find_packages(
        include=["flowtorch", "flowtorch.*"],
        exclude=["debug", "tests", "website"],
    ),
    extras_require={
        "dev": DEV_REQUIRES,
        "test": TEST_REQUIRES,
    },
)
