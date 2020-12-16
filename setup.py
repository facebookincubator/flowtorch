# Copyright (c) Simplex Development Team. All Rights Reserved
# SPDX-License-Identifier: MIT

import os
import re
import sys

from setuptools import find_packages, setup

REQUIRED_MAJOR = 3
REQUIRED_MINOR = 7


TEST_REQUIRES = ["pytest", "pytest-cov"]
DEV_REQUIRES = TEST_REQUIRES + [
    "black",
    "flake8",
    "isort",
    "recommonmark",
    "sphinx",
    "sphinx-autodoc-typehints",
    "sphinx_rtd_theme",
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


# get version string from module
current_dir = os.path.dirname(__file__)
init_file = os.path.join(current_dir, "simplex", "__init__.py")
version_regexp = r"__version__ = ['\"]([^'\"]*)['\"]"
with open(init_file, "r") as f:
    version = re.search(version_regexp, f.read(), re.M).group(1)

# read in README.md as the long description
with open("README.md", "r") as fh:
    long_description = fh.read()

setup(
    name="simplex",
    version=version,
    description="Deep probabilistic modelling with PyTorch",
    author="Simplex Development Team",
    license="MIT",
    url="https://www.github.com/stefanwebb/simplex",
    project_urls={
        "Documentation": "https://www.github.com/stefanwebb/simplex",
        "Source": "https://www.github.com/stefanwebb/simplex",
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
    python_requires=">=3.7",
    install_requires=[
        "torch>=1.6.0",
    ],
    packages=find_packages("simplex", "simplex.*"),
    # package_dir={"": "simplex"},
    extras_require={
        "dev": DEV_REQUIRES,
        "test": TEST_REQUIRES,
    },
)
