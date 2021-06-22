# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved
# SPDX-License-Identifier: MIT

import inspect

import pytest

import flowtorch.bijectors as bijectors
from flowtorch.utils import InterfaceError


class TestSignatures:
    @pytest.mark.parametrize(
        "cls",
        [
            bij_name
            for _, bij_name in bijectors.standard_bijectors + bijectors.meta_bijectors
        ],
    )
    def test_bijectors(self, cls):
        # TODO: Add forward_shape and inverse_shape

        """required_methods = [
            "__init__",
            "_forward",
            "_log_abs_det_jacobian",
            "__call__",
            "param_shapes",
            "inv",
        ]
        optional_methods = ["_inverse"]"""

        self._test_methods(cls)
        self._test_type_hints(cls)
        self._test_default_args(cls)
        self._test_docstrings(cls)

    def _test_methods(self, cls):
        # TODO: Check that certain methods are defined and are not virtual!
        pass

    def _test_type_hints(self, cls):
        sig = inspect.signature(cls.__init__)

        no_hints = []
        for p in sig.parameters.values():
            if p.name != "self" and p.annotation is inspect._empty:
                no_hints.append(p.name)

        if len(no_hints) > 0:
            raise InterfaceError(
                f'Class {cls.__module__}.{cls.__qualname__} has method \
arguments that omit type hints\n\t__init__.{{{", ".join(no_hints)}}}'
            )

    def _test_default_args(self, cls):
        # TODO: Check that default arguments supplied for initializers
        pass

    def _test_docstrings(self, cls):
        # TODO: Validate that docstring is present and in Google format
        pass
