# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved
# SPDX-License-Identifier: MIT
import flowtorch
import flowtorch.bijectors
import flowtorch.parameters
import flowtorch.utils


def test_params_imports():
    params = [cls for cls, _ in flowtorch.utils.list_params()]
    unimported_params = set(params).difference(set(flowtorch.parameters.__all__))

    if len(unimported_params):
        raise ImportError(
            f'The following Params classes are declared but not imported: \
{",".join(unimported_params)}'
        )


def test_bijector_imports():
    bijectors = [cls for cls, _ in flowtorch.utils.list_bijectors()]
    unimported_bijectors = set(bijectors).difference(set(flowtorch.bijectors.__all__))

    if len(unimported_bijectors):
        raise ImportError(
            f'The following Bijector classes are declared but not imported: \
{",".join(unimported_bijectors)}'
        )
