# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved
# SPDX-License-Identifier: MIT
import flowtorch
import flowtorch.bijectors
import flowtorch.distributions
import flowtorch.parameters
import flowtorch.utils


# TODO: Refactoring of the following three test functions


def test_parameters_imports():
    params = [cls for cls, _ in flowtorch.utils.list_parameters()]
    unimported_params = set(params).difference(set(flowtorch.parameters.__all__))

    if len(unimported_params):
        raise ImportError(
            f'The following Parameters classes are declared but not imported: \
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


def test_distribution_imports():
    distributions = [cls for cls, _ in flowtorch.utils.list_distributions()]
    unimported_distributions = set(distributions).difference(
        set(flowtorch.distributions.__all__)
    )

    if len(unimported_distributions):
        raise ImportError(
            f'The following Distribution classes are declared but not imported: \
{",".join(unimported_distributions)}'
        )
