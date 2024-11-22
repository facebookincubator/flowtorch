# Copyright (c) Meta Platforms, Inc

# pyre-unsafe

import inspect

import flowtorch
import flowtorch.bijectors
import flowtorch.distributions
import flowtorch.parameters
import flowtorch.utils


def test_parameters_imports():
    tst_imports(
        "Parameters",
        [cls for cls, _ in flowtorch.utils.list_parameters()],
        [
            c
            for c in flowtorch.parameters.__all__
            if inspect.isclass(flowtorch.parameters.__dict__[c])
        ],
    )


def test_bijector_imports():
    tst_imports(
        "Bijector",
        [cls for cls, _ in flowtorch.utils.list_bijectors()],
        [
            c
            for c in flowtorch.bijectors.__all__
            if inspect.isclass(flowtorch.bijectors.__dict__[c])
        ],
    )


def test_distribution_imports():
    tst_imports(
        "Distribution",
        [cls for cls, _ in flowtorch.utils.list_distributions()],
        [
            c
            for c in flowtorch.distributions.__all__
            if inspect.isclass(flowtorch.distributions.__dict__[c])
        ],
    )


def tst_imports(cls_name, detected, imported):
    unimported = set(detected).difference(set(imported))
    undetected = set(imported).difference(set(detected))

    error_msg = []
    if len(unimported):
        error_msg.append(
            f'The following {cls_name} classes are declared but not imported: \
{", ".join(unimported)}'
        )

    if len(undetected):
        error_msg.append(
            f'The following {cls_name} classes are imported but not detected: \
{", ".join(undetected)}'
        )

    if len(error_msg):
        raise ImportError("\n".join(error_msg))
