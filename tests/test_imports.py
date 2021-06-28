# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved
# SPDX-License-Identifier: MIT
import flowtorch
import flowtorch.params
import flowtorch.utils


def test_params_imports():
    params = [cls for cls, _ in flowtorch.utils.list_params()]
    unimported_params = set(params).difference(set(flowtorch.params.__all__))

    if len(unimported_params):
        raise ImportError(
            f'The following Params classes are declared but not imported: \
{",".join(unimported_params)}'
        )
