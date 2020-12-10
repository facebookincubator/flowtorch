# Copyright (c) Simplex Development Team. All Rights Reserved
# SPDX-License-Identifier: MIT


class Params(object):
    """
    Deferred initialization of parameters.
    """
    def __init__(self, parameter_class, **kwargs):
        self.parameter_class = parameter_class
        self.kwargs = kwargs

    def __call__(self, input_shape, param_shapes):
        hypernet = self.parameter_class(input_shape, param_shapes, **self.kwargs)
        return hypernet
