# Copyright (c) FlowTorch Development Team. All Rights Reserved
# SPDX-License-Identifier: MIT
import torch


class Params(object):
    """
    Deferred initialization of parameters.
    """

    def __init__(self):
        super(Params, self).__init__()
        # self._inv = None

    def __call__(self, input_shape, param_shapes):
        # TODO: Take this class out of Params!
        class ParamsModule(torch.nn.Module):
            def __init__(self, params, modules, buffers):
                super(ParamsModule, self).__init__()
                self.params = params
                self.mods = modules

                # DEBUG
                # for m in modules:
                #    print(m)

                if buffers is not None:
                    for n, v in buffers.items():
                        self.register_buffer(n, v)

            def forward(self, x):
                return self.params.forward(x, modules=self.mods)

        return ParamsModule(self, *self.build(input_shape, param_shapes))

    def forward(self, x, context=None, modules=None):
        return self._forward(x, context=context, modules=modules)

    def _forward(self, x, context=None, modules=None):
        """
        Abstract method to ***
        """
        raise NotImplementedError

    def build(self, input_shape, param_shapes):
        self.input_shape = input_shape
        self.param_shapes = param_shapes
        return self._build(input_shape, param_shapes)

    def _build(self, input_shape, param_shapes):
        """
        Abstract method to ***
        """
        raise NotImplementedError
