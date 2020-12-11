# Copyright (c) Simplex Development Team. All Rights Reserved
# SPDX-License-Identifier: MIT

import functools
import simplex

def lazy_parameters(cls):
    @functools.wraps(cls)
    def wrapper_params(*args, **kwargs):
        return simplex.Params(cls, **kwargs)
    
    return wrapper_params