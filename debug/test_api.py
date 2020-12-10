# Copyright (c) Simplex Development Team. All Rights Reserved
# SPDX-License-Identifier: MIT

import torch
import simplex
import simplex.bijectors as bijectors
import simplex.params as params

# Settings
torch.manual_seed(0)
batch_dim = 100
input_dim = 10

# Create stateless bijector and statefull hypernetwork
"""bijection = bijectors.AffineAutoregressive()
hypernet = params.DenseAutoregressive(input_dim=input_dim, context_dim=0, hidden_dims=[50])

# Try out low-level methods of bijector
x = torch.randn(input_dim)
y = bijection.forward(x, params=hypernet)
y_inv = bijection.inverse(y, params=hypernet)

print(bijection) # <= testing inheritance from simplex.Bijector
print('x', x)
print('y', y)
print('inv(y)', y_inv)"""

hypernet = params.DenseAutoregressive(torch.Size((input_dim,)), [torch.Size(()), torch.Size(())])
