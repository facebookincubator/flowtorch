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

# Example of lazily instantiating hypernetwork
# TODO: Remove layer of indirection from the following (possibly with class decorator)!
#p = simplex.Params(simplex.params.DenseAutoregressive)
#hypernet = p(torch.Size((input_dim,)), [torch.Size(()), torch.Size(())])

#base_dist = torch.distributions.Normal(torch.zeros(input_dim), torch.ones(input_dim))
#x = base_dist.sample()

#print(hypernet(x))

# Example of creating transformed distribution
flow = simplex.bijectors.AffineAutoregressive(simplex.Params(simplex.params.DenseAutoregressive))
base_dist = torch.distributions.Normal(torch.zeros(input_dim), torch.ones(input_dim))

new_dist, params = flow(base_dist)
print(type(new_dist), type(params))

print(new_dist.rsample())
print(new_dist.log_prob(base_dist.sample()))

for n, p in params.named_parameters():
    print(n, p)
