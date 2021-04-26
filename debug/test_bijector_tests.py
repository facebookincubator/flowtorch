import torch
import torch.distributions as dist
import flowtorch
import flowtorch.bijectors as bijectors

import inspect

#bees = inspect.getmembers(bijectors, inspect.isclass)
#print(bees)

print(bijectors.__all__)

# Lazily instantiated flow plus base and target distributions
flow = bijectors.AffineAutoregressive()
input_dim = min(flow.domain.event_dim, 1) * [5]
base_dist = dist.Normal(torch.zeros(input_dim), torch.ones(input_dim))

# Instantiate transformed distribution and parameters
new_dist, params = flow(base_dist)

#flow._inverse(torch.zeros(input_dim), params, None)

y = new_dist.sample()
print(y)