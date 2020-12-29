# Copyright (c) FlowTorch Development Team. All Rights Reserved
# SPDX-License-Identifier: MIT

import torch

import flowtorch
import flowtorch.bijectors as bijectors
import flowtorch.params as params

import scipy
from scipy import stats

import matplotlib.pyplot as plt

# Settings
#torch.manual_seed(0)
batch_dim = 10000
input_dim = 2

# Create non-lazy parameters
base_dist = torch.distributions.Normal(torch.zeros(input_dim), torch.ones(input_dim))
bijection = bijectors.AffineAutoregressive()
lazy_params = params.DenseAutoregressive(hidden_dims=[4], permutation=torch.Tensor([0, 1]))
params = lazy_params(torch.Size([input_dim]), bijection.param_shapes(base_dist))

x = base_dist.rsample(torch.Size([100]))
mean, log_scale = [y.detach().numpy() for y in params(x)]

#print(mean.shape, log_scale.shape)
print(mean[:10,0])
print(mean[:10,1])

plt.plot(mean[:,0], mean[:,1], 'o', color='blue', alpha=0.7, label='mean')
#plt.plot(z_base[:,0], z_base[:,1], 'o', color='red', alpha=0.7, label='base')
plt.title('Samples from MADE')
plt.xlabel('$x_1$')
plt.ylabel('$x_2$')
plt.legend()
plt.show()

# p = flowtorch.params.DenseAutoregressive()
# print(type(p))

# for n, p in params.named_parameters():
#    print(n, p)
