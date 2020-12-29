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
batch_dim = 100000
input_dim = 2

# Example of creating transformed distribution
flow = flowtorch.bijectors.AffineAutoregressive(flowtorch.params.DenseAutoregressive())
base_dist = torch.distributions.Normal(torch.zeros(input_dim), torch.ones(input_dim))

new_dist, params = flow(base_dist)
print(type(new_dist), type(params))

z = new_dist.rsample(torch.Size([batch_dim])).detach().numpy()
print(z.shape)

z_base = base_dist.rsample(torch.Size([batch_dim]))
z_base2 = base_dist.rsample(torch.Size([batch_dim]))

#print(new_dist.log_prob(base_dist.sample()))

#statistic, pvalue = stats.kstest(z[:,0], 'norm')
statistic, pvalue = stats.kstest(z_base2[:,0], 'norm')
print('statistic', statistic, 'p-value', pvalue)

"""
plt.plot(z[:,0], z[:,1], 'o', color='blue', alpha=0.7, label='transformed')
plt.plot(z_base[:,0], z_base[:,1], 'o', color='red', alpha=0.7, label='base')
plt.title('Samples from Transformed Distribution')
plt.xlabel('$x_1$')
plt.ylabel('$x_2$')
plt.legend()
plt.show()
"""



# p = flowtorch.params.DenseAutoregressive()
# print(type(p))

# for n, p in params.named_parameters():
#    print(n, p)
