# Copyright (c) FlowTorch Development Team. All Rights Reserved
# SPDX-License-Identifier: MIT

import torch
import torch.distributions as dist
from torch.distributions import constraints
from torch.distributions.utils import _standard_normal

import flowtorch
import flowtorch.bijectors as bijectors
import flowtorch.params as params

from affine_autoregressive import AffineAutoregressive as OldAffineAutoregressive
from dense_autoregressive import DenseAutoregressive as OldDenseAutoregressive

import scipy
from scipy import stats

import matplotlib.pyplot as plt

torch.manual_seed(1)

# Hyperparameters
input_dim = 2
count_mc_samples = 1000
#learning_rate = 5e-2
#learning_rate = 1e-2
#learning_rate = 5e-4
learning_rate = 1e-4
epochs = 10001

class NealsFunnel(dist.Distribution):
    """
    Neal's funnel.
    p(x,y) = N(y|0,3) N(x|0,exp(y/2))
    """

    support = constraints.real

    def __init__(self, validate_args=None):
        d = 2
        batch_shape, event_shape = torch.Size([]), (d,)
        super(NealsFunnel, self).__init__(
            batch_shape, event_shape, validate_args=validate_args
        )

    def rsample(self, sample_shape=None):
        if not sample_shape:
            sample_shape = torch.Size()
        eps = _standard_normal(
            (sample_shape[0], 2), dtype=torch.float, device=torch.device("cpu")
        )
        z = torch.zeros(eps.shape)
        z[..., 1] = torch.tensor(3.0) * eps[..., 1]
        z[..., 0] = torch.exp(z[..., 1] / 2.0) * eps[..., 0]
        #z[..., 0] = torch.tensor(3.0) * eps[..., 0]
        #z[..., 1] = torch.exp(z[..., 0] / 2.0) * eps[..., 1]
        return z

    def log_prob(self, value):
        if self._validate_args:
            self._validate_sample(value)
        x = value[..., 0]
        y = value[..., 1]

        log_prob = dist.Normal(0, 3).log_prob(y)
        log_prob += dist.Normal(0, torch.exp(y / 2)).log_prob(x)

        return log_prob

# Create Neal's Funnel in 2D
target_dist = NealsFunnel()
#target_dist = torch.distributions.Normal(torch.zeros(input_dim)+5, torch.ones(input_dim)*0.5)
y_target = target_dist.rsample((1000,)).detach().numpy()

# Create IAF with a single bijector
#flow = flowtorch.bijectors.AffineAutoregressive(flowtorch.params.DenseAutoregressive(hidden_dims=[16,16,16,16,16], permutation=torch.LongTensor([1,0])))
flow = flowtorch.bijectors.AffineAutoregressive(flowtorch.params.DenseAutoregressive(hidden_dims=[16,16], permutation=torch.LongTensor([1,0])))
#flow = flowtorch.bijectors.AffineAutoregressive(flowtorch.params.DenseAutoregressive(hidden_dims=[256,256], permutation=torch.LongTensor([1,0])))
#flow = flowtorch.bijectors.AffineAutoregressive(OldDenseAutoregressive(hidden_dims=[16,16], permutation=torch.LongTensor([0,1])))
#flow = OldAffineAutoregressive(OldDenseAutoregressive(hidden_dims=[16,16], permutation=torch.LongTensor([1,0])))
#flow = OldAffineAutoregressive(OldDenseAutoregressive(hidden_dims=[4,4,4,4]))
base_dist = torch.distributions.Normal(torch.zeros(input_dim), torch.ones(input_dim))
new_dist, params = flow(base_dist)
y_initial = new_dist.rsample(torch.Size([1000,])).detach().numpy()

# DEBUG: Print parameters
#for n, p in params.named_parameters():
#    print(n, p.shape)
#print(list(params.parameters())[-1])
#raise Exception()

# DEBUG: Checking permutation [1, 0]
#print('perm', params.permutation)
#raise Exception()

# Training loop
opt = torch.optim.Adam(params.parameters(), lr=learning_rate)
#y = target_dist.sample((count_mc_samples,))

for idx in range(epochs):
    #z0 = tdist.base_dist.rsample(sample_shape=(count_mc_samples,))
    # Minimize KL(p || q)
    #y = target_dist.sample((count_mc_samples,))
    #loss = -new_dist.log_prob(y).mean()

    # Minimize KL(q || p)
    #y2 = new_dist.rsample(torch.Size((count_mc_samples,)))
    #loss = -target_dist.log_prob(y2).mean()

    # Minimize something like Jenson-Shannon I think
    y = target_dist.sample((count_mc_samples,))
    y2 = new_dist.rsample(torch.Size((count_mc_samples,)))
    loss = -(new_dist.log_prob(y).mean() + target_dist.log_prob(y2).mean())/2.0

    if idx % 100 == 0:
        print('epoch', idx, 'loss', loss)
        #print('final.bias', list(params.parameters())[-1].detach().numpy())

    #zk = flow._forward(z0, params)
    #ldj = flow._log_abs_det_jacobian(z0, zk, params)

    #neg_elbo = -nf.log_prob(zk).sum()
    #neg_elbo += tdist.base_dist.log_prob(z0).sum() - ldj.sum()
    
    if not torch.isnan(loss):
        loss.backward()
        opt.step()
        opt.zero_grad()

    else:
        raise Exception("NaN in ELBo loss!")


"""
# Settings
#torch.manual_seed(0)
batch_dim = 1000
input_dim = 8

# Example of creating transformed distribution
flow = flowtorch.bijectors.AffineAutoregressive(flowtorch.params.DenseAutoregressive(hidden_dims=[8]))
base_dist = torch.distributions.Normal(torch.zeros(input_dim), torch.ones(input_dim))

new_dist, params = flow(base_dist)
print(type(new_dist), type(params))

z = new_dist.rsample(torch.Size([batch_dim])).detach().numpy()


#statistic, pvalue = stats.kstest(z[:,0], 'norm')
#print('statistic', statistic, 'p-value', pvalue)


plt.plot(z[:,6], z[:,7], 'o', color='blue', alpha=0.7, label='transformed')
plt.plot(z_base[:,6], z_base[:,7], 'o', color='red', alpha=0.7, label='base')
plt.title('Samples from Transformed Distribution')
plt.xlabel('$x_1$')
plt.ylabel('$x_2$')
plt.legend()
plt.show()
"""

y_learnt = new_dist.rsample(torch.Size([1000,])).detach().numpy()
plt.plot(y_target[:,0], y_target[:,1], 'o', color='blue', alpha=0.7, label='target')
plt.plot(y_initial[:,0], y_initial[:,1], 'o', color='red', alpha=0.7, label='initial')
plt.plot(y_learnt[:,0], y_learnt[:,1], 'o', color='green', alpha=0.7, label='learnt')
#plt.xlim((-4,4))
#plt.ylim((-4,4))
plt.title("Learning Neal's Funnel")
plt.xlabel('$x_1$')
plt.ylabel('$x_2$')
plt.legend()
plt.show()
