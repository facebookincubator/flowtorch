import torch
import torch.distributions as dist
import flowtorch
import flowtorch.params
import flowtorch.bijectors
import matplotlib.pyplot as plt

# Lazily instantiated normalizing flow, base, and target distributions
nn = flowtorch.params.DenseAutoregressive(hidden_dims=[16,16])
flow = flowtorch.bijectors.AffineAutoregressive(nn)
base_dist = dist.Normal(torch.zeros(2), torch.ones(2))
target_dist = torch.distributions.Normal(torch.zeros(2)+5, torch.ones(2)*0.5)

# Instantiate transformed distribution and parameters
new_dist, params = flow(base_dist)

y_initial = new_dist.sample(torch.Size([300,])).detach().numpy()
y_target = target_dist.sample(torch.Size([300,])).detach().numpy()

# Training loop
opt = torch.optim.Adam(params.parameters(), lr=3e-2)
frame = 0
for idx in range(501):
    opt.zero_grad()

    # Minimize KL(p || q)
    y = target_dist.sample((1000,))
    loss = -new_dist.log_prob(y).mean()

    if idx % 100 == 0:
        print('epoch', idx, 'loss', loss)

        # Save SVG
        y_learnt = new_dist.sample(torch.Size([300,])).detach().numpy()

        plt.figure(figsize=(5,5), dpi= 100)
        plt.plot(y_target[:,0], y_target[:,1], 'o', color='blue', alpha=0.95, label='target')
        plt.plot(y_initial[:,0], y_initial[:,1], 'o', color='grey', alpha=0.95, label='initial')
        plt.plot(y_learnt[:,0], y_learnt[:,1], 'o', color='red', alpha=0.95, label='learnt')
        plt.xlim((-4,8))
        plt.ylim((-4,8))
        #plt.title("Learning Neal's Funnel")
        plt.xlabel('$x_1$')
        plt.ylabel('$x_2$')
        plt.legend(loc='lower right', facecolor=(1, 1, 1, 1.0))
        plt.savefig(f'bivariate-normal-frame-{frame}.svg', bbox_inches='tight', transparent=True)
        frame += 1



    loss.backward()
    opt.step()
 