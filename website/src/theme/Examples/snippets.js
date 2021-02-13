const snippets = [
  {
    label: "Bivariate Normal",
    code: 
`import torch
import torch.distributions as dist
import flowtorch
import flowtorch.bijectors as bijectors

# Lazily instantiated flow plus base and target distributions
flow = bijectors.AffineAutoregressive()
base_dist = dist.Normal(torch.zeros(2), torch.ones(2))
target_dist = dist.Normal(torch.zeros(2)+5, torch.ones(2)*0.5)

# Instantiate transformed distribution and parameters
new_dist, params = flow(base_dist)

# Training loop
opt = torch.optim.Adam(params.parameters(), lr=5e-2)
for idx in range(501):
    opt.zero_grad()

    # Minimize KL(p || q)
    y = target_dist.sample((1000,))
    loss = -new_dist.log_prob(y).mean()

    if idx % 100 == 0:
        print('epoch', idx, 'loss', loss)
        
    loss.backward()
    opt.step()`,
  },
];

export default snippets;
