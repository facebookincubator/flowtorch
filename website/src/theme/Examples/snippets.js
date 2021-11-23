const snippets = [
  {
    label: "Bivariate Normal",
    code: 
`import torch
import flowtorch.bijectors as bij
import flowtorch.distributions as dist
import flowtorch.parameters as params

# Lazily instantiated flow plus base and target distributions
params = params.DenseAutoregressive(hidden_dims=(32,))
bijectors = bij.AffineAutoregressive(params=params)
base_dist = torch.distributions.Independent(
  torch.distributions.Normal(torch.zeros(2), torch.ones(2)), 
  1
)
target_dist = torch.distributions.Independent(
  torch.distributions.Normal(torch.zeros(2)+5, torch.ones(2)*0.5),
  1
)

# Instantiate transformed distribution and parameters
flow = dist.Flow(base_dist, bijectors)

# Training loop
opt = torch.optim.Adam(flow.parameters(), lr=5e-3)
frame = 0
for idx in range(3001):
    opt.zero_grad()

    # Minimize KL(p || q)
    y = target_dist.sample((1000,))
    loss = -flow.log_prob(y).mean()

    if idx % 500 == 0:
        print('epoch', idx, 'loss', loss)
        
    loss.backward()
    opt.step()`,
  },
];

export default snippets;
