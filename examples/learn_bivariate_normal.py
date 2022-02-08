# Copyright (c) Meta Platforms, Inc

import os

import flowtorch.bijectors as bij
import flowtorch.distributions as dist
import flowtorch.parameters as params
import matplotlib.pyplot as plt
import torch

os.environ["KMP_DUPLICATE_LIB_OK"] = "True"

"""
This is a simple example to demonstrate training of normalizing flows.
Standard bivariate normal noise is sampled from the base distribution
and we learnt to transform it to a bivariate normal distribution with
independent but not identical components (see the produced figures).

"""


def learn_bivariate_normal() -> None:
    # Lazily instantiated flow plus base and target distributions
    bijectors = bij.AffineAutoregressive(
        params_fn=params.DenseAutoregressive(hidden_dims=(32,))
    )
    base_dist = torch.distributions.Independent(
        torch.distributions.Normal(torch.zeros(2), torch.ones(2)), 1
    )
    target_dist = torch.distributions.Independent(
        torch.distributions.Normal(torch.zeros(2) + 5, torch.ones(2) * 0.5), 1
    )

    # Instantiate transformed distribution and parameters
    flow = dist.Flow(base_dist, bijectors)

    # Fixed samples for plotting
    y_initial = flow.sample(
        torch.Size(
            [
                300,
            ]
        )
    )
    y_target = target_dist.sample(
        torch.Size(
            [
                300,
            ]
        )
    )

    # Training loop
    opt = torch.optim.Adam(flow.parameters(), lr=5e-3)
    frame = 0
    for idx in range(3001):
        opt.zero_grad()

        # Minimize KL(p || q)
        y = target_dist.sample((1000,))
        loss = -flow.log_prob(y).mean()

        if idx % 500 == 0:
            print("epoch", idx, "loss", loss)

            # Save SVG
            y_learnt = (
                flow.sample(
                    torch.Size(
                        [
                            300,
                        ]
                    )
                )
                .detach()
                .numpy()
            )

            plt.figure(figsize=(5, 5), dpi=100)
            plt.plot(
                y_target[:, 0],
                y_target[:, 1],
                "o",
                color="blue",
                alpha=0.95,
                label="target",
            )
            plt.plot(
                y_initial[:, 0],
                y_initial[:, 1],
                "o",
                color="grey",
                alpha=0.95,
                label="initial",
            )
            plt.plot(
                y_learnt[:, 0],
                y_learnt[:, 1],
                "o",
                color="red",
                alpha=0.95,
                label="learnt",
            )
            plt.xlim((-4, 8))
            plt.ylim((-4, 8))
            plt.xlabel("$x_1$")
            plt.ylabel("$x_2$")
            plt.legend(loc="lower right", facecolor=(1, 1, 1, 1.0))
            plt.savefig(
                f"bivariate-normal-frame-{frame}.svg",
                bbox_inches="tight",
                transparent=True,
            )

            frame += 1

        loss.backward()
        opt.step()


if __name__ == "__main__":
    learn_bivariate_normal()
