---
id: normalizing_flows
title: Normalizing Flows
sidebar_label: Normalizing Flows
---

:::caution

This document is under construction!

:::

Mathematically, we represent this as `y = f_\theta(x)` where `x` is a sample of standard Gaussian noise we want `y` to be close to a sample from a target distribution. The function `f`


The field of normalizing flows can be seen as a modern take on the [change of variables method for random distributions](https://en.wikipedia.org/wiki/Probability_density_function#Dependent_variables_and_change_of_variables), where the transformations are high-dimensional, involve neural networks, and are designed for stochastic optimization.
