---
id: shapes
title: Shapes
sidebar_label: Shapes
---

One of the advantages of using FlowTorch is that we have carefully thought out how shape information is propagated from the base distribution through a sequence of bijective transforms. Before we explain how shapes are handled in FlowTorch, let us revisit the shape conventions shared across PyTorch and TensorFlow.

## Shape Conventions
FlowTorch shares the shape conventions of PyTorch's `torch.distributions.Distribution` and TensorFlow's `tfp.distributions.Distribution` for representing random distributions. In these conventions, the shape of a tensor sampled from a random distribution is divided into three parts: the *sample shape*, the *batch shape*, and the *event shape*.

As described in the [TensorFlow documentation](https://www.tensorflow.org/probability/examples/Understanding_TensorFlow_Distributions_Shapes#basics),
> * Event shape describes the shape of a single draw from the distribution; it may be dependent across dimensions. For scalar distributions, the event shape is []. For a 5-dimensional MultivariateNormal, the event shape is [5].
> * Batch shape describes independent, not identically distributed draws, aka a "batch" of distributions.
> * Sample shape describes independent, identically distributed draws of batches from the distribution family.

This is best illustrated with some simple examples.

## Two Principles of FlowTorch Shapes
### Unconditional transformed distribution have only a non-trivial event shape