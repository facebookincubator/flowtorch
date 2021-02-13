---
id: intro
title: Intro
sidebar_label: Intro
slug: /users
---

[FlowTorch](https://flowtorch.ai) is a PyTorch library for representation normalizing flows.

> Simply put, a normalizing flow is a learnable function that inputs samples from a simple random distribution, typically Gaussian noise, and outputs samples from a more complex target distribution.

For instance, [a normalizing flow can be trained](https://arxiv.org/abs/1906.04032) to transform high-dimensional standard Gaussian noise (illustrated conceptually in two dimensions in the table) into samples of a distribution based on a picture of Claude Shannon:

| Input Samples            | Output Samples            |  Training Data
:-------------------------:|:-------------------------:|:-------------------------:
<img src="/img/standard_normal_samples.png" alt="Sample from Bivariate Standard Normal" width="200rem"/>  |  <img src="/img/claude_shannon.png" alt="Sample from Neural Spline Flow" width="200rem"/> | <img src="/img/claude_shannon.png" alt="Sample from Neural Spline Flow" width="200rem"/>

We believe, although still a nascent field, that normalizing flows are a fundamental component of the modern statistics and probabilistic computing toolkit, and they have already found state-of-the-art applications in Bayesian inference, speech synthesis, and ???, to name a few. [FlowTorch](https://flowtorch.ai) is a library that provides PyTorch components for constructing such flows using the latest research in the field.

>Moreover, it defines a well-designed interface so that researchers can easily contribute their implementations.

For more theoretical background on normalizing flows and information about their applications, see the primer [here](users/univariate), which also links to recent survey papers.