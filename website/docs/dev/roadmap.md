---
id: roadmap
title: Roadmap
sidebar_label: Roadmap
---
This page lists the scope of upcoming releases. A more detailed list of planned features and improvements is to be found [here](https://github.com/stefanwebb/flowtorch/projects). We are aiming for a **May 31, 2021** release of `v0.1`! 

## `v0.1`
### Library
* Representative `Bijector`s: `Affine`, `AffineAutoregressive`, `Exp`, `Radial`, `Sigmoid`
* Basic `Params`: `Constant`, `Dense`, `DenseAutoregressive`, `None`
* Composing `Bijector`s: `Cat`, `Compose`, `Reshape`, `Stack`
* Conditional distributions
* Migrating unit tests from Pyro

### Website
* First version of all content and placeholders for `v0.2` features

## `v0.2`
### Docs
* Complete docstrings for all user-facing classes and methods
* Integrate Sphinx output with Docusaurus v2

### Library
* State-of-the-art `Bijector`s: `SplineAutoregressive`
* Caching with cache-flush on gradient update
* Structured representations
* Testing of GPU, serialization, and TorchScript support with unit tests
* Submission of [PyTorch Ecosystem](https://pytorch.org/ecosystem/) [application](https://pytorch.org/ecosystem/join).

### Website
* Complete content for `v0.2` features

## `v0.3`
### Library
* Migrate remaining bijections from Pyro

### Website
* Detailed tutorials on how to implement new `Bijector` and `Params` classes
* Tool to autogenerate `.bib` and `.mdx` bibliography files from `xml`.
* Finish literature search and bibliography
* Make website mobile friendly
* Add [Google Analytics](https://analytics.google.com/)
* Automate updating website with [GitHub Actions](https://github.com/features/actions)
* Create and link to [Colab](https://colab.research.google.com/) for all examples
* Maintain mailing list and list of interested GitHub users for releases 

:::info
After the `v0.3`, I would like to initiate a wider review of the FlowTorch API and a discussion of where the most productive place to focus our efforts is (for example, writing application-base tutorials, implementing training benchmarks, doing normalizing flow research, and so on).
:::