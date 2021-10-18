---
id: docs
title: Docs
sidebar_label: Docs
---
:::info
The easiest way to write a docstring that adheres to FlowTorch conventions is to copy one from a pre-existing class and adapt it to your case.
:::

## Docstrings
It is crucial to add an informative [docstring](https://www.python.org/dev/peps/pep-0257/#id15) to new `bij.Bijector` and `params.Parameters` classes. This docstring should detail what the class does, its functional form, the meaning of *all* input arguments and returned values, and references to any relevant literature.

References should link to their citation in the [bibliography](/dev/bibliography), for example, with [https://flowtorch.ai/dev/bibliography#dinh2014nice](https://flowtorch.ai/dev/bibliography#dinh2014nice). This means you may need to add additional citations to the website with your `Bijector` or `Parameters` implementation.

Be sure to test the formatting of the docstring in the docs using the workflow detailed [here](/dev/ops).
