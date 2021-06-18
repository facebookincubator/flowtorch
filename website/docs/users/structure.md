---
id: structure
title: Structured Representations
sidebar_label: Structured Representations
---

## Bayesian Networks
The *structure* of a distribution refers to the set of independence relationships that hold for the distribution. Suppose we have a distribution over variables, $\{x_1,x_2,\ldots,x_N\}$. At one extreme, the variables are fully *independent* and the distribution can be written as,
$$
p(\mathbf{x}) = \prod^N_{n=1}p(x_n).
$$
At the other extreme, the variables are fully *dependent* and the distribution can be written as,
$$
p(\mathbf{x}) = \prod^N_{n=1}p(x_n\mid x_1,x_2\ldots,x_{n-1}).
$$
In between these two extremes, a distribution will have a factorization with factors that condition on some but not all of the previous variables under a given ordering.

The field of *Probabilistic Graphical Models* studies graphical representations that express these structural relationships within distributions, as well as inference algorithms that operate directly on the graphical structures. For instance, we say that the fully independent distribution factors over the following *directed acyclic graph* (DAG), also known as a *Bayesian network (BN) structure*,

[insert graph svg]

And the fully dependent distribution factors over the fully connected BN structure,

[insert graph svg]

Therefore, full independence corresponds to zero edges, full dependence corresponds to the maximum number of edges in a DAG (that is, $N\ \text{choose}\ 2$), and it seems reasonable that distributions between these two extremes factor according to some graph with an intermediate number of edges. For instance, a graph over $x_1,\ldots,x_7$ might factor according to this graph,

[insert graph svg]

For Bayesian networks, that is, *directed* graphical models (there are other formalisms for *undirected*, *bidirected*, and graphs with mixed directionality), the semantics of the graphical structure are that ... It can be shown that a graph factors according to a BN structure if and only if ...

So in a sense, the BN structure and a distribution's factorization are equivalent, and both express the conditional independence relationships that hold in the distribution.

## Faithfulness and Minimality
One important point to note is that the BN structure for which a distribution factors over may fail to express some of the independence relationships that hold in a distribution - it must not, however, express independence relationships that *do not* hold in the distribution. For instance, any distribution factors according to $\prod^N_{n=1}p(x_n\mid x_1,x_2\ldots,x_{n-1})$, by the chain rule of probability. So, the fully connected DAG is a valid BN structure for the fully *independent* distribution.

* Lack of edges express conditional independent relationships that hold in a distribution, whereas the presence of edges is non-informative.

* Definition of I-map, minimal I-Map, and faithful

* Non-uniqueness of the minimal I-map. Also can have varying number of edges!

## Structure of Normalizing Flows
The dependency structure of normalizing flows is not something that has been considered in the literature, save for a few papers (for example, ?). Typically, they input a fully independent .

However, it can be advantageous to represent some structure in the distribution and use this as an inductive prior for learning. [Cite my work!] showed ...

## Abstractions for Expressing Structure

:::info 

Keeping this discussion in mind, we are developing an abstraction for expressing structure in a normalizing flow for the `v0.2` release. This abstraction is likely to belong to both `Params` and `Bijector`s, and analogously to the `.forward_shape` and `.backward_shape` methods, informs the `TransformedDistribution` class how the dependency structure is effected by each layer of the normalizing flow.

There will likely be two methods exposed to the user on `TransformedDistribution`: `.factorization`, and `.topological_order`. The first, `.factorization` might return a dictionary from variable indices to the parents of that variable in a minimal I-map. Another possibility is for `.factorization` to input a variable indices and return the array of parent indices (in which case, perhaps it should be called `.parents` and perhaps there should be a `.children` too?). This may be better if calculating and returning the whole object is an expensive operation. The second, `.topological_order`, returns an array of indices in topological ordering, possibly only calculating this lazily the first time it is requested.

:::
