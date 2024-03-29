---
id: releases
title: Releases
sidebar_label: Releases
---

A list of FlowTorch releases is to be found [here](https://github.com/facebookincubator/flowtorch/releases). In this section, we detail the process of making a release.

## Versioning Scheme
The versioning scheme we use is a simple system with versions of the form *&#60;major&#62;.&#60;minor&#62;[.dev&#60;build&#62;]*. Some examples are:
* `0.5`;
* `1.4`; and,
* `0.0.dev1`.

We use [`setuptools_scm`](https://github.com/pypa/setuptools_scm) to automatically handle versions, and it is able to bump the version for builds without `.dev`. A description of how [`setuptools_scm`](https://github.com/pypa/setuptools_scm) handles versioning can be found [here](https://github.com/pypa/setuptools_scm/#default-versioning-scheme).
