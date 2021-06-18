---
id: releases
title: Releases
sidebar_label: Releases
---

A list of FlowTorch releases is to be found [here](https://github.com/stefanwebb/flowtorch/releases). In this section, we detail the process of making a release.

## Versioning Scheme
First, let us describe the versioning scheme we use. It is a simple system with versions of the form *&#60;major&#62;.&#60;minor&#62;[.dev&#60;build&#62;]*. Some examples are:
* `0.5`;
* `1.4`; and,
* `0.0.dev1`.

We use [`setuptools_scm`](https://github.com/pypa/setuptools_scm) to automatically handle versions, and it is able to bump the version for builds without `.dev`. A description of how [`setuptools_scm`](https://github.com/pypa/setuptools_scm) handles versioning can be found [here](https://github.com/pypa/setuptools_scm/#default-versioning-scheme).

## Making a Release
Core developers should follow this procedure to release a new version of FlowTorch.

1. Add a version tag to release commit, or optionally, allow [`setuptools_scm`](https://github.com/pypa/setuptools_scm) to bump the version automatically, and push:
```bash
git tag <version>
git push
```
2. Make release notes and add to [GitHub releases](https://github.com/stefanwebb/flowtorch/releases).
3. Build the wheel and test that the package description will render correctly on PyPI:
```bash
python setup.py sdist bdist_wheel
twine check dist/*
```
4. Upload the package to PyPI:
```bash
twine upload dist/*
```
5. Make release announcements on:
    *  [PyTorch Slack](https://pytorch.slack.com) channels `#normalizing_flows`, `#pyro`, and `#announcements`
    * Personal account on Facebook
    * Personal account on Twitter
    * Personal account on LinkedIn

    including link to release notes and [main website](https://flowtorch.ai).