---
id: releases
title: Releases
sidebar_label: Releases
---

Core developers should follow this procedure to release a new version of FlowTorch.

:::note
SW: I am unfamiliar with `setuptools_scm` and am unsure whether we should define a tag every release or use the auto-bump. This doc should be updated when we actually go through the next release process.
:::

1. Add a version tag to release commit, or optionally, allow [`setuptools_scm`](https://github.com/pypa/setuptools_scm) to bump the version automatically according to [these rules](https://github.com/pypa/setuptools_scm/#default-versioning-scheme), and push:
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
    *  [PyTorch Slack](pytorch.slack.com) channels `#normalizing_flows`, `#pyro`, and `#announcements`
    * Personal account on Facebook
    * Personal account on Twitter
    * Personal account on LinkedIn

    including link to release notes and [main website](https://flowtorch.ai).