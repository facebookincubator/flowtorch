---
id: overview
title: Overview
sidebar_label: Overview
---

[FlowTorch](https://flowtorch.ai) is designed with easy extensibility in mind. In this section, we detail the interfaces for normalizing flow bijections and conditioning networks, as well as the software practices that must be followed. First, however, let us explain the process for making a contribution to [FlowTorch](https://flowtorch.ai).

## How to Make a Contribution
1.  New features begin with a discussion between users, independent contributors (that's you!), and the core development team. If you would like to see a new feature or are interested in contributing it yourself, please start a new thread on the forum, tagging it with "new feature."

2. After this discussion has taken place and the details of new feature has been decided upon, the next step is to fork the [flowtorch repo](https://github.com/stefanwebb/flowtorch) using the "Fork" button in the upper right corner.

3. Next, clone your forked repository locally and create a feature branch:

```bash
git clone https://github.com/<your username>/flowtorch.git
cd flowtorch
git checkout -b <your feature name>
```

4. Create your new feature using the instructions for [bijectors](/dev/bijector) and [conditioning networks](/dev/params).

5. Ensure you have [added a docstring](/dev/docs) to your new class and that it is [connected to existing unit tests](/dev/tests).

6. Follow the steps [here](/dev/ops#successful-commits) to ensure that your code is formatted correctly, passes type checks, unit tests, the docs build, and so on.

7. Assuming it passes these tests, commit the changes to your local repo and push to your remote fork.

8. Finally, create a [pull request](https://docs.github.com/en/github/collaborating-with-issues-and-pull-requests/about-pull-requests) (PR) to merge your forked feature branch into the [main master branch](https://github.com/stefanwebb/flowtorch). Give an informative name to your PR and include in the description of your PR the details of which features are added. Ensure your feature branch contains the latest commits from the master branch so as to avoid merge conflicts.

9. The core developers will review your PR and most likely suggest changes to the code. After edits have been made, pushing to the feature branch of your forked remote will update the existing PR that you have opened.

10. After all edits have been made and the tests pass, the core developers will merge your code into the master branch!

:::info
If you are having trouble getting the CI tests to pass, you may create a PR, regardless, in order to get a review and help from the core developers.
:::

:::info
It is preferable to write smaller, incremental PRs as opposed to larger, monolithic ones. Aim to modify only a few files and add less than 500 lines of code.
:::