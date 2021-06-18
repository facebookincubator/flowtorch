---
id: ops
title: Continuous Integration
sidebar_label: Continuous Integration
---

FlowTorch uses [GitHub Actions](https://docs.github.com/en/actions) to run code quality tests on pushes or pull requests to the `master` branch, a process known as [continuous integration](https://en.wikipedia.org/wiki/Continuous_integration) (CI). The tests are run for Python versions 3.6, 3.7, and 3.8, and must be successful for a PR to be merged into `master`. All workflow runs can be viewed [here](https://github.com/stefanwebb/flowtorch/actions), or else viewed from the link at the bottom of the [PR](https://github.com/stefanwebb/flowtorch/pulls) in question.

:::info
Please do not feel intimidated by the thought of having to make your code pass the CI tests! The core developer team is happy to work closely with contributors to integrate their code and merge PRs.
:::

## Workflow Steps
The definition of the steps performed in the build workflow is found [here](https://github.com/stefanwebb/flowtorch/blob/master/.github/workflows/python-package.yml) and is as follows:

1. The version of Python (3.6, 3.7, or 3.8) is installed along with the developer dependencies of FlowTorch;
```bash
python -m pip install --upgrade pip
python -m pip install flake8 black isort pytest mypy
pip install numpy
pip install --pre torch torchvision torchaudio -f https://download.pytorch.org/whl/nightly/cpu/torch_nightly.html
pip install -e .[dev]
```
:::note
Currently, FlowTorch requires the nightly build of PyTorch since it depends on features of [`torch.distributions.transforms`](https://github.com/pytorch/pytorch/blob/master/torch/distributions/transforms.py) that have not yet been released in a stable version.
:::

2. The formatting of the Python code in the [library](https://github.com/stefanwebb/flowtorch/tree/master/flowtorch) and [tests](https://github.com/stefanwebb/flowtorch/tree/master/tests) is checked to ensure it follows a standard using [`black`](https://black.readthedocs.io/en/stable/);
```bash
black --check flowtorch tests
```
3. Likewise, the order and formatting of Python `import` statements in the same folders is checked to ensure it follows a standard using [`isort`](https://pycqa.github.io/isort/);
```bash
isort --check flowtorch tests
```
4. A [static code analysis](https://en.wikipedia.org/wiki/Static_program_analysis), or rather, linting, is performed by [`flake8`](https://flake8.pycqa.org/en/latest/) to find potential bugs;
```bash
flake8 flowtorch tests --count --show-source --statistics
```
5. FlowTorch makes use of type hints, which we consider mandatory for all contributed code, and static types are checked with [`mypy`](https://github.com/python/mypy);
```bash
mypy flowtorch
```
6. Unit tests

pytest + XML coverage report
```bash
pytest --cov=tests --cov-report=xml -W ignore::DeprecationWarning
```

7. The coverage report is uploaded to [Codecov](https://about.codecov.io/) with a [GitHub Action](https://github.com/codecov/codecov-action). This allows us to analyze the results and produce the percentage of code covered badge.

:::note
Uploading code coverage report to `codecov.io` and viewing results has not yet been tested!
:::

8. An API reference is generated automatically from the code's docstrings with [Sphinx](https://www.sphinx-doc.org/en/master/). 
```bash
cd docs
sphinx-apidoc -o source ../flowtorch/
make html
```
If any step fails, the workflow fails and it is inadvisable for the PR to be merged into `master`.

## Successful Commits
To ensure your PR passes, you should perform these steps *before pushing your local commits to the remote branch*.

### Run Tests
Run the tests first so that you can do the code formatting just once as the final step:
```bash
pytest tests -W ignore::DeprecationWarning
```
Fix any failing tests until the above command succeeds.

### Check Types
Check that there are no errors with the type hints:
```bash
mypy flowtorch
```
I find this is one of the most difficult steps to make pass - if you require assistance, comment on your PR, tagging the core developers.

### Build Docs
Check that the docs build successfully and fix any errors:
```bash
cd docs
sphinx-apidoc -o source ../flowtorch/
```
:::note
Currently, the Sphinx output is not integrated with the [Docusaurus v2 website](https://flowtorch.ai/api). In a future release, we would like to both automate this integration plus the deployment of the website when a PR is merged to `master`. In the meanwhile, core developers can run [this script](https://github.com/stefanwebb/flowtorch/blob/master/deploy-website-windows.bat) (on Windows) - more instructions [here](https://github.com/stefanwebb/flowtorch/tree/master/website).
:::

### Formatting and Linting
Having ensured the tests and docs are correct, run the following commands to standardize your code's formatting:
```bash
black flowtorch tests
isort flowtorch tests
```
Now, run these commands in check mode to ensure there are no errors:
```bash
black --check flowtorch tests
isort --check flowtorch tests
```
It is possible you may need to fix some errors by hand.

Finally, run the linter and fix any resulting errors:
```bash
flake8 flowtorch tests
```
At this point, you are ready to commit your changes and push to the remote branch - you're a star! :star: From there, your PR will be reviewed by the core developers and after any modifications are made, merged to the `master` branch.