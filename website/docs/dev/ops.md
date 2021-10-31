---
id: ops
title: Continuous Integration
sidebar_label: Continuous Integration
---
:::info
Please do not feel intimidated by the thought of having to make your code pass the CI tests! The core developer team is happy to work closely with contributors to integrate their code and merge PRs.
:::

FlowTorch uses [GitHub Actions](https://docs.github.com/en/actions) to run code quality tests on pushes or pull requests to the `main` branch, a process known as [continuous integration](https://en.wikipedia.org/wiki/Continuous_integration) (CI). The tests are run for Python versions 3.7, 3.8, and 3.9, and must be successful for a PR to be merged into `main`. All workflow runs can be viewed [here](https://github.com/facebookincubator/flowtorch/actions), or else viewed from the link at the bottom of the [PR](https://github.com/facebookincubator/flowtorch/pulls) in question.


## Workflow Steps
The definition of the steps performed in the build workflow is found [here](https://github.com/facebookincubator/flowtorch/blob/main/.github/workflows/python-package.yml) and is as follows:

1. The version of Python (3.7, 3.8, or 3.9) is installed along with the developer dependencies of FlowTorch;
```bash
python -m pip install --upgrade pip
python -m pip install flake8 black usort pytest mypy
pip install numpy
pip install --pre torch torchvision torchaudio
pip install -e .[dev]
```

2. Each Python source is checked for containing the mandatory copyright header by a [custom script](https://github.com/facebookincubator/flowtorch/blob/main/scripts/copyright_headers.py):
```bash
python scripts/copyright_headers.py --check flowtorch tests scripts examples
```

3. The formatting of the Python code in the [library](https://github.com/facebookincubator/flowtorch/tree/main/flowtorch) and [tests](https://github.com/facebookincubator/flowtorch/tree/main/tests) is checked to ensure it follows a standard using [`black`](https://black.readthedocs.io/en/stable/);
```bash
black --check flowtorch tests
```
4. Likewise, the order and formatting of Python `import` statements in the same folders is checked to ensure it follows a standard using [`usort`](https://usort.readthedocs.io/en/stable/);
```bash
usort check flowtorch tests
```
5. A [static code analysis](https://en.wikipedia.org/wiki/Static_program_analysis), or rather, linting, is performed by [`flake8`](https://flake8.pycqa.org/en/latest/) to find potential bugs;
```bash
flake8 . tests --count --show-source --statistics
```
6. FlowTorch makes use of type hints, which we consider mandatory for all contributed code, and static types are checked with [`mypy`](https://github.com/python/mypy);
```bash
mypy --disallow-untyped-defs flowtorch
```
7. Unit tests:

pytest + XML coverage report
```bash
pytest --cov=tests --cov-report=xml -W ignore::DeprecationWarning tests/
```

8. The coverage report is uploaded to [Codecov](https://about.codecov.io/) with a [GitHub Action](https://github.com/codecov/codecov-action). This allows us to analyze the results and produce the percentage of code covered badge.

If any step fails, the workflow fails and you will not be able to merge the PR into `main`.

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
mypy --disallow-untyped-defs flowtorch
```
I find this is one of the most difficult steps to make pass - if you require assistance, comment on your PR, tagging the core developers.

### Formatting and Linting
Having ensured the tests and docs are correct, run the following commands to standardize your code's formatting:
```bash
black flowtorch tests
usort format flowtorch tests
```
Now, run these commands in check mode to ensure there are no errors:
```bash
black --check flowtorch tests
usort check flowtorch tests
```
It is possible you may need to fix some errors by hand.

Finally, run the linter and fix any resulting errors:
```bash
flake8 flowtorch tests
```
At this point, you are ready to commit your changes and push to the remote branch - you're a star! :star: From there, your PR will be reviewed by the core developers and after any modifications are made, merged to the `main` branch.
