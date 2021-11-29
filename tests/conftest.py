# Copyright (c) Meta Platforms, Inc

import random

import numpy as np
import pytest
import torch


@pytest.fixture(scope="function", autouse=True)
def set_seeds_before_every_test():
    torch.manual_seed(42)
    np.random.seed(42)
    random.seed(42)

    yield  # yield control to the test to run
