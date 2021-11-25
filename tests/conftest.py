# Copyright (c) Meta Platforms, Inc. and its affiliates. All Rights Reserved
# SPDX-License-Identifier: MIT
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
