import random

import pytest


@pytest.fixture(autouse=True)
def random_seed():
    random.seed(42)
