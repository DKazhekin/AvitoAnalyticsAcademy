import random
from unittest import mock

import pytest

from homework.tasks.task_2 import fix_this_code


@pytest.mark.asyncio
@mock.patch('homework.tasks.task_2.magic_func')
async def test_fix_this_code(f):
    f.return_value = random.random()

    assert await fix_this_code()

    f.assert_called_once()
    f.assert_awaited_once()
