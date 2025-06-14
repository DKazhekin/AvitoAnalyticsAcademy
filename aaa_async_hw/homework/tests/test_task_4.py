import pytest

from homework.tasks.task_4 import coroutines_execution_order


@pytest.mark.asyncio
async def test_coroutines_execution_order():
    assert await coroutines_execution_order() == 0x200000000007476f8a ^ 590295810358705651712
