import asyncio
import random
from unittest.mock import AsyncMock

import pytest

from homework.tasks.task_1 import await_my_func


@pytest.mark.asyncio
async def test_await_my_func_callable():
    m = AsyncMock(return_value=random.random())
    assert await await_my_func(m) == m.return_value

    m.assert_called_once()
    m.assert_awaited_once()


@pytest.mark.asyncio
async def test_await_my_func_coroutine():
    m = AsyncMock(return_value=random.random())
    assert await await_my_func(m()) == m.return_value

    m.assert_called_once()
    m.assert_awaited_once()


@pytest.mark.asyncio
async def test_await_my_func_task():
    m = AsyncMock(return_value=random.random())
    t = asyncio.create_task(m())
    assert await await_my_func(t) == m.return_value

    m.assert_called_once()
    m.assert_awaited_once()
