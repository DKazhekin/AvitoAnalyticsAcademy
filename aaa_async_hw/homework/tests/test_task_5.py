import asyncio
import math
import random
import time

import pytest

from homework.tasks.task_5 import limit_execution_time, limit_execution_time_many


class RandomWaiter:
    def __init__(self, delay: float):
        self._delay = delay
        self._called = False
        self._should_be_cancelled = False
        self._is_cancelled = False

    @property
    def is_called(self) -> bool:
        return self._called

    @property
    def is_correctly_canceled(self) -> bool:
        if not self._should_be_cancelled:
            return True

        return self._is_cancelled

    async def random_wait(self) -> None:
        self._called = True

        delay = self._delay
        if random.random() < 0.5:
            delay *= 2
            self._should_be_cancelled = True

        try:
            await asyncio.sleep(delay)
        except asyncio.CancelledError:
            self._is_cancelled = True


@pytest.mark.asyncio
async def test_limit_execution_time():
    delay = 0.5
    n_calls = 10
    for i in range(n_calls):
        rw = RandomWaiter(delay - 0.1)
        t = time.monotonic()
        await limit_execution_time(rw.random_wait(), delay)
        assert rw.is_called, "некоторые корутины не были вызваны!"
        duration = time.monotonic() - t

        # нельзя сравнивать float друг с другом напрямую!
        assert math.isclose(duration, delay, abs_tol=0.1), "время выполнения не соответствует ожидаемому!"
        assert rw.is_correctly_canceled, "корутина не была корректно завершена!"


@pytest.mark.asyncio
async def test_limit_execution_time_many():
    delay = 0.5
    n_calls = 10
    for i in range(n_calls):
        waiters = [RandomWaiter(delay - 0.1) for _ in range(5)]
        t = time.monotonic()
        await limit_execution_time_many(*[rw.random_wait() for rw in waiters], max_execution_time=delay)
        duration = time.monotonic() - t

        assert all(rw.is_called for rw in waiters), "некоторые корутины не были вызваны!"
        # нельзя сравнивать float друг с другом напрямую!
        assert math.isclose(duration, delay, abs_tol=0.1), "время выполнения не соответствует ожидаемому!"

        # отдаём управление, чтобы отменённые задачки узнали об этом
        await asyncio.sleep(0)

        assert all(rw.is_correctly_canceled for rw in waiters), "некоторые корутины не были корректно завершены!"

