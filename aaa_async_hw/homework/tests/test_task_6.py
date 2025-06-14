import asyncio
import time
from typing import Coroutine

import pytest

from homework.tasks.task_6 import AbstractLongTaskCreator, FastHandlerWithLongBackgroundTask, BackgroundCoroutinesWatcher


class SleepingTaskCreator(AbstractLongTaskCreator):
    def __init__(self, delay: float):
        self._delay = delay
        self._called_count: int = 0
        self._cancelled_count: int = 0

    @property
    def called_count(self) -> int:
        return self._called_count

    @property
    def cancelled_count(self) -> int:
        return self._cancelled_count

    def create_long_task(self) -> Coroutine:
        return self._long_task()

    async def _long_task(self):
        self._called_count += 1
        if self._called_count % 2 == 0:
            return

        try:
            await asyncio.sleep(self._delay)
        except asyncio.CancelledError:
            self._cancelled_count += 1


class RunningTasksExposedBCW(BackgroundCoroutinesWatcher):
    @property
    def running_tasks_count(self) -> int:
        return len(self._running_tasks)


@pytest.mark.asyncio
async def test_fast_handler_with_long_background_task_just_works():
    sleep_delay = 0.1
    n_runs = 10

    task_creator = SleepingTaskCreator(delay=sleep_delay)
    bcw = RunningTasksExposedBCW()
    handler = FastHandlerWithLongBackgroundTask(task_creator, bcw)

    start_time = time.monotonic()

    for _ in range(n_runs):
        await handler.handle_request()

    assert time.monotonic() - start_time < sleep_delay, "Хендлер исполняется слишком долго!"

    # Дадим запланированным задачкам поработать
    await asyncio.sleep(0)
    assert task_creator.called_count == n_runs, "Созданная корутина не запустилась"

    # Дадим запланированным корутинам подчистить хвосты
    await asyncio.sleep(0)
    assert bcw.running_tasks_count == n_runs // 2, "Ссылки на завершённые корутины не удаляются"

    # Завершим запущенные задачки принудительно
    await handler.close()

    # Дадим отменённым корутинам поработать
    await asyncio.sleep(0)

    assert task_creator.cancelled_count == n_runs // 2, "Запущенная корутина не была отменена"
