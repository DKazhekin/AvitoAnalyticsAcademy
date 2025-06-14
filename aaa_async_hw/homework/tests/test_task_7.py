import asyncio
import math
import time

import numpy as np
import pytest
from sklearn.cluster import KMeans

from homework.tasks.task_7 import AbstractModel, Handler


class ClusteringModel(AbstractModel):
    DATASET_SIZE = 10_000
    DIMENSIONS = 10
    N_CLUSTERS = 100

    def __init__(self):
        self._called_count = 0

    @property
    def called_count(self) -> int:
        return self._called_count

    def compute(self):
        self._called_count += 1

        # heavy computations
        dataset = self._create_dataset()
        KMeans(n_clusters=self.N_CLUSTERS, n_init='auto', random_state=42).fit(dataset)

    def _create_dataset(self) -> np.ndarray:
        return np.random.randint(low=0, high=100, size=(self.DATASET_SIZE, self.DIMENSIONS))


@pytest.mark.asyncio
async def test_heavy_computations(monkeypatch):
    monkeypatch.setenv('OMP_NUM_THREADS', '1')

    model = ClusteringModel()
    handler = Handler(model)
    expected_calls_count = 0

    single_task_time = []
    for i in range(10):
        expected_calls_count += 1

        t = time.monotonic()
        await handler.handle_request()
        assert model.called_count == expected_calls_count
        single_task_time.append(time.monotonic() - t)

    single_task_time_mean = np.mean(single_task_time)
    single_task_time_std = np.std(single_task_time)
    print(f'{single_task_time_mean=}, {single_task_time_std=}')

    two_tasks_time = []
    for j in range(10):
        expected_calls_count += 2
        t = time.monotonic()
        tasks = [
            handler.handle_request()
            for _ in range(2)
        ]
        await asyncio.gather(*tasks)
        assert model.called_count == expected_calls_count
        two_tasks_time.append(time.monotonic() - t)

    two_tasks_time_mean = np.mean(two_tasks_time)
    two_tasks_time_std = np.std(two_tasks_time)
    print(f'{two_tasks_time_mean=}, {two_tasks_time_std=}')

    assert math.isclose(single_task_time_mean, two_tasks_time_mean, rel_tol=0.3)

