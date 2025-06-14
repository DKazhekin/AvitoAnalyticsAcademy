import asyncio
import random
import time

import hypothesis
import pytest

from homework.tasks.task_3 import coroutines_execution_order, Ticket


async def just_return_ticket(t: Ticket) -> Ticket:
    return t


async def sleep_and_return(ticket: Ticket, max_sleep_duration: float) -> Ticket:
    await asyncio.sleep(random.random() * max_sleep_duration)
    return ticket


@pytest.mark.asyncio
async def test_several_coroutines_results_easy():
    tickets = [
        Ticket(number=2, key='мыла'),
        Ticket(number=1, key='мама'),
        Ticket(number=3, key='раму'),
    ]
    coros = [just_return_ticket(t) for t in tickets]
    result = await coroutines_execution_order(coros)
    assert result == 'мамамылараму'


@pytest.mark.asyncio
@hypothesis.given(...)
async def test_several_coroutines_results_medium(keys: list[str]):
    tickets = [Ticket(number=i, key=key) for i, key in enumerate(keys)]
    random.shuffle(tickets)

    coros = [just_return_ticket(t) for t in tickets]
    result = await coroutines_execution_order(coros)
    assert result == ''.join(keys)


@pytest.mark.asyncio
@hypothesis.given(...)
async def test_several_coroutines_results_hard(keys: list[str]):
    tickets = [Ticket(number=i, key=key) for i, key in enumerate(keys)]
    random.shuffle(tickets)

    coros = [sleep_and_return(t, 0.1) for t in tickets]
    started = time.monotonic()
    result = await coroutines_execution_order(coros)
    duration = time.monotonic() - started

    assert result == ''.join(keys)
    assert duration < 0.2, "корутины должны исполняться конкурентно!"
