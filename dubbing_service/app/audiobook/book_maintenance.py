"""Application-owned recovery/notification ticks over canonical book job receipts."""
from __future__ import annotations

import asyncio
import logging

log = logging.getLogger(__name__)


async def run_book_maintenance(stop: asyncio.Event, *, interval=15,
                               recovery_tick=None, notification_tick=None, casting_tick=None):
    if recovery_tick is None:
        from .book_creator_recovery import maintenance_tick as recovery_tick
    if notification_tick is None:
        from .book_creator_notifications import maintenance_tick as notification_tick
    if casting_tick is None:
        from .casting_review import maintenance_tick as casting_tick
    loop = asyncio.get_running_loop()
    workers = set()

    async def execute(fn, args):
        try:
            await asyncio.to_thread(fn, *args)
        except Exception:
            log.exception("Book recovery dispatch failed; its durable receipt retains the result")

    def schedule(fn, *args):
        # The existing render dispatch owns idempotency and marks its job active
        # before scheduling. This set only lets shutdown await those operations.
        def launch():
            task = asyncio.create_task(execute(fn, args))
            workers.add(task)
            task.add_done_callback(workers.discard)
        loop.call_soon_threadsafe(launch)

    try:
        while not stop.is_set():
            for tick, kwargs in ((recovery_tick, {"schedule": schedule}), (notification_tick, {}), (casting_tick, {"schedule": schedule})):
                try:
                    await asyncio.to_thread(tick, **kwargs)
                except Exception:
                    log.exception("Book maintenance pass failed; retrying on the next bounded pass")
            try:
                await asyncio.wait_for(stop.wait(), timeout=interval)
            except TimeoutError:
                pass
    finally:
        # Never release the canonical process lock while a resumed render writes.
        await asyncio.sleep(0)
        if workers:
            await asyncio.gather(*workers, return_exceptions=True)
