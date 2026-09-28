from __future__ import annotations

import asyncio
import heapq
from dataclasses import dataclass, field
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from data_models.writer_data_models import PostWithContext


class TimedPendingQueue:
    def __init__(self) -> None:
        self._items: dict[int, tuple[float, PostWithContext]] = {}
        self._heap: list[tuple[float, int]] = []

    def schedule(self, post_id: int, eligible_at: float, post: PostWithContext) -> None:
        self._items[post_id] = (eligible_at, post)
        heapq.heappush(self._heap, (eligible_at, post_id))

    def remove(self, post_id: int) -> None:
        self._items.pop(post_id, None)

    def drain_due(self, now: float) -> list[tuple[int, PostWithContext]]:
        result: list[tuple[int, PostWithContext]] = []
        while self._heap and self._heap[0][0] <= now:
            eligible_at, post_id = heapq.heappop(self._heap)
            if post_id not in self._items:
                continue
            stored_at, post = self._items[post_id]
            if stored_at != eligible_at:
                continue
            del self._items[post_id]
            result.append((post_id, post))
        return result

    def __len__(self) -> int:
        return len(self._items)

    def __contains__(self, post_id: int) -> bool:
        return post_id in self._items


@dataclass
class Feed:
    name: str
    queue: asyncio.LifoQueue = field(default_factory=asyncio.LifoQueue)


@dataclass
class TimedFeed(Feed):
    latency_seconds: int = 0
    pending: TimedPendingQueue = field(default_factory=TimedPendingQueue)


@dataclass
class RevisionFeed(TimedFeed):
    pass


@dataclass
class RetryFeed(TimedFeed):
    max_post_age_seconds: float = 0
