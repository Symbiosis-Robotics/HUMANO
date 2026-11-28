from __future__ import annotations

import time
from collections import defaultdict
from collections.abc import Iterator
from contextlib import contextmanager
from datetime import datetime


def timestamp_str() -> str:
    now = datetime.now()
    return now.isoformat().replace(":", "-").replace(".", "-")


class Timer:
    def __init__(self):
        self.reset()

    @contextmanager
    def __call__(self, key: str) -> Iterator[None]:
        self.tick(key)
        try:
            yield None
        finally:
            self.tock(key)

    def reset(self) -> None:
        self.counts = defaultdict(int)
        self.times = defaultdict(float)
        self.start_times = {}

    def tick(self, key: str) -> None:
        if key not in self.start_times:
            self.start_times[key] = time.time()

    def tock(self, key: str) -> None:
        if key not in self.start_times:
            raise ValueError(f"Timer is not ticking for key: {key}")
        self.counts[key] += 1
        self.times[key] += time.time() - self.start_times[key]
        del self.start_times[key]

    def get_time(self, key: str, reset: bool = True) -> float:
        ret = self.times[key]
        if reset:
            self.times[key] = 0
            self.counts[key] = 0
        return ret

    def get_average_times(self, reset: bool = True) -> dict[str, float]:
        ret = {key: self.times[key] / self.counts[key] for key in self.counts}
        if reset:
            self.reset()
        return ret
