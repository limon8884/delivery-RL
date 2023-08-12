from time import time
from collections import defaultdict
from typing import Optional


class TimeLogger():
    def __init__(self, prefix: Optional[str]) -> None:
        self.prefix = prefix
        self.last_ts = time()
        self.total_timings = defaultdict(float)
        self.num_calls = defaultdict(int)

    def __call__(self, key: Optional[str] = None):
        # assert self.last_ts is not None, 'call start method before using'
        now = time()
        if key:
            self._update(key, now - self.last_ts)
        self.last_ts = now

    def _update(self, key, interval):
        self.num_calls[key] += 1
        self.total_timings[key] += interval

    def get_timings(self):
        return {self.prefix + key: self.total_timings[key] / self.num_calls[key] for key in self.total_timings}
