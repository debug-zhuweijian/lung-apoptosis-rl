"""Bounded memoization of exact frozen-world scalar requests, never observations."""
from collections import OrderedDict
import math


class ScalarScoreCache:
    def __init__(self, capacity=4096):
        if isinstance(capacity, bool) or not isinstance(capacity, int) or capacity < 0:
            raise ValueError("SCORE_CACHE_CAPACITY_INVALID")
        self.capacity = capacity
        self.values = OrderedDict()
        self.hits = self.misses = 0

    def get(self, key):
        if key not in self.values:
            self.misses += 1
            return None
        self.hits += 1
        self.values.move_to_end(key)
        return self.values[key]

    def put(self, key, value):
        value = float(value)
        if not math.isfinite(value) or not 0 <= value <= 1:
            raise ValueError("INVALID_ENDPOINT_OUTPUT")
        if self.capacity:
            self.values[key] = value
            self.values.move_to_end(key)
            while len(self.values) > self.capacity:
                self.values.popitem(last=False)
        return value
