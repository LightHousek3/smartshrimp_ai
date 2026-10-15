import math
import secrets
import threading
import time
from collections import deque

from app.core.errors import ServiceError


def authenticate(provided, expected):
    if not provided or not secrets.compare_digest(provided.encode(), expected.encode()):
        raise ServiceError("unauthorized", "Missing or invalid API key", 401)


class SlidingWindowLimiter:
    """One configured service key; one process/replica is the deployment contract."""

    def __init__(self, limit, clock=time.monotonic):
        self.limit = limit
        self.clock = clock
        self.events = deque()
        self.lock = threading.Lock()

    def check(self):
        now = self.clock()
        with self.lock:
            while self.events and self.events[0] <= now - 60:
                self.events.popleft()
            if len(self.events) >= self.limit:
                retry = max(1, math.ceil(self.events[0] + 60 - now))
                raise ServiceError(
                    "rate_limit_exceeded",
                    "Request limit exceeded",
                    429,
                    {"Retry-After": str(retry)},
                )
            self.events.append(now)
