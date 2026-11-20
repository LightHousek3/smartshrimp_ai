"""Shared monotonic budgets and native Gemini HTTP timeouts."""

import time
from contextlib import contextmanager
from contextvars import ContextVar

from google.genai import types

_deadline = ContextVar("rag_deadline", default=None)


@contextmanager
def deadline_scope(seconds):
    parent = _deadline.get()
    end = time.monotonic() + seconds
    token = _deadline.set(min(parent, end) if parent is not None else end)
    try:
        yield
    finally:
        _deadline.reset(token)


def request_options(seconds):
    end = _deadline.get()
    remaining = min(seconds, end - time.monotonic()) if end is not None else seconds
    if remaining <= 0:
        raise TimeoutError("RAG deadline exceeded")
    return types.HttpOptions(
        timeout=max(1, int(remaining * 1000)),
        retry_options=types.HttpRetryOptions(attempts=1),
    )
