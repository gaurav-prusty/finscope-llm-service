"""Global token-bucket rate limiter, protecting the service (and LLM spend)
from request bursts.

One shared bucket across every request, not one per client -- there's no
client identity yet (no auth, no API keys issued to callers), so a per-IP or
per-token bucket would just be per-IP with extra bookkeeping and unbounded
memory growth as new IPs show up. A global bucket is the honest MVP for a
single-owner service; splitting it per-client is a real Phase-2-and-later
concern once callers are actually distinguishable.

Contrast with services/edgar.py's _RateLimiter: that one enforces a strict
minimum interval between OUR outbound calls to SEC, which has a hard rate
ceiling we must never exceed. This one protects OUR inbound endpoints, where
occasional bursts are fine but sustained hammering isn't -- a token bucket
(bursts up to `capacity`, refilling over time) is the standard fit, the same
idea as Guava's RateLimiter or Resilience4j's RateLimiter in burst mode.

To test: ./.venv/Scripts/python.exe -m pytest tests/test_client.py tests/test_ratelimit.py -v
"""

import threading
import time

from starlette.middleware.base import BaseHTTPMiddleware, RequestResponseEndpoint
from starlette.requests import Request
from starlette.responses import JSONResponse, Response


class TokenBucket:
    def __init__(self, capacity: int, refill_per_second: float) -> None:
        self._capacity = capacity
        self._refill_per_second = refill_per_second
        self._tokens = float(capacity)
        self._last_refill = time.monotonic()
        self._lock = threading.Lock()

    def try_acquire(self) -> bool:
        with self._lock:
            now = time.monotonic()
            elapsed = now - self._last_refill
            self._tokens = min(self._capacity, self._tokens + elapsed * self._refill_per_second)
            self._last_refill = now
            if self._tokens < 1:
                return False
            self._tokens -= 1
            return True


class RateLimitMiddleware(BaseHTTPMiddleware):
    def __init__(self, app, bucket: TokenBucket) -> None:
        super().__init__(app)
        self._bucket = bucket

    async def dispatch(self, request: Request, call_next: RequestResponseEndpoint) -> Response:
        if not self._bucket.try_acquire():
            return JSONResponse(status_code=429, content={"detail": "Rate limit exceeded, try again shortly."})
        return await call_next(request)
