"""Tests for app/middleware/ratelimit.py.

TokenBucket is tested directly (no HTTP layer) for its core math, then once
through a real minimal FastAPI app to prove the middleware wiring actually
produces a 429 -- built fresh here rather than reusing app.main's shared
`app`, since that would let one test's bucket exhaustion leak into another
test's assertions (the bucket is stateful across calls by design).
"""

import time

from fastapi import FastAPI
from fastapi.testclient import TestClient

from app.middleware.ratelimit import RateLimitMiddleware, TokenBucket


def test_token_bucket_allows_bursts_up_to_capacity() -> None:
    bucket = TokenBucket(capacity=3, refill_per_second=0.0)

    assert bucket.try_acquire() is True
    assert bucket.try_acquire() is True
    assert bucket.try_acquire() is True
    assert bucket.try_acquire() is False  # capacity exhausted, no refill


def test_token_bucket_refills_over_time() -> None:
    bucket = TokenBucket(capacity=1, refill_per_second=20.0)  # 1 token every 50ms

    assert bucket.try_acquire() is True
    assert bucket.try_acquire() is False
    time.sleep(0.1)
    assert bucket.try_acquire() is True


def _build_test_app(capacity: int, refill_per_second: float) -> FastAPI:
    app = FastAPI()
    app.add_middleware(RateLimitMiddleware, bucket=TokenBucket(capacity, refill_per_second))

    @app.get("/ping")
    def ping() -> dict[str, bool]:
        return {"ok": True}

    return app


def test_requests_within_capacity_all_succeed() -> None:
    client = TestClient(_build_test_app(capacity=3, refill_per_second=0.0))

    for _ in range(3):
        assert client.get("/ping").status_code == 200


def test_requests_beyond_capacity_get_429() -> None:
    client = TestClient(_build_test_app(capacity=2, refill_per_second=0.0))

    assert client.get("/ping").status_code == 200
    assert client.get("/ping").status_code == 200
    response = client.get("/ping")

    assert response.status_code == 429
    assert response.json()["detail"]
