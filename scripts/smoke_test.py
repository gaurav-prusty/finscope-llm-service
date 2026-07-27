"""Concurrency smoke test (smoke-test gate item 2: "handles N concurrent
    requests without crashing").

Usage:
    ./.venv/Scripts/python.exe scripts/smoke_test.py <base_url>

Fires a burst of concurrent requests at /health (free -- tests the raw
infra/middleware stack, including the rate limiter, under real
concurrency) and a smaller burst at /summarize (real EDGAR + Anthropic
calls -- costs real money, kept small on purpose).

"Without crashing" means every request gets a real HTTP response back (even
a 429 from the rate limiter is a correct, handled response, not a crash) --
not that every request returns 200. A crash is a connection failure or
timeout with no response at all.
"""

import argparse
import concurrent.futures
import time

import httpx


def _timed_call(fn) -> tuple[int | None, float, str | None]:
    start = time.monotonic()
    try:
        response = fn()
        return response.status_code, time.monotonic() - start, None
    except Exception as e:
        return None, time.monotonic() - start, f"{type(e).__name__}: {e}"


def _run_burst(label: str, fn, n: int) -> None:
    print(f"\n--- {label}: {n} concurrent requests ---")
    with concurrent.futures.ThreadPoolExecutor(max_workers=n) as pool:
        results = list(pool.map(lambda _: _timed_call(fn), range(n)))

    responded = [(status, duration) for status, duration, _ in results if status is not None]
    crashed = [(duration, error) for status, duration, error in results if status is None]

    print(f"  {len(responded)}/{n} got an HTTP response (no connection failure/timeout)")
    if responded:
        durations = [d for _, d in responded]
        statuses = sorted({s for s, _ in responded})
        print(f"  status codes seen: {statuses}")
        print(
            f"  duration: min={min(durations):.2f}s max={max(durations):.2f}s "
            f"avg={sum(durations) / len(durations):.2f}s"
        )
    for duration, error in crashed:
        print(f"  NO RESPONSE (crash): duration={duration:.2f}s error={error}")


def main() -> None:
    parser = argparse.ArgumentParser(description="Concurrency smoke test against a live deployment.")
    parser.add_argument("base_url", help="e.g. https://xxxx.lambda-url.ap-south-1.on.aws")
    parser.add_argument("--health-n", type=int, default=20, help="Concurrent /health requests (free)")
    parser.add_argument(
        "--summarize-n", type=int, default=3, help="Concurrent /summarize requests (real LLM cost)"
    )
    args = parser.parse_args()
    base_url = args.base_url.rstrip("/")

    _run_burst("/health", lambda: httpx.get(f"{base_url}/health", timeout=30.0), args.health_n)
    _run_burst(
        "/summarize (real LLM calls -- costs money, kept small)",
        lambda: httpx.post(f"{base_url}/summarize", json={"ticker": "AAPL"}, timeout=90.0),
        args.summarize_n,
    )


if __name__ == "__main__":
    main()
