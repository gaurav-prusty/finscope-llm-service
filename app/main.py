"""FastAPI application entrypoint.

Run locally with:
    uvicorn app.main:app --reload

This is the Spring Boot @SpringBootApplication equivalent: it builds the app
object, and uvicorn (the ASGI server — think embedded Tomcat) serves it.
Routes are added with decorators instead of @RestController/@GetMapping, but
the shape is the same: a function per endpoint, a return value FastAPI
serializes to JSON.
"""

from fastapi import FastAPI

from app.config import get_settings
from app.middleware.ratelimit import RateLimitMiddleware, TokenBucket

app = FastAPI(
    title="FinScope LLM Service",
    description="Turns SEC filings into validated, structured JSON summaries.",
    version="0.1.0",
)

_settings = get_settings()
app.add_middleware(
    RateLimitMiddleware,
    bucket=TokenBucket(
        capacity=_settings.rate_limit_capacity,
        refill_per_second=_settings.rate_limit_refill_per_second,
    ),
)


@app.get("/health")
def health() -> dict[str, str]:
    """Liveness check. No LLM call here — this must stay fast and free."""
    settings = get_settings()
    return {"status": "ok", "env": settings.app_env}
