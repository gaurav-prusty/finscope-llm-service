"""Application settings.

pydantic-settings is the Python analog of Spring's @ConfigurationProperties: it
reads typed values from the environment (and a local .env file), validates them,
and exposes them as a single object. get_settings() is lru_cached so the whole
app shares one instance — effectively a singleton config bean.

Precedence (highest first): real environment variables > .env file > defaults
here. In AWS (Part 11) the API key arrives as an injected env var from Secrets
Manager, so nothing about this file changes between local and cloud.
"""

from functools import lru_cache

from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_file=".env", extra="ignore")

    # --- LLM provider ---
    # Optional so the app can boot (and /health can pass) before the key exists.
    # The summarize path (Part 5) will fail loudly if it's missing.
    anthropic_api_key: str | None = None
    llm_model: str = "claude-sonnet-5"
    llm_max_tokens: int = 16000
    llm_timeout_seconds: float = 60.0
    # Tenacity is the ONE retry layer (Part 6) -- the SDK's own silent retry is
    # disabled (max_retries=0 on the Anthropic client) so every attempt is
    # observable and governed by this single, logged policy.
    llm_max_retries: int = 3

    # --- SEC EDGAR (Part 1) ---
    # SEC's fair-access policy REQUIRES a descriptive User-Agent with a contact.
    # Override this in .env with your real name/email before fetching.
    sec_user_agent: str = "FinScope-LLM-Service (contact: set-me@example.com)"

    # --- Rate limiting (Part 6) ---
    # A single shared token bucket across all inbound requests -- see
    # app/middleware/ratelimit.py for why (no per-client identity yet).
    rate_limit_capacity: int = 20
    rate_limit_refill_per_second: float = 5.0

    # --- App ---
    app_env: str = "local"


@lru_cache
def get_settings() -> Settings:
    return Settings()
