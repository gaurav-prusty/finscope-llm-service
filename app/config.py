"""Application settings.

pydantic-settings is the Python analog of Spring's @ConfigurationProperties: it
reads typed values from the environment (and a local .env file), validates them,
and exposes them as a single object. get_settings() is lru_cached so the whole
app shares one instance - effectively a singleton config bean.

Precedence (highest first): real environment variables > .env file > defaults
here. In AWS (Part 11) the API key arrives as an injected env var from Secrets
Manager, so nothing about this file changes between local and cloud.
"""

from functools import lru_cache

from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_file=".env", extra="ignore")

    # --- LLM provider ---
    # Which LLMClient implementation services/ actually gets -- a deployment-level
    # switch (app/llm/client.py's get_llm_client()), not a per-request choice.
    # Defaults to cerebras (live-verified end to end). Groq's free tier caps TOKENS
    # PER MINUTE at 8,000 for every structured-output model, and one full Item 1A
    # section is ~12-13K tokens, so a real filing is rejected with a 413 no matter
    # how long you wait. Gemini's free tier works but is 20 requests/day/model with
    # frequent 503 "high demand" overloads. Cerebras: full filing fits (65K context,
    # 30K tokens/min), streaming works; tradeoff is 5 requests/min. Flip to "anthropic", "groq" or "cerebras" via
    # this alone. (Cerebras: the same-day-deploy fallback when Gemini's per-model
    # daily quota and 503 overloads blocked it -- full filing fits its context.)
    llm_provider: str = "cerebras"
    # Optional so the app can boot (and /health can pass) before any key exists.
    # The summarize path (Part 5) will fail loudly if the *active* provider's key
    # is missing.
    anthropic_api_key: str | None = None
    llm_model: str = "claude-sonnet-5"
    gemini_api_key: str | None = None
    # gemini-2.5-flash/-flash-lite 404 ("no longer available to new users") on a
    # freshly created API key -- a live Google rollout restriction discovered by
    # actually calling the API, not documented anywhere findable in advance.
    # gemini-3.6-flash confirmed free-tier eligible (ai.google.dev/gemini-api/docs/pricing)
    # and live-verified working against a real key, but its free tier is only
    # 20 requests/day/model/project -- also discovered live (see CLAUDE.md).
    gemini_model: str = "gemini-3.6-flash"
    groq_api_key: str | None = None
    # Only openai/gpt-oss-20b and openai/gpt-oss-120b support strict-mode
    # structured outputs on Groq (llama-3.3-70b-versatile does NOT, despite
    # being the model usually cited for Groq's generous free tier) -- confirmed
    # via console.groq.com/docs/structured-outputs. Both gpt-oss sizes share the
    # same free-tier limits (1,000 RPD, 30 RPM, 8K TPM), so the bigger model is
    # picked for quality at no quota cost, same reasoning as choosing Sonnet 5
    # over Haiku originally.
    groq_model: str = "openai/gpt-oss-120b"
    # Cerebras free tier: gpt-oss-120b, 65K context, 30K uncached tokens/min,
    # 1M tokens/day, but only 5 requests/min (inference-docs.cerebras.ai).
    # Note the model id has no "openai/" prefix, unlike Groq's.
    cerebras_api_key: str | None = None
    cerebras_model: str = "gpt-oss-120b"
    # Output budget. A real summary is ~1.5K tokens; 8000 leaves ample room for a
    # reasoning model's thinking tokens. Was 16000, but Cerebras counts the
    # RESERVED output budget toward its 30K tokens/min free-tier cap (13K input +
    # 16K reserved => 429 "Tokens per minute limit exceeded", found live).
    llm_max_tokens: int = 8000
    llm_timeout_seconds: float = 60.0
    # Tenacity is the ONE retry layer (Part 6) -- each client disables its SDK's
    # own silent retry so every attempt is observable and governed by this one,
    # logged policy, for whichever provider is active.
    llm_max_retries: int = 3

    # --- SEC EDGAR (Part 1) ---
    # SEC's fair-access policy REQUIRES a descriptive User-Agent with a contact.
    # Override this in .env with your real name/email before fetching.
    sec_user_agent: str = "FinScope-LLM-Service (contact: set-me@example.com)"
    edgar_cache_dir: str = ".cache/edgar"

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
