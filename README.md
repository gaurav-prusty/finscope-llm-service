# finscope-llm-service

Phase 1 of the **FinScope** portfolio (5 repos → one AI financial-research assistant over SEC
EDGAR). This repo is a production-grade LLM API service that turns a raw SEC filing (or a
section of one) into a validated, structured JSON summary - company, period, key financial
highlights, categorized risk factors, sentiment. It is the LLM "core service" every later phase
in the portfolio calls.

**Live**: deployed on AWS Lambda (container image) behind a Function URL. See
`phase1-summary.md` for the full deploy story, including two real production-grade problems
diagnosed and fixed along the way (a Docker/Lambda image-format incompatibility, and Lambda's
read-only filesystem breaking a local disk cache). The live deployment currently runs on
**Cerebras** (free-tier `gpt-oss-120b`), redeployed 2026-10-08 after Anthropic credits ran out and
Groq/Gemini free tiers proved unusable -- see Post-Phase-1 changes in `CLAUDE.md` and
`phase1-summary.md`. All three endpoints are live: `/health`, `/summarize`, and `/summarize/stream`
(real incremental SSE).

This is also a **learning project** - built with Claude Code under a teaching contract, meaning
every non-trivial decision was narrated and reasoned through, not just generated. `phase1-summary.md`
is the record of the judgment calls; this README is the practical "how to run it" reference.

## What it does

```
POST /summarize        -> full structured JSON summary (blocking)
POST /summarize/stream -> the same summary, streamed token-by-token over SSE
GET  /health            -> liveness check
```

Given a stock ticker, the service:
1. Resolves it to a SEC CIK, fetches the company's most recent 10-K, extracts the Item 1A (Risk
   Factors) section (`app/services/edgar.py`).
2. Sends it to an LLM (Anthropic Claude, Google Gemini, Groq, or Cerebras -- whichever `LLM_PROVIDER`
   selects) with a versioned prompt (`app/llm/prompts/v1.py`), constrained to a strict Pydantic
   schema (`app/llm/schemas.py`) via that provider's structured-output API.
3. Validates the response; on a schema-validation failure, retries once with a repair prompt
   before failing loudly (`app/services/summarize.py`).
4. Returns financial highlights, categorized risk factors, a sentiment + rationale, and a
   caveats list of anything the excerpt couldn't determine.

## Architecture

```
app/
  main.py              # FastAPI app, routes, exception -> HTTP status mapping
  config.py            # typed settings (pydantic-settings), env-var-first precedence
  llm/
    client.py           # provider-agnostic LLMClient (ABC) + Anthropic/Gemini/Groq/Cerebras clients,
                          # and get_llm_client() -- resolves config.py's llm_provider toggle.
                          # supports_streaming flags which providers can stream (Groq can't)
    prompts/             # versioned prompt-as-code (v1.py, immutable once shipped)
    schemas.py           # the output contract -- every LLM response validates against this
  services/
    edgar.py             # polite, cached SEC EDGAR fetch
    summarize.py          # build prompt -> call -> validate -> repair-then-fail
  middleware/
    ratelimit.py          # hand-rolled token-bucket rate limiter
  telemetry/
    cost.py               # per-request token + $ logging
tests/                  # ~74 tests: unit, offline-mocked, and live-gated (skip without an API key
                          # for whichever provider is active)
scripts/
  smoke_test.py          # concurrency smoke test against a live deployment
Dockerfile              # multi-stage build; also runs under AWS Lambda via the Web Adapter
requirements.txt        # human-edited manifest (floor versions)
requirements.lock.txt   # frozen `uv pip freeze` snapshot -- what the Docker image installs from
phase1-summary.md     # the decisions record -- why things are built this way
```

## Tech stack

Python 3.12 · FastAPI · Uvicorn · Pydantic v2 · Anthropic Claude API (Sonnet 5) + Google Gemini
API (Gemini 3.6 Flash, free tier) + Groq API (gpt-oss-120b, free tier, 8K tokens/min -- too small
for a full filing) + Cerebras API (gpt-oss-120b, free tier -- the current default) · httpx ·
tenacity · pytest · Docker · AWS (Lambda, ECR, Secrets Manager, Budgets, IAM)

## Local setup

```powershell
uv venv
uv pip install -r requirements.txt
copy .env.example .env
# then edit .env: set SEC_USER_AGENT (a real contact string -- SEC's fair-access policy requires
# a descriptive User-Agent on every EDGAR request), and the API key for whichever LLM_PROVIDER
# you're using:
#   LLM_PROVIDER=cerebras (the default) -> set CEREBRAS_API_KEY (free tier: 65K context, 30K tokens/min,
#                                          5 req/min: https://cloud.cerebras.ai)
#   LLM_PROVIDER=gemini               -> set GEMINI_API_KEY (free tier, 20 req/day/model + frequent 503s: https://aistudio.google.com/apikey)
#   LLM_PROVIDER=groq                 -> set GROQ_API_KEY (8K tokens/min cap: a full filing is rejected with a 413)
#   LLM_PROVIDER=anthropic            -> set ANTHROPIC_API_KEY (https://console.anthropic.com/settings/keys)
# Only the active provider's key needs a real value -- see app/llm/client.py's get_llm_client().
# Note: POST /summarize/stream returns 501 under LLM_PROVIDER=groq only -- Groq's API can't combine
# structured output with streaming. All three endpoints work on cerebras, gemini and anthropic.
# Keep LLM_MAX_TOKENS=8000 (the default): Cerebras counts the reserved output budget toward its
# tokens/min cap, and a stale 16000 in your .env overrides the code default and triggers 429s.
```

## Run locally

```powershell
uvicorn app.main:app --reload
```

```powershell
curl.exe -s http://127.0.0.1:8000/health
```

## Test

```powershell
./.venv/Scripts/python.exe -m pytest -q
```

~88 tests (the live-gated ones are quota-sensitive -- under Cerebras run them one file at a time,
~65s apart; its 30K tokens/min cap fits only one full-filing call per minute), ~3 minutes (several are live-gated integration tests that make a real call against
whichever provider `LLM_PROVIDER` selects; they skip automatically if that provider's API key
isn't set, so the suite stays green without a key too -- the streaming ones also skip cleanly
under `LLM_PROVIDER=groq`, since Groq not supporting streaming is expected behavior, not a
failure). Target a single file, e.g. the prompt regression tests:

```powershell
./.venv/Scripts/python.exe -m pytest tests/test_prompts.py -v
```

## Docker

```powershell
docker build --provenance=false --sbom=false --platform linux/amd64 -t finscope-llm-service:dev .
```

(`--provenance=false --sbom=false` disables Docker BuildKit's default attestation manifests --
required for the image to also be Lambda-deployable, see `phase1-summary.md`; harmless for
local-only use too.)

```powershell
docker run -d -p 8000:8000 --env-file .env --name finscope-dev finscope-llm-service:dev
docker ps
curl.exe -s http://127.0.0.1:8000/health
```

```powershell
docker rm -f finscope-dev
```

Regenerate the lockfile after changing `requirements.txt`:
```powershell
uv pip freeze --python ./.venv/Scripts/python.exe > requirements.lock.txt
```

## Calling the API

```powershell
'{"ticker": "AAPL"}' | Out-File -Encoding utf8 -NoNewline body.json
curl.exe -s -X POST http://127.0.0.1:8000/summarize -H "Content-Type: application/json" --data-binary "@body.json"
curl.exe -s -N -X POST http://127.0.0.1:8000/summarize/stream -H "Content-Type: application/json" --data-binary "@body.json"
```

(`curl.exe`, not bare `curl` -- on Windows PowerShell, bare `curl` is an alias for
`Invoke-WebRequest` and doesn't understand `-H`/`--data-binary` the same way.)

## Deploying to AWS

Full narrative (including three real production bugs hit and fixed along the way) is in
`phase1-summary.md`. Command reference:

**Prerequisites**: an AWS account, an IAM user with `AmazonEC2ContainerRegistryFullAccess`,
`SecretsManagerReadWrite`, `AWSLambda_FullAccess`, and `CloudWatchLogsFullAccess`, and the AWS
CLI configured (`aws configure`) with that user's credentials.

**1. Store the API key in Secrets Manager** (console: Secrets Manager -> Store a new secret ->
Other type of secret -> Plaintext -> paste the raw key -> name it `finscope/<provider>-api-key`,
e.g. `finscope/cerebras-api-key` for the current default provider).

**2. Push the image to ECR:**
```powershell
aws ecr get-login-password --region <AWS_REGION> | docker login --username AWS --password-stdin <AWS_ACCOUNT_ID>.dkr.ecr.<AWS_REGION>.amazonaws.com
docker tag finscope-llm-service:dev <AWS_ACCOUNT_ID>.dkr.ecr.<AWS_REGION>.amazonaws.com/finscope-llm-service:latest
docker push <AWS_ACCOUNT_ID>.dkr.ecr.<AWS_REGION>.amazonaws.com/finscope-llm-service:latest
```

**3. Create the Lambda function** (console: Create function -> Container image -> point at the
ECR image -> architecture x86_64). Bump memory to 1024 MB and timeout to 1 min 30 sec
(Configuration -> General configuration) -- the LLM calls and retry backoff need more than
Lambda's tiny defaults.

**4. Set environment variables** (Configuration -> Environment variables): `SEC_USER_AGENT`
(plain value), `LLM_PROVIDER` (plain value, e.g. `cerebras`), and the active provider's API key env
var (`CEREBRAS_API_KEY` / `GROQ_API_KEY` / `GEMINI_API_KEY` / `ANTHROPIC_API_KEY`) as a deploy-time snapshot copied
from Secrets Manager -- Lambda has no native live-resolution of secrets as env vars the way some
other AWS compute services do; see `phase1-summary.md` for why that tradeoff was chosen
deliberately):
```powershell
aws lambda update-function-configuration --function-name finscope-llm-service --environment "Variables={LLM_PROVIDER=cerebras,CEREBRAS_API_KEY=$(aws secretsmanager get-secret-value --secret-id finscope/cerebras-api-key --query SecretString --output text --region <AWS_REGION>),SEC_USER_AGENT='your-contact-string',EDGAR_CACHE_DIR=/tmp/.cache/edgar}" --region <AWS_REGION>
```
(`EDGAR_CACHE_DIR=/tmp/.cache/edgar` is required -- Lambda's filesystem is read-only except
`/tmp`; see `phase1-summary.md`.)

**5. Create a Function URL** (Configuration -> Function URL -> Create function URL -> Auth type
`NONE`, Invoke mode `Response streaming`). `NONE` auth means the URL is publicly invocable by
anyone who has it -- a deliberate tradeoff for a low-traffic learning deployment backed by a
budget alarm, not a default to copy uncritically into a production system. `Response streaming`
invoke mode is required for `/summarize/stream` on Cerebras (and harmless under `groq`, which can't stream) --
`/summarize` still works either way. Under `groq`, `/summarize/stream` returns a fast 501 instead;
see app/llm/client.py's `supports_streaming`.

**6. Verify:**
```powershell
curl.exe -s https://<FUNCTION_URL>/health
curl.exe -s -X POST https://<FUNCTION_URL>/summarize -H "Content-Type: application/json" --data-binary "@body.json"
./.venv/Scripts/python.exe scripts/smoke_test.py https://<FUNCTION_URL>
```

**Before creating anything billable**: set an AWS Budgets alarm first (console: Budgets -> Create
budget -> Customize -> Cost budget -> Monthly -> a threshold you're comfortable with -> email
alert). Do this before Secrets Manager/ECR/Lambda exist, not after.

## See also

- `phase1-summary.md` - the decisions record: every genuinely hard judgment call across all 12
  parts of this phase, and how it actually landed.
