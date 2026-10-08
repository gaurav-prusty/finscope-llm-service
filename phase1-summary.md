# finscope-llm-service - Phase 1 summary

This is the record of the genuinely hard judgment calls made across all 12 parts of Phase 1,
why each one was decided the way it was, and how it actually landed once built and tested -
written so a future FinScope phase (finscope-rag, finscope-agents, finscope-mcp,
finscope-platform) can understand *why* this repo is built the way it is, not just read the code
and guess.

---

## Part 1 - EDGAR fetch

**Decision**: extract filing sections with a regex heuristic (last match of the start heading,
skipping Table-of-Contents occurrences; earliest match of any end-heading candidate after that),
rather than a proper HTML/XBRL parser.

**How it landed**: validated against Apple's real FY2025 10-K at build time, then re-validated
against Microsoft's (Part 9) before trusting it as a second regression fixture - it generalized
correctly both times, including correctly skipping the Table-of-Contents occurrence. Documented
explicitly as a v1 best-effort heuristic, not a guarantee. This is a real, acknowledged fragility
of the repo: a sufficiently differently-formatted 10-K could break the boundary detection.

## Part 2 - Output schema

**Decision**: `FilingMeta` (deterministic, our own EDGAR fetch) and `FilingAnalysis` (the only
model sent to the LLM) are two separate models, never merged - we never ask the LLM to
re-derive facts (company name, ticker, dates) we already have from a source of truth.

Four further design calls, all decided the same direction - favor structure and calibration
honesty over convenience:
- **Financial highlights as `{metric, value, period}` strings, not floats** - filings report
  units inconsistently ("$416.2 billion" vs "416,199" thousands); forcing numeric types risks
  mangled values.
- **Risk factors categorized, not one prose blob** - queryable/comparable later, matches Item
  1A's own itemized structure, avoids later phases having to regex-parse free text back apart.
- **Sentiment as a tone enum + one-sentence rationale, not a numeric score** - LLM-generated
  continuous scores are known to be poorly calibrated and inconsistent run-to-run.
- **A `caveats` list instead of a self-reported confidence number** - "what couldn't be
  determined" is a verifiable claim; "0.82 confidence" is not.

**How it landed**: every nested model sets `additionalProperties: false` (a structural test
guards this recursively). Constraints Anthropic's structured-output API can't enforce
server-side (`min_length`, non-empty strings) are still enforced client-side by pydantic after
the fact - this became the exact mechanism Part 5's repair loop hooks into.

## Part 3 - Provider-agnostic LLM client

**Decision**: `LLMClient` is an ABC with one method at first (`generate_structured`), kept
deliberately task-agnostic - it doesn't know `FilingAnalysis` exists.

**How it landed**: the payoff wasn't theoretical. Part 5's repair-loop tests, Part 6's retry
tests, and Part 8's streaming tests all use fake `LLMClient` implementations to deterministically
force failures the real API wouldn't reliably reproduce on demand. Without the interface, testing
the repair loop would have meant hoping the real model misbehaved on command.

## Part 4 - Prompt-as-code + versioning

**Decision**: prompt versions are immutable per-version modules (`v1.py`, never edited once
shipped; a wording change means writing `v2.py`) behind a tiny registry, plain Python
strings/functions rather than a templating engine.

**How it landed**: writing the actual v1 prompt surfaced a real tension before any code proved
it either way - the pipeline only fetches Item 1A (Risk Factors), but `financial_highlights`
requires ≥1 entry, and a risk-factors-only excerpt has no headline revenue/margin figures. Rather
than redesign the schema, the prompt was written to instruct the model to extract whatever
concrete figures *do* appear in the text rather than invent summary numbers. Flagged as untested
until Part 5's live call - see below for how it actually resolved.

## Part 5 - Repair-then-fail policy

**Decision**: one repair attempt on a `pydantic.ValidationError`, then fail loudly
(`SummarizationFailedError`) - locked in at plan time. The two open design calls: repair-retry
logic lives in `services/summarize.py`, not the client (repair-then-fail is business policy, not
a generic client capability); repair-prompt wording lives in the versioned prompt module, so it
freezes alongside whichever prompt version produced the original attempt.

**A real mechanism discovered, not assumed**: read the Anthropic SDK's own source
(`lib/_parse/_response.py`) to confirm `messages.parse()` runs full pydantic validation
client-side and raises the *original* `ValidationError` uncaught - before any raw response text
is available to us. That fact shaped the repair prompt: it can describe the validation failure
precisely (pydantic's error names the exact field and constraint) but can't show the model its
own prior output verbatim, because we never get it back on a validation failure.

**How Part 4's open question resolved**: the live end-to-end call on the real AAPL fixture
validated on the *first* attempt (no repair needed) and produced a real but minor financial
highlight ("Manufacturing purchase obligation coverage period: up to 150 days") rather than a
hallucinated revenue figure - the fallback instruction worked as designed. Whether that's a
*good* highlight (versus just a schema-valid one) was correctly deferred as a quality question to
Part 9, not treated as solved here.

## Part 6 - Non-determinism engineering

**Decision**: tenacity is made the *one* retry layer - the Anthropic SDK's own silent retry is
explicitly disabled (`max_retries=0`) rather than layered underneath tenacity's. Retries only
transient failures (connection errors, 429, 5xx) - never a 4xx (fails identically every time) and
never `pydantic.ValidationError` (a different failure mode entirely, owned by Part 5).

**How it landed**: the boundary between "the call didn't complete" (Part 6, client layer) and
"the call completed but the content is wrong" (Part 5, service layer) held up under both live and
scripted testing. Mixing them would have quietly broken Part 5's "exactly one repair attempt"
guarantee.

## Part 7 - Cost telemetry

**Decision**: hardcode standard (non-promotional) per-model pricing rather than a time-limited
intro discount that happened to be active; unpriced models return `None`, never `0.0`.

**How it landed**: both choices are about the failure mode of the telemetry itself, not the happy
path - a discount that silently expires understates cost with no warning; `0.0` for an unknown
model is indistinguishable from "genuinely free."

## Part 8 - Endpoints + the streaming/repair tradeoff

**Decision**: `/summarize/stream` deliberately does **not** get Part 5's repair-then-fail
guarantee. `/summarize` (blocking) is unchanged and keeps it.

**Why**: read the SDK's streaming internals and confirmed structured-output validation on a
stream only resolves once a content block completes - near the end of the stream, after most
content has already been sent to the client. There's no clean way to retry mid-stream without the
caller having already seen most of a response we're about to discard. Rather than build a
half-working repair-in-a-stream, this was documented as a real, deliberate scope boundary in the
client's own docstring.

## Part 9 - Prompt regression tests

**Decision**: didn't assume the plan's "held-out fixture filings" (plural) was satisfied by
reusing the one AAPL fixture already exercised throughout Parts 4–8.

**How it landed**: live-fetched a second company (Microsoft) and empirically confirmed the Part 1
extraction regex generalized *before* trusting it as a fixture - it correctly stopped at the real
Item 1B boundary, not a coincidental match. Regression assertions target shape/quality (risk
count, category diversity, the financial-highlights fallback still working, sentiment never
reading as "confident" on pure risk disclosure) rather than schema validity, which pydantic
already guarantees on every real call - these tests exist to catch a prompt edit that stays
schema-valid but quietly gets worse.

## Part 10 - Dockerize

**A bug caught by reasoning, not by failure**: while designing the non-root-user Dockerfile step,
reasoned through (before ever running `docker build`) that `edgar.py`'s disk cache write would
`PermissionError` under an unprivileged user unless `/app` was explicitly `chown`'d - a failure a
bare `docker build` or a `/health`-only check would never surface, since `/health` never touches
EDGAR.

**A bug caught live**: `docker run --env-file` does not strip quotes around values the way
`python-dotenv` does. A quoted `ANTHROPIC_API_KEY` in `.env` produced a literal-quote-included key
and a real 401 from Anthropic - correctly mapped to a 502 by Part 8's exception handler rather
than crashing, which was itself a small confirmation that Part 8's error handling worked as
designed under a real failure, not just a scripted one.

## Part 11 - AWS deploy

The most eventful part of the phase. Every finding below was diagnosed from a real error, not
anticipated in the plan.

**Deploy-target pivot**: the original plan targeted AWS App Runner. Discovered live (owner
checked the AWS console directly) that App Runner has no free tier at all - it bills for
provisioned container memory continuously, never scales to zero. Researched alternatives and
pivoted to **AWS Lambda (container image) + Lambda Web Adapter + Function URL**: perpetual free
tier, scales to zero, free managed HTTPS. Every markdown contract (this repo's CLAUDE.md/SKILLS.md,
`finscope-ai-roadmap.md`, `model-choice.md`) was rewired *before* touching AWS, so documentation
and reality never diverged mid-deploy.

**A claim corrected mid-session**: initially described Lambda's Secrets Manager integration as
"the same shape" as App Runner's native secret-resolution. That was wrong, and was caught and
corrected by checking AWS's own docs rather than letting the assumption stand. Lambda has no live
secret-to-env-var resolution at all. The real choice, put to the owner explicitly rather than
decided silently: the official **AWS Parameters and Secrets Lambda Extension** (the "correct"
cloud-native pattern, but distributed as a Lambda Layer - a zip-packaging concept that doesn't
attach cleanly to container-image functions - and requires a real code change to fetch secrets
via a local HTTP call instead of `os.environ`) versus a **deploy-time snapshot** (Secrets Manager's
value copied into Lambda's env config at deploy time - zero app code changes, Secrets Manager
stays the source of truth you copy from, at the cost of no live auto-rotation). Chose the
snapshot, deliberately, given a zero-traffic learning-project context and `config.py` untouched
since Part 0.

**Region migration mid-deploy**: started in `ap-south-2` (Hyderabad - lowest latency from
Bangalore). A risk was explicitly flagged in advance ("newer regions sometimes lag on niche
service availability") and then confirmed real: **Lambda Function URLs are not available in
ap-south-2 at all**. Migrated Secrets Manager, ECR, and Lambda to `ap-south-1` (Mumbai) - IAM and
the Budgets alarm needed no changes, since both are global/account-level, not regional. Worth
remembering for any future phase choosing a region: check feature availability for the *specific*
services you need before committing, not just "is the region open."

**IAM least-privilege working exactly as designed, twice**: the IAM user created for CLI/console
work (`finscope-deployer`) was deliberately scoped to only the services actually needed at setup
time (ECR, Secrets Manager, App Runner, CloudWatch Logs). This correctly blocked it from (a)
Lambda's auto-role-creation (`iam:CreateRole` denied) and (b) even *listing* existing roles
(`iam:ListRoles` denied) - fixed by creating the execution role once as root and referencing it
by ARN manually, rather than widening the IAM user's permissions just to route around the
friction. Separately, the pivot from App Runner to Lambda meant the IAM user's policies were
stale (still had `AWSAppRunnerFullAccess`, no Lambda permission at all) - a real gap from
updating documentation during the pivot without also updating the actual AWS-side permissions;
fixed by swapping in `AWSLambda_FullAccess`.

**A Docker/Lambda incompatibility, diagnosed not guessed**: the first image push was rejected by
Lambda ("image manifest ... not supported"). Root cause (confirmed via research, not assumed):
Docker's BuildKit builder attaches provenance/SBOM attestations by default since Docker Desktop
23+, producing a multi-manifest OCI image index that Lambda can't parse - Lambda only accepts a
plain single-architecture Docker v2 manifest. Fixed with
`docker build --provenance=false --sbom=false --platform linux/amd64`.

**The first genuinely required app code change since Part 0**: `/summarize` crashed on its first
real Lambda invocation with `OSError: Read-only file system: '.cache'`. Diagnosed from the actual
CloudWatch traceback, not guessed from symptoms. Lambda's execution environment mounts the entire
container filesystem read-only except `/tmp` - a fundamentally different constraint from Part
10's non-root permission fix (that solved *who* can write; this is *nothing* can write, root or
not). Fixed by making `edgar.py`'s cache directory a `Settings` field (`edgar_cache_dir`,
defaulting to the original `.cache/edgar`) instead of a hardcoded path - local and Docker
behavior stayed byte-for-byte unchanged (confirmed with a full green `pytest -q` before
redeploying), and only Lambda gets `EDGAR_CACHE_DIR=/tmp/.cache/edgar` via one new environment
variable.

**Function URL auth, a deliberate cost/security tradeoff**: `NONE` auth (public) chosen over
`AWS_IAM` (which would require SigV4-signed requests, breaking plain-`curl` testing). Flagged
explicitly as real exposure - anyone who finds the URL can invoke it and consume Anthropic
budget; Part 6's rate limiter is the only guard, not a hard cap - accepted given the low-traffic
learning-project context and the budget alarm as backstop, not adopted as a default worth
copying into anything with real traffic.

## Part 12 - Smoke-test gate + write-up

**Decision**: the smoke-test gate's "N concurrent requests without crashing" item was reframed
around what "crashing" actually means for an HTTP service - a request that gets *any* real HTTP
response (including a 429 from the rate limiter under burst load) is handled correctly; only a
connection failure or timeout with no response at all counts as a crash. `scripts/smoke_test.py`
fires a real concurrent burst against the live Function URL and reports both, not just a pass/fail
against 200s.

**How it landed**: 20/20 concurrent `/health` requests got real responses (a mix of 200s and
429s - the rate limiter visibly working under real concurrent load on the live deployment, not
just in a unit test), and all 3 concurrent `/summarize` calls succeeded end-to-end against the
real pipeline.

**Documentation as a real deliverable, not an afterthought**: this document and README.md were
promoted to their own contract (CLAUDE.md's "Phase closeout contract") specifically because later
FinScope phases depend on being able to read back *why* Phase 1 is built this way without
re-deriving it from git history or chat transcripts.

## Addendum (post-Phase-1) - the provider saga: Anthropic -> Gemini -> Groq -> Cerebras

Written for Phase 2+: **the live service no longer runs on Anthropic.** Everything below happened
after Part 12 closed, driven by one fact - Anthropic credits ran out - and it is the most useful
record in this file of how provider selection actually goes wrong.

**What shipped**: four `LLMClient` implementations behind `get_llm_client()`, one `llm_provider`
setting to switch. Current default and what is deployed on Lambda: **Cerebras**, free-tier
`gpt-oss-120b`. All three endpoints work, including `POST /summarize/stream` (only Groq can't
stream structured output and returns 501). Part 3's bet - a task-agnostic interface -
held through three new providers with zero changes to `services/summarize.py` or the prompts.

**The judgment calls, and what each one taught**:

1. *Groq looked like the clear winner on paper and was unusable.* We picked it on requests/day
   (1,000 vs Gemini's 20). Its strict mode then rejected our schema twice - `additionalProperties:
   false` on every object, and every property listed in `required` (pydantic omits defaulted
   fields like `caveats`). Fixed inside `GroqClient` by rewriting a copy of the schema
   (`_to_groq_strict_schema`), deliberately NOT by changing `schemas.py`: provider quirks belong
   in the provider's client, not in the provider-neutral contract. Then the real wall: **8,000
   tokens/minute** on every structured-output model vs ~13K tokens for one full Item 1A section.
   A 413 that no retry or wait can ever fix. *Lesson: evaluate a provider on the dimension that
   binds for your payload (tokens/min vs request size), not the headline number.*
2. *An offline test encoded the bug.* The Groq guard test asserted the request carried the raw
   `model_json_schema()` - exactly what Groq rejects. It passed while production would 400.
   *Lesson: a test that mirrors the implementation proves consistency, not correctness; only a
   live call against the real API validates what you send.*
3. *Gemini worked and still couldn't carry the deploy*: 20 requests/day/model, 5/minute, and
   frequent 503 "high demand". The daily quota ran out mid-testing. Its quota is per model, so
   other Gemini models were a zero-code fallback, but the 503s made it unreliable for a same-day
   proof.
4. *Cerebras, chosen under a deadline.* Hugging Face was ruled out (~$0.10/month of credit);
   Mistral (500K tokens/min free) is the noted backup; AWS Bedrock (Nova Lite ~$0.001 per summary,
   and no API key at all on Lambda via IAM) is the best long-term AWS-native story and the likely
   upgrade path. Cerebras won on risk: same model family as the Groq work, same strict-schema
   rules, so `_to_groq_strict_schema` was reused unchanged. Built as a plain-`httpx` client
   instead of adding an SDK - no new dependency, no lockfile regeneration, no Docker dependency
   change, which mattered on a same-day deploy. One small provider-neutral `LLMProviderError`
   (HTTP status, or `None` for network failure) stands in for the SDK exception hierarchy the
   other clients get for free.
5. *Reserved output counts against the token budget.* Cerebras 429'd with 13K input +
   `max_completion_tokens=16000` against a 30K/min cap. A summary needs ~1.5K output tokens, so
   the default dropped to 8,000. A **stale `LLM_MAX_TOKENS=16000` in a local `.env`** then
   silently overrode the fixed code default (env > .env > defaults) and caused two more
   confusing 429s - the Lambda was fine only because it had no such variable.
6. *A real Part 8 bug surfaced by an upstream 503.* The provider call inside a streaming
   generator is lazy - it fires on first iteration, after the `200` header is already sent - so
   exception handlers can no longer turn a failure into a 502 ("response already started").
   `stream_summarize_filing` now catches `Exception` and yields an in-band `StreamError`
   event; the deliberately broad catch keeps `services/` free of provider SDK imports.

7. *Streaming on Cerebras: ship it off, then prove it on.* First deploy returned a clean 501 on
   `/summarize/stream` because the docs were silent on streaming + strict schemas. Reading further
   showed they flag only the legacy `json_object` mode as incompatible, and put token usage on
   the final chunk, so `CerebrasClient.stream_structured()` was implemented the same day by
   hand-parsing the OpenAI-style SSE over `httpx` (no repair step, as with every streaming path
   here). *Lesson: "unverified" is a reason to test, and a 501 guard is cheap to flip once a live
   test says yes.*

**How it landed**: redeployed 2026-10-08 on Lambda, live-verified at the same Function URL - all
three endpoints: `/health`, `/summarize` (real AAPL summaries with identical schema but different
wording and sentiment on each call: genuine non-determinism), and `/summarize/stream` (about 70
genuinely incremental SSE deltas, then a `done` event carrying the full validated summary). The
Part 9 regression thresholds pass on Cerebras output for both AAPL and MSFT. Tokens/minute is the
binding free-tier limit (a full-filing call reserves ~21K of 30K), so live tests run one per
minute and `scripts/smoke_test.py` should use `--summarize-n 1` on this provider.

**Open items for whoever picks this up**: (a) `financial_highlights` is weak filler on an
Item-1A-only excerpt (unchanged from Parts 4/5/9) - the real fix is also fetching Item 7 (MD&A),
not relaxing the schema; (b) Cerebras's 5 requests/minute and 30K tokens/minute free caps make
concurrent callers (Phase 2's ingestion/eval loops) a real constraint - budget for a paid tier or
Bedrock before building anything that calls this in a loop.
