FROM python:3.12.3-slim-bookworm

# Grab just the uv binary from Astral's own distroless image (a multi-stage
# COPY --from=) -- avoids installing uv via pip inside our image, and avoids
# pulling in that image's entire toolchain, just these two files.
COPY --from=ghcr.io/astral-sh/uv:latest /uv /uvx /usr/local/bin/

# Lambda Web Adapter: lets this unmodified uvicorn app run under
# AWS Lambda -- inert under plain `docker run`, since only Lambda's own
# runtime scans /opt/extensions/.
COPY --from=public.ecr.aws/awsguru/aws-lambda-adapter:1.0.1 /lambda-adapter /opt/extensions/lambda-adapter

WORKDIR /app

# Dependencies before app code -- Docker caches each instruction as its own
# layer; if only app/ changes on a rebuild, this (slow) layer is reused from
# cache instead of reinstalling everything. Same reasoning as resolving
# Maven/Gradle deps before compiling src/.
COPY requirements.lock.txt .
RUN uv pip install --system --no-cache -r requirements.lock.txt

COPY app/ app/

# Containers run as root by default. Create an unprivileged user and hand it
# ownership of /app -- edgar.py's disk cache (.cache/edgar/, Part 1) writes
# relative to the working directory at runtime, so without this the very
# first /summarize request would crash with PermissionError the instant it
# tried to create that directory. A bare `docker build` or a /health-only
# check would never catch this -- /health never touches EDGAR.
RUN useradd --create-home --uid 1000 appuser && chown -R appuser:appuser /app
USER appuser

# AWS_LWA_PORT tells the Lambda Web Adapter which local port to proxy to --
# matches --port 8000 below. AWS_LWA_INVOKE_MODE=response_stream is the
# adapter-side half of enabling SSE streaming for /summarize/stream; the
# Function URL itself needs a matching setting too (configured later).
ENV AWS_LWA_PORT=8000
ENV AWS_LWA_INVOKE_MODE=response_stream

EXPOSE 8000

# No curl in a slim image -- stdlib urllib instead of an extra apt-get layer.
HEALTHCHECK --interval=30s --timeout=5s --start-period=5s --retries=3 \
    CMD python -c "import urllib.request; urllib.request.urlopen('http://localhost:8000/health')" || exit 1

# --host 0.0.0.0, not localhost -- otherwise the app only accepts connections
# from inside its own network namespace and -p port mapping can't reach it.
# No --reload -- that's a dev-only file-watcher. ANTHROPIC_API_KEY /
# SEC_USER_AGENT are deliberately absent from this file -- they arrive via
# `docker run -e` / `--env-file` at runtime; config.py's env-var-first
# precedence (Part 0) already handles this with zero code changes.
CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8000"]
