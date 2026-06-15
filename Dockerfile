# syntax=docker/dockerfile:1
FROM python:3.12-slim

# System libs for audio decoding (soundfile/librosa) and ffmpeg for mp3/m4a.
RUN apt-get update && apt-get install -y --no-install-recommends \
    libsndfile1 ffmpeg \
    && rm -rf /var/lib/apt/lists/*

# uv for dependency management
COPY --from=ghcr.io/astral-sh/uv:latest /uv /uvx /bin/

WORKDIR /app

# Install deps first for layer caching (runtime group only, no dev).
COPY pyproject.toml uv.lock ./
RUN uv sync --frozen --no-dev --no-install-project

COPY app ./app

ENV PATH="/app/.venv/bin:$PATH"
EXPOSE 8000

# Models load once at startup; keep a single worker so they aren't loaded N times.
CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8000"]
