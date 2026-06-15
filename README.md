# checkia-ml-api

A FastAPI service exposing Bambara-enabling NLP models behind a
stable HTTP contract:

- **`POST /translate`** — text translation between Bambara, French, and English.
- **`POST /transcribe`** — speech-to-text for Bambara and French audio.

Each model is selected by configuration, so models can be swapped via environment
variables without changing the API contract or any client.

## Requirements

- [uv](https://docs.astral.sh/uv/) (manages the Python 3.12 toolchain and deps)
- A Hugging Face token for gated models (set `HF_TOKEN`)
- A Google Cloud Translation API key (set `GOOGLE_TRANSLATE_API_KEY`)

## Setup

```bash
uv sync --group dev          # create the venv and install everything
cp .env.example .env         # then fill in HF_TOKEN and any model overrides
```

## Run

```bash
uv run uvicorn app.main:app --reload
```

Models are downloaded and loaded **once at startup**, so the first start is slow;
afterwards they are held in memory. Open http://127.0.0.1:8000/docs for the
interactive API.

## API

```
GET  /health      -> {"status": "ok"}
GET  /models      -> {"translation": "<id>", "asr": "<id>"}

POST /translate
  req  {"text": "I ni ce", "source_lang": "bm", "target_lang": "fr"}
  resp {"translated_text": "...", "source_lang": "bm", "target_lang": "fr", "model": "<id>"}

POST /transcribe        # multipart file upload ...
  req  file=<audio>  language=bm (optional)
       # ... or JSON with a URL:
  req  {"audio_url": "https://...", "language": "bm"}
  resp {"text": "...", "language": "bm", "duration_s": 12.4, "model": "<id>"}
```

Language codes at the boundary are `bm`, `fr`, `en`. Responses always include the
`model` that served the request. Response fields are append-only.

## Configuration

All via environment / `.env` (see `.env.example`):

| Variable    | Purpose                                   | Default                              |
|-------------|-------------------------------------------|--------------------------------------|
| `MT_MODEL`  | Translation provider label (reported as `model`) | `google-translate-v2`          |
| `GOOGLE_TRANSLATE_API_KEY` | Google Cloud Translation API key (required) | —                  |
| `ASR_MODEL` | ASR model (HF id or local path)           | `sudoping01/bambara-asr-v2`          |
| `DEVICE`    | `cpu` \| `cuda` \| `mps`                   | `cpu`                                |
| `HF_TOKEN`  | Hugging Face token (required for gated)   | —                                    |

The default `ASR_MODEL` is gated: accept its terms on Hugging Face and set
`HF_TOKEN` before first run.

## Tests

```bash
uv run pytest -m "not real"   # fast: full API contract with fake models, no downloads
uv run pytest -m real         # downloads and runs the configured models
```

The `real` ASR test is skipped automatically when `HF_TOKEN` is unset.

## Evaluation

See [`eval/README.md`](eval/README.md) — scores the configured model on a held-out
set (BLEU/ChrF for translation, WER for ASR).

## Docker

```bash
docker build -t checkia-ml-api .
docker run -p 8000:8000 --env-file .env checkia-ml-api
```
