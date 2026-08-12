# docextract

Fine-tuned open-weight model that extracts structured metadata (invoices, POs, similar) from English and Hindi documents into strict JSON matching a fixed schema.

## Setup

```bash
uv venv --python 3.11
source .venv/bin/activate
uv pip install -e ".[dev]"
uv run pre-commit install
```

## Layout

- `src/docextract/` — package: `data`, `train`, `eval`, `infer`, `serve`, `api`, `jobs`, `retrieval`, `mcp`, `gates`
- `tests/` — pytest suite
- `configs/` — schema, models, train, eval, serve configs
- `data/` — train, validation, golden (held-out), benchmark splits
- `scripts/` — one-off CLI entry points
- `experiments/` — run logs, tracking exports
- `artifacts/` — outputs (adapters, merged models, quantization)
- `.github/workflows/` — CI quality gates

## Serving

```bash
# Start FastAPI server
uv run uvicorn docextract.api.main:app --reload --port 8000

# Start Celery worker (optional — requires Redis)
uv run celery -A docextract.jobs.quantize_task worker --loglevel=info

# Start Redis (required for Celery)
docker run -d -p 6379:6379 redis:alpine
```

Environment variables:

| Variable | Default | Description |
|----------|---------|-------------|
| `DOCEXTRACT_MODEL_PATH` | `artifacts/merged-model` | Path to merged HF model or GGUF file |
| `DOCEXTRACT_MODEL_ID` | `docextract-qwen3-4b` | Model id returned by `/v1/models` |
| `DOCEXTRACT_QUANTIZATION` | `none` | Serving backend (`none`, `gguf`) |
| `CELERY_BROKER_URL` | unset | When set, quantization jobs run asynchronously via Celery |

API endpoints:

- `GET /health` — service health and model load status
- `GET /v1/models` — OpenAI-compatible model list
- `POST /v1/chat/completions` — OpenAI-compatible chat completions (JSON or SSE stream)
- `POST /extract` — invoice extraction with schema validation
- `POST /v1/jobs/quantize` — queue or run model quantization
- `GET /v1/jobs/{job_id}` — poll quantization job status
