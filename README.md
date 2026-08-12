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
