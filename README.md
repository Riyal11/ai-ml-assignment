# docextract

Fine-tuned open-weight model that extracts structured invoice metadata from English and Hindi documents into strict JSON matching a fixed schema.

## Quick start

```bash
uv venv --python 3.11
# Windows: .venv\Scripts\activate
source .venv/bin/activate

uv pip install -e ".[dev]"
uv run pre-commit install

# Quality checks
uv run ruff check .
uv run black --check .
uv run mypy src
uv run bandit -c pyproject.toml -r src
uv run pytest -q
```

## Training

```bash
uv run python scripts/train.py \
  --config configs/train/lora_default.yaml \
  --dataset path/to/train.jsonl
```

QLoRA defaults: `configs/train/qlora_default.yaml`. Override flags: `--method`, `--r`, `--alpha`, `--lr`, `--epochs`, `--base-model`, `--output-dir`.

Base model default: `Qwen/Qwen3-4B-Instruct-2507` (see `docs/model_selection_memo.md`).

## Evaluation

```bash
# Stub CLI — evaluation pipeline exists in-package; dedicated script pending
uv run python scripts/evaluate.py --split golden
```

Programmatic entrypoint today:

```python
from pathlib import Path
from docextract.data.dataset import Split
from docextract.eval.pipeline import run_evaluation

run_evaluation(
    model_path=Path("artifacts/merged-model"),
    dataset_path=Path("data/golden/eval.jsonl"),
    output_dir=Path("experiments/eval"),
    split=Split.GOLDEN,
)
```

**Note:** `run_evaluation` currently uses stub predictions (`{}`) until real inference is wired. Results land in `results.json`.

## Quality gate

```bash
uv run python scripts/quality_gate.py \
  --results experiments/eval/results.json \
  --criteria docs/acceptance_criteria.md
```

Exits `0` if all quantitative thresholds pass, `1` otherwise. CI also runs the gate against fixture results.

## Serving

```bash
uv run uvicorn docextract.api.main:app --reload --port 8000

# Optional Celery worker (requires Redis + CELERY_BROKER_URL)
uv run celery -A docextract.jobs.quantize_task worker --loglevel=info
docker run -d -p 6379:6379 redis:alpine
```

| Variable | Default | Description |
|----------|---------|-------------|
| `DOCEXTRACT_MODEL_PATH` | `artifacts/merged-model` | HF model dir or GGUF file |
| `DOCEXTRACT_MODEL_ID` | `docextract-qwen3-4b` | Id returned by `/v1/models` |
| `DOCEXTRACT_QUANTIZATION` | `none` | Serving backend (`none`, `gguf`) |
| `CELERY_BROKER_URL` | unset | When set, quantization jobs run via Celery; otherwise sync fallback |

Endpoints: `GET /health`, `GET /v1/models`, `POST /v1/chat/completions` (JSON or SSE), `POST /extract`, `POST /v1/jobs/quantize`, `GET /v1/jobs/{job_id}`.

**Note:** `InferenceService` returns stub completions until model weights are loaded for real.

## Tests

```bash
uv run pytest -q
# CI uses: uv run pytest -q --cov-fail-under=80
```

## Directory structure

```
src/docextract/     # package: data, train, eval, api, jobs, gates, mcp, retrieval, …
scripts/            # train, quality_gate, quantize, benchmark CLIs
configs/            # schema, train YAMLs, bench prompts
docs/               # brief, acceptance criteria, model card, design notes
tests/              # pytest suite + fixtures
data/               # train / validation / golden / benchmark splits
artifacts/          # adapters, merged models, GGUF outputs
experiments/        # hyperparameter log, eval/bench results
.github/workflows/  # CI: lint, pytest, quality gate
```

## License

Code in this repository is intended for the assignment submission under the same open-use spirit as the base model: **Apache 2.0** (`Qwen/Qwen3-4B-Instruct-2507`). See `docs/model_card.md` for model license and usage caveats.
