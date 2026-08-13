# docextract

Open-weight invoice extraction pipeline for English and Hindi documents into strict
JSON matching a fixed schema. **Production deployment uses the base**
`Qwen/Qwen3-4B-Instruct-2507` model; QLoRA fine-tuning was attempted (run-001, run-002)
but degraded golden-set performance and is documented as a negative result.

## Results (submission snapshot)

| Metric | Base model | Fine-tuned (run-002) |
|--------|------------|----------------------|
| Golden F1 | **0.857** | 0.738 |
| Schema validity | **100%** | — |
| Hindi eval F1 (synthetic) | **0.960** | — |
| Human review (20 samples) | **3.25/5.0** (6 perfect) | — |

See `SUMMARY.md` for trade-offs, `docs/model_card.md` for go/no-go gates, and
`docs/human_review.md` for the qualitative audit.

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

## Data

Training and evaluation data are not committed to git (`data/` is in `.gitignore`).
To regenerate the synthetic SuperStore dataset from source PDFs:

```bash
uv run python scripts/extract_superstore_invoices.py \
  --pdf-dir data/raw/superstore \
  --output-dir data
```

Splits are written under `data/train/`, `data/validation/`, `data/golden/`, and
`data/benchmark/`.

Or use your own invoice JSONL files with the schema defined in
`configs/schema/invoice_schema.json`.

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
uv run python scripts/evaluate.py \
  --model-path Qwen/Qwen3-4B-Instruct-2507 \
  --dataset data/golden/eval.jsonl \
  --output-dir experiments/eval-base \
  --split golden
```

Use `--stub` to force empty predictions (pipeline tests only). For adapters, pass
`--model-path artifacts/run-002/checkpoint-520` (or any PEFT checkpoint dir).

Programmatic entrypoint:

```python
from pathlib import Path
from docextract.data.dataset import Split
from docextract.eval.pipeline import run_evaluation

run_evaluation(
    model_path=Path("Qwen/Qwen3-4B-Instruct-2507"),
    dataset_path=Path("data/golden/eval.jsonl"),
    output_dir=Path("experiments/eval-base"),
    split=Split.GOLDEN,
)
```

`run_evaluation` loads real Hugging Face / PEFT weights when the model path is a Hub ID
or local checkpoint; stub mode applies only to non-loadable paths or when `--stub` is set.
Results are written to `results.json` under `--output-dir`.

## Quality gate

```bash
uv run python scripts/quality_gate.py \
  --results experiments/eval-base/results.json \
  --criteria docs/acceptance_criteria.md
```

Exits `0` if all quantitative thresholds pass, `1` otherwise. CI also runs the gate against fixture results.

## Serving

```bash
# Recommended: point at the production base model (downloads from Hugging Face on first run)
# Windows: set DOCEXTRACT_MODEL_PATH=Qwen/Qwen3-4B-Instruct-2507
export DOCEXTRACT_MODEL_PATH=Qwen/Qwen3-4B-Instruct-2507
uv run uvicorn docextract.api.main:app --reload --port 8000

# Optional Celery worker (requires Redis + CELERY_BROKER_URL)
uv run celery -A docextract.jobs.quantize_task worker --loglevel=info
docker run -d -p 6379:6379 redis:alpine
```

| Variable | Default | Description |
|----------|---------|-------------|
| `DOCEXTRACT_MODEL_PATH` | `artifacts/merged-model` | HF Hub ID, local model dir, PEFT adapter, or GGUF file |
| `DOCEXTRACT_MODEL_ID` | `docextract-qwen3-4b` | Id returned by `/v1/models` |
| `DOCEXTRACT_QUANTIZATION` | `none` | Serving backend (`none`, `gguf`) |
| `CELERY_BROKER_URL` | unset | When set, quantization jobs run via Celery; otherwise sync fallback |

Endpoints: `GET /health`, `GET /v1/models`, `POST /v1/chat/completions` (JSON or SSE), `POST /extract`, `POST /v1/jobs/quantize`, `GET /v1/jobs/{job_id}`.

**Note:** `InferenceService` loads real weights for Hub IDs and valid local checkpoints.
Empty or missing local paths use a stub predictor (for API tests). GGUF quantization was
attempted but not completed on Windows — see `SUMMARY.md` and `docs/model_card.md`.

## Tests

```bash
uv run pytest -q
# CI uses: uv run pytest -q --cov-fail-under=80
```

## Directory structure

```
src/docextract/     # package: data, train, eval, api, jobs, gates, mcp, retrieval, …
scripts/            # train, evaluate, quality_gate, quantize, benchmark CLIs
configs/            # schema, train YAMLs, bench prompts
docs/               # brief, acceptance criteria, model card, design notes
tests/              # pytest suite + fixtures
data/               # train / validation / golden / benchmark splits
artifacts/          # QLoRA adapters and checkpoint metadata (no merged weights in git)
experiments/        # hyperparameter log, eval/bench results
.github/workflows/  # CI: lint, pytest, quality gate
```

## Related documentation

- `SUMMARY.md` — engineering trade-offs and known limitations
- `CHANGELOG.md` — submission changelog
- `docs/model_card.md` — model details, eval metrics, deployment decision
- `docs/data_contract.md` — JSON schema and dataset split contract
- `docs/human_review.md` — 20-sample qualitative audit
- `docs/training_diary.md` — fine-tuning incidents and catastrophic forgetting
- `docs/acceptance_criteria.md` — quantitative quality gates

## License

Code in this repository is intended for the assignment submission under the same open-use spirit as the base model: **Apache 2.0** (`Qwen/Qwen3-4B-Instruct-2507`). See `docs/model_card.md` for model license and usage caveats.
