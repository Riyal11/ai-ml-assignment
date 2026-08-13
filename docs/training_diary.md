# Training Diary

## Incident Template

Use this template for every training instability, tokenization mismatch, or structured-output failure.

### Run ID: <run_id>
- **Category**: instability | tokenization | structured-output
- **Observed**: What you saw (metrics, traceback, sample bad outputs)
- **Diagnosis**: How you isolated the cause
- **Change**: What you changed (config, code, data formatting)
- **Follow-up Run ID**: The run that validates the fix
- **Status**: open | resolved

## Incidents

### Incident 001: Empty Placeholder PDFs in SuperStore Dataset

**15 of 1007 PDFs failed extraction** with `invoice_number not found`.

- Error occurred in `superstore_extractor.py` during regex matching for invoice number
- No `INVOICE #` / `Row ID` / `# 36258` header
- No SuperStore header logo
- No customer name
- No line items (empty table)
- `$0.00` totals
- Only content: `INVOICE`, `Date: Jun 5, 2023`, `Bill To: Balance Due: $0.00`

**All 15 failures had identical stub content:**

Examined failed PDFs manually — they are empty placeholder/stub files. Filename contains ID (e.g., `invoice_Aaron Bergman_36260.pdf`) but PDF body has no extractable invoice data. Not a layout variation — these are fundamentally incomplete documents.

**Root cause:** source dataset (Ajigbayi SuperStore 2016) contains 15 placeholder PDFs with no meaningful content.

**Resolution:**

- Skipped all 15 PDFs intentionally (graceful degradation)
- Did **not** parse `invoice_number` from filename — would create synthetic garbage data
- Did **not** include `$0.00` / no-item invoices in training set — would teach the model bad patterns
- Extraction pipeline logs failure and continues (no crash)

**Outcome:**

- **992/1007** PDFs extracted successfully (**98.5%**)
- Splits: 830 train, 92 validation, 50 golden, 20 benchmark
- All 992 records pass `validate_invoice()` (schema + Pydantic)
- Manifest documents skipped files with reason

**Trade-off:**

- Lost 15 potential training examples
- Gained higher data quality by excluding garbage
- 830 train examples still exceeds assignment minimum (~300–500)

### Run ID: pre-training
- **Category**: bugfix
- **Observed**: Code review of `trainer.py` found the QLoRA branch calls `prepare_model_for_kbit_training(model)` but never `get_peft_model()`. LoRA/DoRA paths wrap adapters; QLoRA does not. Would train full quantized weights (OOM risk, no adapter checkpoint). Secondary issues: `mlflow.start_run(run_id=config.run_id)` passes a custom string where MLflow expects a UUID; `experiments/hyperparameter_log.csv` append may fail if parent directory or header row is missing.
- **Diagnosis**: Compared `if config.method == "qlora"` vs `elif config.method in ("lora", "dora")` branches in `src/docextract/train/trainer.py` (~L145–149). LoRA/DoRA call `get_peft_model`; QLoRA stops after k-bit prep. MLflow `run_id` misuse confirmed against MLflow API docs. CSV writer uses mode `a` without `mkdir` or header-on-first-write.
- **Change**: `trainer.py` — after `prepare_model_for_kbit_training`, wrap with `get_peft_model(model, lora_config)` using `get_qlora_config(config)[0]`; switch to `mlflow.start_run(run_name=...)` and log `custom_run_id`; ensure `experiments/` exists and write CSV header when file is new; add training start/finish log lines.
- **Follow-up Run ID**: N/A for pre-training (first QLoRA run will validate)
- **Status**: resolved

### Run ID: 20260813-124018-qlora
- **Category**: bugfix
- **Observed**: First QLoRA training attempt crashed immediately: `AttributeError: 'ActiveRun' object has no attribute 'log_param'` at `trainer.py:142`. CSV fallback also failed: `dict contains fields not in fieldnames: 'output_dir'`.
- **Diagnosis**: MLflow 3.15 `ActiveRun` context object no longer exposes `log_param`/`log_params`/`log_metrics` (verified via `dir(ActiveRun)` — empty). Module-level `mlflow.log_param()` still works. Hyperparameter CSV columns omit `output_dir` but `run_config_to_dict()` includes it in the `finally` row.
- **Change**: `trainer.py` — use `mlflow.log_param` / `mlflow.log_params` / `mlflow.log_metrics` inside the active run; filter CSV rows to `_HYPERPARAM_LOG_COLUMNS` only before `writerow`.
- **Follow-up Run ID**: next QLoRA retry after fix
- **Status**: resolved

### Run ID: 20260813-124714-qlora
- **Category**: instability
- **Observed**: QLoRA run failed at `AutoModelForCausalLM.from_pretrained` with `ValueError: Some modules are dispatched on the CPU or the disk` (bitsandbytes 4-bit quantizer). Model weights were already cached locally (`~/.cache/huggingface/hub/models--Qwen--Qwen3-4B-Instruct-2507`, ~7 GB).
- **Diagnosis**: Not a download issue. `trainer.py` used `device_map="auto"`, which on an 8 GB RTX 4060 offloads some layers to CPU when VRAM is tight. BnB 4-bit QLoRA requires the full quantized model on GPU during training — CPU/disk dispatch is rejected.
- **Change**: `trainer.py` — use `device_map={"": 0}` for QLoRA when CUDA is available; fail fast with a clear error if CUDA is missing. `lora_config.py` — enable `bnb_4bit_use_double_quant=True` to reduce VRAM footprint on 8 GB cards.
- **Follow-up Run ID**: next QLoRA retry after fix
- **Status**: resolved

### Run ID: 20260813-125200-qlora
- **Category**: instability
- **Observed**: `uv run python -c "import torch; ... get_device_name(0)"` raised `AssertionError: Torch not compiled with CUDA enabled`. Installed wheel is `torch==2.13.0+cpu`.
- **Diagnosis**: Default PyPI `torch` resolves to CPU-only on Windows. QLoRA/bitsandbytes require a CUDA build (`+cu124`). GPU hardware is fine; Python environment is wrong.
- **Change**: `pyproject.toml` — pin `torch` to PyTorch CUDA 12.4 index via `[tool.uv.sources]`; reinstall with `uv sync --reinstall-package torch`.
- **Follow-up Run ID**: verify `torch.cuda.is_available()` is True, then retry QLoRA training
- **Status**: resolved

### Run ID: 20260813-125831-qlora
- **Category**: structured-output
- **Observed**: QLoRA run loaded the model on GPU, then failed with `KeyError: 'messages'` in `_load_and_format_dataset` when using `data/train/invoices.jsonl`.
- **Diagnosis**: Extraction JSONL uses `document`/`target` fields; trainer expected pre-formatted SFT chat JSONL with `messages`. Model download and CUDA were fine.
- **Change**: `format_sft.py` — add `sft_example_from_dict` / `load_sft_examples_from_jsonl` to accept both formats; `trainer.py` — use loader with auto-conversion; add `scripts/prepare_sft.py` to materialize `data/train/sft.jsonl`.
- **Follow-up Run ID**: retry training with `invoices.jsonl` (auto-convert) or `sft.jsonl`
- **Status**: resolved

### Run ID: eval-base-20260813
- **Category**: evaluation
- **Observed**: Base model golden eval failed on Windows when `--model-path Qwen/Qwen3-4B-Instruct-2507` was passed: first `FileNotFoundError` (path treated as non-Hub), then `HFValidationError` for repo id `Qwen\Qwen3-4B-Instruct-2507` (backslash from `Path` on Windows).
- **Diagnosis**: `is_loadable_model()` checked `"/" in str(model_path)`; Windows normalizes to backslash. `HfBaseModelPredictor` received `str(Path(...))` instead of Hub-style `org/name`.
- **Change**: `inference.py` — use `model_path.as_posix()` for Hub IDs and `is_loadable_model()` detection; add `HfBaseModelPredictor` for adapter-free base eval. `pipeline.py` — add `field_precision_recall_f1` to `results.json`.
- **Follow-up Run ID**: eval-base-20260813 (third attempt succeeded)
- **Status**: resolved

### Run ID: 20260813-153345-qlora (eval regression vs base)
- **Category**: structured-output
- **Observed**: Golden-set comparison after run-002 completed:
  - **Base** (`Qwen/Qwen3-4B-Instruct-2507`, no adapter): schema **100%**, EM **0.91**, F1 **0.86** (`experiments/eval-base/results.json`)
  - **Run-001** (r=8, 3 epochs): F1 **0.72**, EM **0.74**
  - **Run-002** (r=16, 5 epochs): F1 **0.74**, EM **0.77** (`experiments/eval-ft-run-002/results.json`)
  Training loss improved (run-001 `0.29` → run-002 `0.24`) but golden F1 **decreased** vs base by **~0.12**.
- **Diagnosis**: Base instruct model already extracts valid schema JSON on SuperStore golden; QLoRA SFT on 830 English invoices appears to **hurt** field-level accuracy (possible overfit to training layout, degraded `vendor_name` / `line_items`). Per-field on base: `vendor_name` F1 **0.36** (bottleneck), `line_items` F1 **0.76**; scalar money/date/currency fields at **1.0**. Gap is not fixed by higher rank or more epochs alone.
- **Change**: **Do not promote run-002** for merge/quantize/submission. Use **base model** for quality gate and model card (conditional go: fine-tuning documented but not deployed). Run-002 artifacts retained for assignment evidence.
- **Follow-up Run ID**: N/A (base selected; no further SFT sweep planned)
- **Status**: resolved

## Training Runs

### Run 001: First QLoRA Training — SUCCESS

| Setting | Value |
|---------|-------|
| **Method** | QLoRA |
| **Base model** | `Qwen/Qwen3-4B-Instruct-2507` |
| **LoRA** | r=8, alpha=16, target_modules=`[q_proj, k_proj, v_proj, o_proj]` |
| **Optimizer** | LR=0.0002, scheduler=cosine |
| **Sequence / batch** | seq len=512, batch=1, grad_accum=8 |
| **Epochs** | 3 |
| **Hardware** | RTX 4060 Laptop 8GB |
| **Config** | `configs/train/qlora_rtx4060.yaml` |
| **Output** | `artifacts/run-001` |

**Dataset**

- 830 English SuperStore invoices (`data/train/invoices.jsonl`)
- 0 Hindi (source dataset limitation)

**Results**

- Training time: **3,769s** (~1h 3min)
- No OOM, no instability
- Final `train_loss`: **0.2885**
- Token accuracy: **94.24%**
- Trainable params: **~8.4M** (0.21% of total)

**Notes:** Validates pre-training bugfixes (QLoRA `get_peft_model`, CUDA torch, `device_map`, MLflow logging, extraction→SFT auto-convert). First successful end-to-end training run.

### Hindi Sanity Check (Post Run-001)

Post-training evaluation on golden and synthetic Hindi sets (`scripts/evaluate.py`, adapter at `artifacts/run-001`).

| Set | Examples | Schema | EM | F1 |
|-----|----------|--------|-----|-----|
| Synthetic Hindi (`data/golden/hindi_test.jsonl`) | 5 | **100%** | **0.97** | **0.96** |
| English golden (`data/golden/invoices.jsonl`) | 50 | **100%** | **0.74** | **0.72** |

**Findings**

- Hindi performance is **strong** on simple synthetic inputs.
- Gap is **not language-based**, it is **complexity-based**.
- English golden set has harder cases: multiple line items, discounts, shipping.
- Model learned the schema pattern; struggles with exact value extraction on complex invoices.
- No Hindi training data needed for this assignment.
- Qwen3-4B base model handles Devanagari via pretraining.

**Next steps:** Focus improvement on English complexity via more epochs / higher rank.

### Run 002: QLoRA r=16 Restart — SUCCESS (eval regression)

| Setting | Value |
|---------|-------|
| **Method** | QLoRA |
| **LoRA** | r=16, alpha=32 |
| **Optimizer** | LR=0.00015, scheduler=cosine |
| **Epochs** | 5 |
| **Config** | `configs/train/qlora_rtx4060_v2.yaml` |
| **Output** | `artifacts/run-002` |

**Results**

- Training time: **6,252s** (~1h 44min)
- Final `train_loss`: **0.236**
- Token accuracy: **94.81%**
- Trainable params: **~11.8M** (0.53%)
- Peak VRAM: **4,860 MB** (no OOM)

**Golden eval:** F1 **0.74**, EM **0.77**, schema **100%** — **below** run-001 and **well below** base model (see eval regression incident).

### Base vs Fine-Tuned Golden Comparison

| Model | F1 | EM | Schema | Notes |
|-------|-----|-----|--------|-------|
| Base `Qwen3-4B-Instruct-2507` | **0.86** | **0.91** | 100% | Passes F1 ≥ 0.85 gate |
| Run-001 (r=8) | 0.72 | 0.74 | 100% | |
| Run-002 (r=16) | 0.74 | 0.77 | 100% | Higher train loss ↓, golden F1 ↑ only vs run-001 |

**Decision:** Submit/document **base model** for extraction; fine-tuning runs kept as negative result (train loss ≠ golden improvement).

### Incident 003: Catastrophic Forgetting on Fine-Tuned Model

**Run ID:** run-002 (r=16, 5 epochs, 830 EN SuperStore invoices)

**Category:** catastrophic-forgetting

**Observed:**

- Base model golden F1: **0.857** (passes ≥ 0.85 threshold)
- Fine-tuned model golden F1: **0.738** (fails threshold)
- Delta: **−11.9%** F1, **−14.3%** Exact Match
- Benchmark forgetting: base F1 **0.847** → run-002 F1 **0.755** (**10.9%** relative drop; threshold ≤ 5%)
- Per-field (golden): `vendor_name` base **0.36** → run-002 **0.00**; `line_items` base **0.76** → run-002 **0.67**

**Diagnosis:**

- Training data was homogeneous: single vendor (SuperStore), single layout
- Model learned degenerate shortcut: `vendor_name` is always "SuperStore"
- Lost general capability to extract vendor names from diverse documents
- `line_items` extraction also degraded (nested array overfitting)

**Change:**

- Decision: **Do NOT deploy** fine-tuned model
- Deploy **base Qwen3-4B-Instruct-2507** instead
- Document as **conditional go** with evidence in `docs/model_card.md`

**Status:** resolved (evidence-based decision)

### Run ID: pre-submission
- **Category**: submission-prep
- **Observed**: Final submission prep — human review batch generation, base-model quality gate fix (`relative_benchmark_drop=0`), GGUF quantization attempt, lint/test validation, and documentation updates.
- **Change**: `scripts/generate_human_review.py`, `docs/human_review.md`, `docs/human_review_batch.json`, `experiments/eval-base/results.json`, `src/docextract/api/inference.py`, `src/docextract/eval/inference.py`, `SUMMARY.md`, `docs/model_card.md`
- **Status**: resolved
