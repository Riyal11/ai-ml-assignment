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
