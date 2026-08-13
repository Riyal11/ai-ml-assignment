# Model Card: docextract-Qwen3-4B-Invoice

Model card for the fine-tuned invoice extraction model. Follows the [MLflow Model
Card](https://mlflow.org/docs/latest/model-registry.html#model-cards) structure.
Quantitative results are placeholders until the first training and evaluation run
complete.

## Model Details

| Property | Value |
|----------|-------|
| **Model name** | docextract-Qwen3-4B-Invoice |
| **Base model** | [Qwen/Qwen3-4B-Instruct-2507](https://huggingface.co/Qwen/Qwen3-4B-Instruct-2507) |
| **Fine-tuning method** | LoRA / QLoRA via PEFT + TRL (`SFTTrainer`) |
| **Model type** | Causal language model, instruction-tuned |
| **Parameters** | ~3.8B base; trainable adapter params logged per run |
| **License** | Apache 2.0 |
| **Framework** | PyTorch, Transformers, PEFT |
| **Intended use** | Extract structured invoice metadata from English and Hindi documents into strict JSON |

See `docs/model_selection_memo.md` for the base-model comparison and selection rationale.

## Intended Use

### Primary use

Enterprise invoice and purchase-order digitization: convert plain-text or OCR output
into machine-readable JSON for downstream ERP, accounting, or workflow systems.

### Supported inputs

- Plain-text invoices (digital PDFs converted to text)
- OCR output from scanned documents (clean, typed text)
- English and Hindi source documents (including mixed-language invoices)

### Supported outputs

Strict JSON matching the invoice schema defined in `configs/schema/invoice_schema.json`
and validated by Pydantic models in `src/docextract/data/schemas.py`.

### Out-of-scope uses

- Handwritten document extraction
- Free-form summarization or Q&A
- Legal or financial advice
- Processing documents containing sensitive PII without appropriate governance controls

## Factors

Performance may vary based on:

| Factor | Impact |
|--------|--------|
| **Language (EN vs HI)** | Hindi may show slightly lower field F1 due to tokenizer optimization for English-dominant corpora |
| **Document quality** | OCR noise, missing characters, and layout artifacts degrade extraction accuracy |
| **Schema complexity** | More line items increase alignment difficulty and context length usage |
| **Numeric format** | Decimal separators (`1.000,50` vs `1,000.50`), currency symbols vs ISO codes |
| **Date format** | Model is trained for `YYYY-MM-DD`; other formats may require post-processing |
| **Context length** | Documents exceeding `max_seq_length` (default 2048 tokens) are truncated |

## Metrics

Automated evaluation on validation and golden splits. Status reflects acceptance
thresholds in `docs/acceptance_criteria.md`.

| Metric | Validation | Golden | Threshold | Status |
|--------|-----------|--------|-----------|--------|
| Schema Validity Rate | TBD | TBD | ≥ 90% | TBD |
| Field-level F1 | TBD | TBD | ≥ 0.85 | TBD |
| Exact Match | TBD | TBD | ≥ 0.75 | TBD |
| Catastrophic Forgetting (retention ratio) | TBD | TBD | ≥ 0.90 | TBD |
| Hindi F1 Gap (|EN F1 − HI F1|) | TBD | TBD | ≤ 0.05 | TBD |

Human review metrics (see `docs/human_review.md`):

| Metric | Value | Threshold | Status |
|--------|-------|-----------|--------|
| Mean human score (0–6) | TBD | ≥ 4.0 | TBD |
| Pass rate (≥ 4/6) | TBD | ≥ 80% | TBD |

## Evaluation Data

| Split | Size | Purpose | Training access |
|-------|------|---------|-----------------|
| Training set | ~300–500 synthetic examples | Supervised fine-tuning | Read |
| Validation set | Derived from training split | Loss monitoring, checkpoint selection | Loss only |
| Golden set | ~50 held-out examples | Final certification | **Never** |
| Benchmark set | General-capability subset | Catastrophic forgetting detection | **Never** |

Split isolation is enforced by `guard_train_path()` in `src/docextract/data/dataset.py`.
See `docs/data_contract.md` for the full contract.

## Training Data

- **Source:** Synthetic document → JSON pairs (no real customer data)
- **Languages:** Mixed English and Hindi
- **Domain:** Invoices, purchase orders, and similar structured business documents
- **Format:** Chat-templated SFT examples (`src/docextract/data/format_sft.py`)
- **Schema:** Fixed 8-field invoice schema with `additionalProperties: false`

## Quantitative Analyses

Planned breakdowns to run after the first evaluation:

### Per-language

| Language | Schema Validity | Field F1 | Exact Match |
|----------|----------------|----------|-------------|
| English (en) | TBD | TBD | TBD |
| Hindi (hi) | TBD | TBD | TBD |

### Per-field

| Field | Precision | Recall | F1 |
|-------|-----------|--------|-----|
| invoice_number | TBD | TBD | TBD |
| vendor_name | TBD | TBD | TBD |
| invoice_date | TBD | TBD | TBD |
| line_items | TBD | TBD | TBD |
| subtotal | TBD | TBD | TBD |
| tax_amount | TBD | TBD | TBD |
| total_amount | TBD | TBD | TBD |
| currency | TBD | TBD | TBD |

### Line-item subfields

| Subfield | Precision | Recall | F1 |
|----------|-----------|--------|-----|
| description | TBD | TBD | TBD |
| quantity | TBD | TBD | TBD |
| unit_price | TBD | TBD | TBD |

## Ethical Considerations

- **Synthetic data** avoids leakage of real customer PII into the training set
- **Apache 2.0 license** permits commercial deployment without usage caps
- **Bias risk:** Synthetic data may not represent the full diversity of real-world
  invoice layouts, vendors, or regional formats
- **Language equity:** Hindi performance may lag English due to tokenizer and training
  data balance; monitor per-language metrics in production
- **Hallucination risk:** The model may invent field values under ambiguous input;
  human review and schema validation gates mitigate but do not eliminate this

## Caveats and Recommendations

### Known limitations

- Struggles with handwritten documents and low-quality scans
- Date format variations beyond `YYYY-MM-DD` are not guaranteed
- Currency symbols (₹, $) may appear instead of ISO 4217 codes (`INR`, `USD`)
- Multi-page documents exceeding the context window are truncated
- Line-item order sensitivity in automated metrics may not reflect human judgment

### Recommended use

Structured digital invoices and clean OCR output. Pair model output with
deterministic post-processing (date normalization, currency mapping) and schema
validation before downstream ingestion.

### Production monitoring

- Track **schema validity rate** on a rolling sample of live requests
- Alert when validity drops below 90% or Hindi F1 gap exceeds 0.05
- Log disagreements between automated and human review for periodic re-audit

## Go / No-Go Recommendation

> **Status**: [PENDING FIRST TRAINING RUN]
>
> **Evidence**:
> - Schema validity: TBD% (threshold: ≥ 90%)
> - Field F1: TBD (threshold: ≥ 0.85)
> - Forgetting: TBD (threshold: ≥ 0.90)
> - Human review pass rate: TBD (threshold: ≥ 80% scoring ≥ 4/6)
>
> **Recommendation**: [GO / CONDITIONAL GO / NO-GO — to be determined after evaluation]
