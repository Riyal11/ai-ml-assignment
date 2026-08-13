# Model Card: docextract-Qwen3-4B-Invoice

Model card for the invoice extraction system. Production deployment uses the
**base** Qwen3-4B-Instruct model; QLoRA fine-tuning was attempted but degraded
golden-set performance (documented below). Follows the [MLflow Model Card](https://mlflow.org/docs/latest/model-registry.html#model-cards) structure.

## Model Details

| Property | Value |
|----------|-------|
| **Model name** | docextract-Qwen3-4B-Invoice (base model for serving) |
| **Base model** | [Qwen/Qwen3-4B-Instruct-2507](https://huggingface.co/Qwen/Qwen3-4B-Instruct-2507) |
| **Fine-tuning method** | QLoRA via PEFT + TRL attempted (run-001 r=8, run-002 r=16); **not deployed** |
| **Production artifact** | Base model (no adapter merge) |
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

Automated evaluation on golden and benchmark splits. Status reflects acceptance
thresholds in `docs/acceptance_criteria.md`. Results from `experiments/eval-base/`
(English golden, 50 examples) unless noted.

| Metric | Base Model (Golden) | Fine-Tuned run-002 (Golden) | Threshold | Status |
|--------|---------------------|----------------------------|-----------|--------|
| Schema Validity Rate | **100%** | **100%** | ≥ 90% | Pass |
| Field-level F1 | **0.857** | 0.738 | ≥ 0.85 | **Base passes; FT fails** |
| Exact Match | **0.910** | 0.766 | ≥ 0.75 | Pass (base) |
| Catastrophic Forgetting (relative F1 drop on benchmark) | — | **10.9%** (retention 0.89) | ≤ 5% | **FT fails; evidence of forgetting** |
| Hindi F1 (synthetic eval, 50 ex) | **0.960** | — | ≤ 0.05 gap vs EN | Pass (base multilingual) |

Human review metrics (see `docs/human_review.md`):

| Metric | Value | Threshold | Status |
|--------|-------|-----------|--------|
| Mean human score (0–5) | **3.25** | ≥ 4.0 | Below threshold |
| Perfect extractions | **6/20 (30%)** | — | — |
| Pass rate (≥ 4/6 on rubric) | Below 80% | ≥ 80% | Below threshold |

**Qualitative assessment (20 samples, base model):**

- Human review score: **3.25/5.0** (20 samples)
- Perfect extractions: **6/20 (30%)**
- Top error: **vendor_name** confusion (product brand mistaken for invoice issuer)
- Secondary error: hallucinated category-code line items with **0** `unit_price`

## Evaluation Data

| Split | Size | Purpose | Training access |
|-------|------|---------|-----------------|
| Training set | 830 synthetic SuperStore examples | Supervised fine-tuning (QLoRA) | Read |
| Validation set | 92 examples | Held-out from extraction split | Loss only |
| Golden set | 50 held-out examples | Final certification | **Never** |
| Benchmark set | 20 examples | Catastrophic forgetting detection | **Never** |

Split isolation is enforced by `guard_train_path()` in `src/docextract/data/dataset.py`.
See `docs/data_contract.md` for the full contract.

## Training Data

- **Source:** Synthetic SuperStore PDF extraction → JSON pairs (no real customer data)
- **Languages:** English only in training (830 examples); Hindi via base model pretraining
- **Domain:** SuperStore invoices (single vendor layout — see caveats)
- **Format:** Chat-templated SFT examples (`src/docextract/data/format_sft.py`)
- **Schema:** Fixed 8-field invoice schema with `additionalProperties: false`

## Quantitative Analyses

Planned breakdowns to run after the first evaluation:

### Per-language

| Language | Schema Validity | Field F1 | Exact Match |
|----------|----------------|----------|-------------|
| English (en) — base | 100% | 0.857 | 0.910 |
| Hindi (hi) — base, synthetic eval | 100% | 0.960 | 0.960 |

### Per-field (English golden, base model)

| Field | Precision | Recall | F1 |
|-------|-----------|--------|-----|
| invoice_number | 1.00 | 1.00 | 1.00 |
| vendor_name | 0.36 | 0.36 | 0.36 |
| invoice_date | 1.00 | 1.00 | 1.00 |
| line_items | 0.74 | 0.81 | 0.76 |
| subtotal | 1.00 | 1.00 | 1.00 |
| tax_amount | 1.00 | 1.00 | 1.00 |
| total_amount | 1.00 | 1.00 | 1.00 |
| currency | 1.00 | 1.00 | 1.00 |

Fine-tuned run-002 per-field breakdown (golden, `experiments/eval-ft-run-002-v2/`):

| Field | Base F1 | run-002 F1 | Delta |
|-------|---------|------------|-------|
| invoice_number | 1.00 | 1.00 | 0.00 |
| vendor_name | 0.36 | **0.00** | −0.36 |
| invoice_date | 1.00 | 1.00 | 0.00 |
| line_items | 0.76 | 0.67 | −0.09 |
| subtotal | 1.00 | 0.98 | −0.02 |
| tax_amount | 1.00 | 0.42 | −0.58 |
| total_amount | 1.00 | 0.96 | −0.04 |
| currency | 1.00 | 1.00 | 0.00 |

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

- **Fine-tuning on homogeneous SuperStore data caused catastrophic forgetting** — QLoRA runs (r=8, r=16) reduced golden F1 from 0.86 (base) to ~0.74 (FT)
- Base model confuses `vendor_name` vs customer on some layouts (F1 0.36 on golden)
- Nested `line_items` are the second bottleneck (F1 0.76 on base)
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

### Quantization

Note: GGUF quantization was attempted but could not be completed because
`scripts/quantize.py` does not resolve Hugging Face Hub model IDs on Windows and
the `convert_hf_to_gguf.py` / `llama-quantize` toolchain was not available in
this environment. Only unquantized base benchmark results are included
(`experiments/bench_results_base.json`).

## Go / No-Go Recommendation

**Status: CONDITIONAL GO — Use Base Model**

**Evidence:**

| Metric | Base Model | Fine-Tuned (run-002) | Threshold | Winner |
|--------|------------|----------------------|-----------|--------|
| Schema Validity | 100% | 100% | ≥ 90% | Tie |
| Field F1 | **0.857** | 0.738 | ≥ 0.85 | **Base** |
| Exact Match | **0.910** | 0.766 | ≥ 0.75 | **Base** |

**Finding:** Fine-tuning on 830 homogeneous SuperStore invoices caused catastrophic forgetting. The fine-tuned model overfit to the single-vendor layout and lost general extraction capability. Training loss improved (0.29 → 0.24) while golden F1 dropped ~12 points vs base.

**Recommendation:**

- Deploy the **base Qwen3-4B-Instruct-2507** model for production extraction.
- Retain run-001/run-002 artifacts as evidence that fine-tuning was attempted and evaluated.
- Future work: retrain with diverse vendor names, invoice layouts, and mixed-language data.

**Hindi:** Base model handles Devanagari via native multilingual pretraining. Synthetic eval (`data/golden/hindi_eval.jsonl`, 50 examples): F1 **0.96**, schema **100%**.

**Catastrophic forgetting (benchmark, 20 examples):** Base F1 **0.847** vs run-002 F1 **0.755** — **10.9% relative drop** (threshold ≤ 5%). Fine-tuned model fails the forgetting gate; another reason to deploy base.
