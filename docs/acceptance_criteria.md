# Acceptance Criteria

These thresholds must be met before any model artifact is promoted to production.

## Structured Output Validity

| Metric | Threshold | Rationale |
|--------|-----------|-----------|
| JSON Schema Validity Rate | **≥ 90%** | Downstream systems require parseable JSON; invalid output causes pipeline failure. |
| Field-Level Exact Match (EM) | **≥ 85%** | Exact value match on all 8 invoice fields. |
| Field-Level F1 | **≥ 0.85** | Micro-averaged across all fields (invoice_number, vendor_name, invoice_date, line_items, subtotal, tax_amount, total_amount, currency). |

## General Capability (Catastrophic Forgetting)

| Metric | Threshold | Rationale |
|--------|-----------|-----------|
| Base vs. Fine-Tuned on General Benchmark | **≤ 5% relative drop** on standard LM eval subset (e.g., MMLU subset, GSM8K subset) | Fine-tuning must not destroy general reasoning/language capability. |

## Human Review (Qualitative)

Scoring rubric and review protocol: `docs/human_review.md`.

| Criterion | Threshold |
|-----------|-----------|
| Correctness (field values match document) | **≥ 85%** on 20-sample manual audit |
| Faithfulness (no hallucinated fields) | **≥ 90%** |
| Formatting (strict JSON, no markdown) | **100%** |

**Human review minimum pass rate:** ≥ 80% of 20 samples scoring ≥ 4/6 on the
combined rubric (correctness + faithfulness + formatting, 0–2 each).

## CI Quality Gate

The pipeline fails if **any** of the above quantitative thresholds is not met. See `src/docextract/gates/` for the gate implementation.