# Human Review Notes

Manual audit protocol for validating fine-tuned model outputs against the golden set.
Complements automated metrics in `src/docextract/eval/` and the CI quality gate in
`src/docextract/gates/`.

## Rubric

Each dimension is scored on a **3-point scale (0–2)**. Reviewers read the source
document and the model's JSON output side by side.

| Dimension | Score | Definition |
|-----------|-------|------------|
| **Correctness** | 0 | Multiple fields wrong or missing; values do not match the document |
| | 1 | Most fields correct; one or two minor errors (e.g. typo in vendor name) |
| | 2 | All extracted fields match the source document |
| **Faithfulness** | 0 | Hallucinated fields or values not supported by the document |
| | 1 | No invented fields, but one value inferred beyond what the text states |
| | 2 | Every value is directly grounded in the source text |
| **Formatting** | 0 | Not valid JSON, or wrapped in markdown fences / prose |
| | 1 | Valid JSON but missing required keys or wrong types |
| | 2 | Valid JSON matching the strict invoice schema (`configs/schema/invoice_schema.json`) |

**Total score (0–6):**

| Total | Interpretation |
|-------|----------------|
| 6 | Perfect |
| 4–5 | Minor issues — acceptable for production with monitoring |
| 2–3 | Major issues — requires remediation before promotion |
| 0–1 | Unusable — block deployment |

**Minimum pass rate:** ≥ 80% of the 20-sample audit must score ≥ 4/6 (see
`docs/acceptance_criteria.md`).

## Sample Set

- **Target:** ~20 outputs, balanced English/Hindi where the golden set allows
- **Source:** `data/golden/` only — never used for training or hyperparameter tuning
- **Selection:** Random stratified sample by language (aim for ~10 EN + ~10 HI)
- **Inputs:** Raw document text + model prediction JSON from the evaluation pipeline
- **Reviewers:** At least one reviewer per sample; a second reviewer for any sample
  flagged with `disagreement_flag = yes`

## Review Log Template

Fill one row per sample after the first training run completes.

| example_id | language | correctness (0-2) | faithfulness (0-2) | formatting (0-2) | total (0-6) | auto_f1 | disagreement_flag | disagreement_reason |
|------------|----------|-------------------|--------------------|--------------------|-------------|---------|-------------------|---------------------|
| Pending first training run | — | — | — | — | — | — | — | — |
| Pending first training run | — | — | — | — | — | — | — | — |
| Pending first training run | — | — | — | — | — | — | — | — |
| Pending first training run | — | — | — | — | — | — | — | — |
| Pending first training run | — | — | — | — | — | — | — | — |
| Pending first training run | — | — | — | — | — | — | — | — |
| Pending first training run | — | — | — | — | — | — | — | — |
| Pending first training run | — | — | — | — | — | — | — | — |
| Pending first training run | — | — | — | — | — | — | — | — |
| Pending first training run | — | — | — | — | — | — | — | — |
| Pending first training run | — | — | — | — | — | — | — | — |
| Pending first training run | — | — | — | — | — | — | — | — |
| Pending first training run | — | — | — | — | — | — | — | — |
| Pending first training run | — | — | — | — | — | — | — | — |
| Pending first training run | — | — | — | — | — | — | — | — |
| Pending first training run | — | — | — | — | — | — | — | — |
| Pending first training run | — | — | — | — | — | — | — | — |
| Pending first training run | — | — | — | — | — | — | — | — |
| Pending first training run | — | — | — | — | — | — | — | — |
| Pending first training run | — | — | — | — | — | — | — | — |

## Disagreement Analysis

Document cases where automated metrics and human judgment diverge. Flag any row
where `disagreement_flag = yes` and record the reason.

### Expected disagreement patterns

| Pattern | Automated signal | Human signal | Likely cause |
|---------|------------------|--------------|--------------|
| Normalization mismatch | Auto F1 = 1.0 | Correctness ≤ 1 | Date format (`2025-6-5` vs `2025-06-05`), whitespace, or decimal representation differs from gold but human accepts the extraction |
| Semantic equivalence | Auto F1 = 0.0 | Correctness ≥ 1 | Strict exact match penalizes paraphrased vendor names or equivalent numeric forms (`1000` vs `1,000.00`) that a human would accept |
| Schema vs parseability | Formatting = 2 | Schema validity = fail | Model output is parseable JSON but misses required keys or uses wrong types — human sees clean JSON, automated gate rejects |
| Line-item alignment | Auto F1 low on `line_items` | Correctness high | Index-based line-item matching in metrics misaligns reordered but correct items |
| Hindi tokenization | Auto F1 moderate | Correctness high | Devanagari text extracted correctly but tokenizer/normalization causes field-level mismatch |

### When to flag disagreement

Set `disagreement_flag = yes` when:

- Human total score differs from auto F1 by more than 0.3 (on the 0–1 scale), or
- Human formatting score is 2 but `schema_validity_rate` marks the sample invalid, or
- Human correctness is ≤ 1 but auto F1 ≥ 0.9

## Summary Statistics Template

Complete after all 20 samples are reviewed.

| Statistic | Value |
|-----------|-------|
| Mean human score (0–6) | TBD |
| Mean auto F1 | TBD |
| Spearman correlation (human total vs auto F1) | TBD |
| Number of disagreements | TBD |
| Most common disagreement type | TBD |
| Pass rate (samples scoring ≥ 4/6) | TBD |
| Pass rate threshold met (≥ 80%) | TBD |

**Reviewer sign-off:** _Pending first training run_
