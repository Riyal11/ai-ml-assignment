# Human Review Notes

Manual audit protocol for validating fine-tuned model outputs against the golden set.
Complements automated metrics in `src/docextract/eval/` and the CI quality gate in
`src/docextract/gates/`.

## Rubric

Each sample is scored on a **0–5 scale**. Reviewers read the source document and the
model's JSON output side by side.

| Score | Definition |
|-------|------------|
| 5 | Perfect — all fields match, no hallucinations, valid schema JSON |
| 4 | Minor issues — one field slightly wrong or formatting quirk |
| 3 | Moderate issues — a key field wrong (e.g., vendor_name) but most correct |
| 2 | Major issues — multiple fields wrong or hallucinated items |
| 1 | Critical issues — most fields wrong but some structure present |
| 0 | Unusable — invalid JSON or completely wrong extraction |

**Minimum pass rate:** ≥ 80% of the 20-sample audit must score ≥ 3/5.

## Sample Set

- **Target:** ~20 outputs, balanced English/Hindi where the golden set allows
- **Source:** `data/golden/` only — never used for training or hyperparameter tuning
- **Selection:** Random stratified sample by language (aim for ~10 EN + ~10 HI)
- **Inputs:** Raw document text + model prediction JSON from the evaluation pipeline
- **Reviewers:** At least one reviewer per sample; a second reviewer for any sample
  flagged with `disagreement_flag = yes`

## Human Review (20 samples)

| # | Example ID | Score | Key Errors |
|---|-----------|-------|------------|
| 1 | superstore-10670 | 3/5 | vendor_name="Nokia" (product brand → vendor confusion) |
| 2 | superstore-11116 | 2/5 | vendor_name="SanDisk"; hallucinated extra line items |
| 3 | superstore-12169 | 5/5 | None |
| 4 | superstore-12051 | 3/5 | vendor_name="Cisco" (product brand → vendor confusion) |
| 5 | superstore-12333 | 2/5 | vendor_name="Safco"; hallucinated 0-price category item |
| 6 | superstore-11954 | 3/5 | vendor_name="Wilson Jones"; description truncated |
| 7 | superstore-12204 | 3/5 | vendor_name="Canon" (product brand → vendor confusion) |
| 8 | superstore-11631 | 2/5 | vendor_name="Appliances, Office Supplies, OFF-AP-4731" (category leak) |
| 9 | superstore-11753 | 2/5 | vendor_name="SanDisk"; qty absorbed into description |
| 10 | superstore-10338 | 2/5 | vendor_name="Brother Wireless Fax, Digital" (description as vendor) |
| 11 | superstore-10340 | 2/5 | vendor_name="Brother Wireless Fax, Color"; extra 0-price item |
| 12 | superstore-11911 | 5/5 | None |
| 13 | superstore-11877 | 3/5 | vendor_name="Art, Office Supplies" (category as vendor) |
| 14 | superstore-11221 | 2/5 | vendor_name="Safco"; desc truncated; extra 0-price item |
| 15 | superstore-11645 | 5/5 | None |
| 16 | superstore-11508 | 5/5 | None |
| 17 | superstore-11789 | 5/5 | None |
| 18 | superstore-10403 | 4/5 | Extra 0-price category item (vendor correct) |
| 19 | superstore-10404 | 5/5 | None |
| 20 | superstore-11633 | 3/5 | vendor_name="Konica" (product brand → vendor confusion) |

**Summary:** 6/20 perfect (30%). Average score: 3.25/5.0.
**Primary failure mode:** vendor_name confusion — model extracts product brands (Nokia, SanDisk, Cisco, Canon, Konica, Safco, Wilson Jones) instead of invoice issuer "SuperStore".
**Secondary failure mode:** Hallucinated extra line items with category codes and 0 unit_price.

Source: `docs/human_review_batch.json` (base model, first 20 golden examples).

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

## Summary Statistics

| Statistic | Value |
|-----------|-------|
| Mean human score (0–5) | **3.25** |
| Perfect extractions (5/5) | **6/20 (30%)** |
| Pass rate threshold (≥ 80% scoring ≥ 3/5) | **Not met** (only 30% score ≥ 4/5) |
| Primary failure mode | `vendor_name` confusion (product brand vs issuer) |
| Secondary failure mode | Hallucinated category-code line items with 0 `unit_price` |

**Note:** Automated F1 penalizes minor formatting differences (e.g. `tax_amount: 0` vs `0.0`) and strict field mismatches that humans may accept when `invoice_number` and `total_amount` are correct.

**Reviewer sign-off:** Complete 20-sample audit (base model).
