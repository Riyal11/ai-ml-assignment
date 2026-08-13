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
