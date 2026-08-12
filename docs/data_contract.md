# Data Contract

Contract for invoice-extraction structured output and the datasets that
anchor it. Single source of truth for what "valid" means and who may read
which split.

## 1. Target JSON schema

Canonical source: `configs/schema/invoice_schema.json` (JSON Schema
Draft 2020-12).

| Field | Type | Constraint |
|-------|------|------------|
| `invoice_number` | string | required |
| `vendor_name` | string | required |
| `invoice_date` | string | required, `YYYY-MM-DD`, valid calendar date |
| `line_items` | array | required, `minItems: 1` |
| `─ description` | string | required |
| `─ quantity` | number | required, `>= 0` |
| `─ unit_price` | number | required, `>= 0` |
| `subtotal` | number | required, `>= 0` |
| `tax_amount` | number | required, `>= 0` |
| `total_amount` | number | required, `>= 0` |
| `currency` | string | required, ISO 4217 `^[A-Z]{3}$` |

`additionalProperties: false` applies at both the top level and the
line-item level. Unknown fields are invalid, not ignored — strictness
matters for downstream systems that merge model output verbatim.

## 2. Pydantic model

Defined in `src/docextract/data/schemas.py`.

- `LineItem`: `description` (stripped, `min_length=1`), `quantity`
  (`Decimal >= 0`), `unit_price` (`Decimal >= 0`). `strict=True`,
  `extra="forbid"`.
- `Invoice`: all eight fields, `Decimal` for every monetary value,
  `ConfigDict(strict=True, extra="forbid")`.
- Validators: `invoice_date` regex + `date.fromisoformat` (rejects both
  wrong formats and impossible dates like `2025-02-31`), `currency`
  `^[A-Z]{3}$`, string fields strip surrounding whitespace.

Note: numeric `Decimal` coercion in strict mode accepts ints/Decimals
but not string numbers. That is intentional — model output should be
JSON-native numbers, not strings.

## 3. Record contract

`DocumentRecord` (`src/docextract/data/dataset.py`):

```
example_id: str
document: str            # raw source document text
target: dict[str, object]  # the expected JSON object
language: str            # e.g. "en", "hi"
split: Split             # TRAIN | VALIDATION | GOLDEN | BENCHMARK
```

One record per example. `target` must validate against both the JSON
Schema and the Pydantic `Invoice` model.

## 4. Split semantics and isolation

| Split | Training | Validation / checkpoint | Final certification | Forgetting check |
|-------|----------|-------------------------|---------------------|------------------|
| `train` | read | — | — | — |
| `validation` | loss computation only | read | — | — |
| `golden` | **never** | **never** | read | — |
| `benchmark` | **never** | **never** | — | read |

Enforced by `validate_train_split()` (rejects `GOLDEN`/`BENCHMARK`) and
`validate_eval_split()` (accepts all; evaluation may legitimately run on
any split). `load_train_dataset()` / `load_eval_dataset()` are the only
approved readers and currently raise `NotImplementedError` until real
loaders land.

## 5. Path isolation mechanism

`guard_train_path()` is the enforcement point. Given a path it:

1. **Resolves** the path with `Path.resolve()` — collapses `.`/`..` and
   follows symlinks, so a link that points at `data/golden` is caught.
2. **Ancestor check** against the allowed roots `data/train` and
   `data/validation` — the resolved path must be exactly a root or live
   under it.
3. **Case-insensitive** comparison, so `DATA/TRAIN/...`` and
   `data\GOLDEN\...` cannot slip through on case-insensitive
   filesystems.
4. Raises `ValueError` for anything else.

### Why symlink defense matters

`Path.resolve()` resolves symlinks on all platforms. A symlink
`data/train/link -> data/golden/file.json` resolves to the golden file,
which the ancestor check then rejects. See
`tests/test_dataset_isolation.py`.

### Boundary

Guard is checked at the loader boundary in `load_train_dataset` /
`load_eval_dataset`. Callers outside `src/docextract/data/` are expected
to route reads through those loaders, not touch `Path` directly.

## 6. Validation responsibilities

| Function | Checks | Verdict |
|----------|--------|---------|
| `validate_dict_against_json_schema()` | JSON Schema Draft 2020-12 | `(bool, errors)` |
| `validate_invoice_pydantic()` | Pydantic `Invoice` | `(bool, errors)` |
| `validate_invoice()` | both, errors tagged `source: json_schema\|pydantic` | `(bool, errors)` |

A record is valid only if **both** checks pass. Tagging lets quality
gates attribute failures to schema drift vs. model-side type/semantic
errors. Pydantic errors carry full field paths; JSON Schema errors carry
the failing keyword path.