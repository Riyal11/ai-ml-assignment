# Integration Design

Day 3 §3.3 write-up: MCP-style document-store connector, hybrid retrieval POC,
and separation of deterministic rules from probabilistic extraction. Full
implementations are out of scope; stubs live under `src/docextract/mcp/` and
`src/docextract/retrieval/`.

---

## Architecture

```
[Document Store] → [MCP Connector] → [Hybrid Retriever] → [Extraction Model]
                                                              ↓
                                                    [Rule Engine] → [Human Review Queue]
                                                              ↓
                                                    [Validated JSON] → [ERP/API]
```

| Stage | Role |
|-------|------|
| Document Store | SharePoint / S3 / ERP attachment vault |
| MCP Connector | Typed fetch / list / write tools |
| Hybrid Retriever | BM25 + dense → top-k candidates |
| Extraction Model | Probabilistic invoice JSON |
| Rule Engine | Deterministic checks; **flag only** |
| Human Review | Resolve flagged discrepancies |
| ERP/API | Downstream consumer of accepted JSON |

---

## 1. MCP-Style Connector

**Target:** enterprise document store (SharePoint, S3, or ERP vault). Interface
is store-agnostic; adapters bind one backend.

### Tools (JSON-RPC methods)

| Method | Params | Returns |
|--------|--------|---------|
| `fetch_document` | `doc_id: str` | `bytes \| str` (+ content type in adapter metadata) |
| `get_invoice_schema` | _(none)_ | `dict` — canonical invoice JSON Schema |
| `write_extraction_result` | `doc_id: str`, `invoice_json: dict`, `content_hash: str` | `bool` (accepted) |
| `list_pending_documents` | `limit: int = 100` | `list[dict]` (`doc_id`, optional `language`, `received_at`) |

### Typed I/O (Pydantic sketch)

| Model | Fields |
|-------|--------|
| `FetchDocumentRequest` | `doc_id: str` |
| `WriteExtractionRequest` | `doc_id`, `invoice_json`, `content_hash` |
| `ListPendingRequest` | `limit: int` (1–1000) |
| `PendingDocument` | `doc_id`, `language?`, `received_at` |

### Errors / auth / idempotency

| Concern | Design |
|---------|--------|
| Errors | Structured codes: `not_found`, `permission_denied`, `rate_limited`, `validation_failed`, `conflict` |
| Auth | `Authorization: Bearer <API_KEY>`; optional OAuth2 client-credentials for Graph/STS |
| Idempotency | `sha256(doc_id + ":" + content_hash)` — retries no-op; hash mismatch → `conflict` |

**Stub:** `src/docextract/mcp/connector.py` — abstract `DocumentConnector` with
`fetch`, `write`, `list_pending` (no backend logic).

---

## 2. Hybrid Sparse + Dense Retrieval POC

**Problem:** Find relevant invoices before extraction; do not run the LLM on
every file in a large corpus.

| Channel | Method | Strength |
|---------|--------|----------|
| Sparse | BM25 (`rank-bm25` / Whoosh) | Exact tokens: vendor, invoice #, dates |
| Dense | BGE-small / E5 + cosine | Semantic / EN–HI paraphrase |
| Hybrid | Linear mix **or** RRF | Robust under both query styles |

```
score = α · normalize(bm25) + (1 − α) · cosine          # weighted
rank_hybrid(d) = 1/(k + rank_sparse) + 1/(k + rank_dense)  # RRF, k≈60
```

| POC choice | Value |
|------------|-------|
| Corpus | ~100 synthetic invoice texts (EN/HI) |
| Indexing | chunk → BM25 index + dense vectors → combined metadata |
| Example query | `"Find invoices from Acme Corp in March 2024"` |
| Retrieval | `top_k=10` → feed each hit to extraction |

**Stub:** `src/docextract/retrieval/hybrid.py` — `HybridRetriever.search(query, top_k)`
structure only; no index build.

---

## 3. Separation of Concerns (graded)

Deterministic rules **must not** silently overwrite model output.

| Layer | Type | Responsibility | Example |
|-------|------|----------------|---------|
| Rule Engine | Deterministic | Schema, math, currency | `total ≈ subtotal + tax` |
| Model | Probabilistic | Text → entities (EN/HI) | Extract `vendor_name` from Hindi |
| Orchestrator | Hybrid | Route by confidence / flags | Low confidence → human review |

### Principles

1. Rules **never** silently override model JSON.
2. Rules **flag** discrepancies (e.g. `model_total=100.00, computed=105.00`).
3. Human decides which value to trust.
4. Raw model output is always kept for audit.

### Post-processing

1. Model emits raw JSON.
2. Schema validator (JSON Schema + Pydantic `Invoice`).
3. Rule engine: line sums, totals, ISO currency, calendar dates.
4. Pass → accept → `write_extraction_result`.
5. Fail → `needs_review` with model JSON **and** rule findings.
6. Never auto-correct without an audit log entry.

Maps to existing code: `validate_invoice`, `POST /extract`,
`docs/human_review.md`, quality gate thresholds.

---

## Scope

| In scope | Out of scope |
|----------|--------------|
| This design doc | Full SharePoint/S3 adapters |
| `connector.py` / `hybrid.py` stubs | Production BM25/vector cluster |
| Interface contracts above | Silent auto-correction of fields |
