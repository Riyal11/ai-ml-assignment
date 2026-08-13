# SUMMARY

Honest wrap-up of trade-offs, gaps, and what another week would buy for the
docextract assignment.

## Trade-offs made

| Decision | Chose | Rejected / deferred | Why |
|----------|-------|---------------------|-----|
| Base model | **Qwen3-4B-Instruct** | Larger 7B+ / Llama-3.2-3B | Single-GPU budget; Apache 2.0; native JSON mode; stronger Hindi tokenizer coverage than Llama for this brief |
| Quantization | **GGUF Q4_K_M** | AWQ as primary | Simpler local path via llama.cpp; AWQ better for vLLM throughput but heavier toolchain for Day 3 |
| Serving stack | **llama-cpp-python** (+ Transformers for `none`) | vLLM / SGLang | Faster to stand up for a take-home; lower peak throughput and less production polish |
| Async jobs | **Celery + Redis**, with **sync fallback** | Celery-only | Works without infra for demos; async when `CELERY_BROKER_URL` is set |

## What I'd do differently with another week

- Run real LoRA/QLoRA sweeps and fill `docs/training_diary.md` with incidents, not templates.
- Add a vLLM (or SGLang) serving backend for higher throughput and proper TTFT measurement.
- Deep-dive Hindi/Devanagari tokenization (numeral fertility, chat-template quirks) instead of leaving the TODO in `tokenizer_utils`.
- Multi-page / long-invoice path: chunking or >32K context strategy with citation of which pages fed extraction.
- Production monitoring: rolling schema-validity dashboard and alert when validity or Hindi F1 gap slips.

## Data Quality Notes

- 15 of 1007 source PDFs were empty placeholders (no invoice data, `$0.00` totals)
- These were intentionally excluded to maintain training set quality
- 98.5% extraction success rate (992 valid invoices)
- All extracted records pass strict schema validation

## Known weak spots

- **QLoRA adapter wiring was broken (pre-training audit)** — `trainer.py` skipped `get_peft_model()` on the QLoRA path; fixed in pre-training bugfix commit. First real QLoRA run still pending to validate end-to-end.
- **`InferenceService` is a stub** — returns dummy JSON; no real HF/GGUF weight load in the API path yet.
- **Dataset loaders raise `NotImplementedError`** — `load_train_dataset` / `load_eval_dataset` wait on finalized on-disk format.
- **Evaluation pipeline uses stub predictions** — `_stub_predict` returns `{}`, so metrics are structural only until inference is wired.
- **No production resilience** — limited retries/circuit breakers on store I/O and model calls.
- **Benchmarks are scaffolding** — scripts and methodology exist; numbers in docs are targets, not measured GPU runs.
- **Celery path is thin** — task retries exist at the decorator level, but there is no dead-letter queue or rich failure UX.

## Anything else I'd improve

- OpenTelemetry tracing across `/extract`, chat completions, and job workers.
- Structured logs with correlation IDs per request / `doc_id`.
- MLflow Model Registry for adapter versioning and promotion gates.
- A/B framework for comparing base vs fine-tuned (or LoRA vs QLoRA) on golden without silent overwrite of outputs.

## Related docs

- `docs/model_selection_memo.md` — why Qwen3-4B
- `docs/acceptance_criteria.md` / `docs/model_card.md` — gates and go/no-go template
- `docs/quantization_notes.md` — GGUF vs AWQ/GPTQ rationale
- `docs/integration_design.md` — MCP connector, hybrid retrieval, rules vs model
- `docs/human_review.md` — qualitative rubric
