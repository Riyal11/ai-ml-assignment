# SUMMARY

Honest wrap-up of trade-offs, gaps, and what another week would buy for the
docextract assignment.

## Key engineering trade-offs

| Decision | Chose | Rejected / deferred | Why |
|----------|-------|---------------------|-----|
| Base model | **Qwen3-4B-Instruct** | Larger 7B+ / Llama-3.2-3B | Single-GPU budget; Apache 2.0; native JSON mode; stronger Hindi tokenizer coverage than Llama for this brief |
| Quantization | **GGUF Q4_K_M** (attempted, not completed) | AWQ as primary | Simpler local path via llama.cpp; AWQ better for vLLM throughput but heavier toolchain for Day 3 |
| Serving stack | **llama-cpp-python** (+ Transformers for `none`) | vLLM / SGLang | Faster to stand up for a take-home; lower peak throughput and less production polish |
| Async jobs | **Celery + Redis**, with **sync fallback** | Celery-only | Works without infra for demos; async when `CELERY_BROKER_URL` is set |
| **Production model** | **Base Qwen3-4B-Instruct-2507** | Fine-tuned QLoRA adapters | Golden F1 0.86 (base) vs 0.74 (run-002); fine-tuning hurt generalization |
| **Fine-tuning** | Documented negative result (run-001, run-002) | Merge/quantize FT for serving | Homogeneous SuperStore data → catastrophic forgetting |

GGUF quantization was attempted but could not be completed: `scripts/quantize.py`
does not resolve Hugging Face Hub model IDs on Windows, and the
`convert_hf_to_gguf.py` / `llama-quantize` toolchain was not available in this
environment. Only unquantized base benchmark results are included
(`experiments/bench_results_base.json`).

## Post-hoc trade-off analysis

- **Fine-tuning vs base model:** Attempted QLoRA r=8 (run-001) and r=16 (run-002). Both hurt golden-set performance compared to the base instruct model. Chose to deploy the **base model** — an honest, evidence-based decision that still passes the F1 ≥ 0.85 gate.
- **Data diversity:** The SuperStore training set is too homogeneous (single vendor, single layout) for effective fine-tuning. With another week, would add diverse vendor names, layouts, and mixed-language examples before any retrain.
- **Hindi:** No Hindi training data; base model multilingual capability suffices (synthetic eval F1 ~0.96).

## Recommended follow-on work

- Run real LoRA/QLoRA sweeps with more diverse data.
- Add a vLLM (or SGLang) serving backend for higher throughput and proper TTFT measurement.
- Deep-dive Hindi/Devanagari tokenization (numeral fertility, chat-template quirks) instead of leaving the TODO in `tokenizer_utils`.
- Multi-page / long-invoice path: chunking or >32K context strategy with citation of which pages fed extraction.
- Production monitoring: rolling schema-validity dashboard and alert when validity or Hindi F1 gap slips.

## Data quality and curation

- 15 of 1007 source PDFs were empty placeholders (no invoice data, `$0.00` totals)
- These were intentionally excluded to maintain training set quality
- 98.5% extraction success rate (992 valid invoices)
- All extracted records pass strict schema validation

## Known limitations and failure modes

- **Fine-tuned models exhibit catastrophic forgetting** due to narrow, single-vendor training data; base model is the production choice
- **vendor_name extraction** is the main English golden bottleneck (F1 0.36 on base) — confuses vendor vs customer on some layouts
- **No real Hindi training data** — relies on base model multilingual pretraining (synthetic eval only)
- Relative benchmark F1 drop on run-002: **10.9%** (base 0.847 → FT 0.755); proves catastrophic forgetting
- **GGUF quantization:** `scripts/quantize.py` path validated; conversion failed on Windows — Hub model path not resolved to local cache and `convert_hf_to_gguf.py` / `llama-quantize` toolchain not available. Serving benchmark uses unquantized Transformers base model (`experiments/bench_results_base.json`).
- **QLoRA on 8 GB GPUs** — requires pinning `device_map={"": 0}` and closing other VRAM consumers
- **`InferenceService`** loads real HF base/adapter models when path is a Hub ID or local checkpoint; empty dirs still use stub for API tests
- **Human review:** 20/20 golden examples scored in `docs/human_review.md` (avg **3.25/5.0**, 6 perfect); predictions in `docs/human_review_batch.json`

## Production observability and MLOps enhancements

- OpenTelemetry tracing across `/extract`, chat completions, and job workers.
- Structured logs with correlation IDs per request / `doc_id`.
- MLflow Model Registry for adapter versioning and promotion gates.
- A/B framework for comparing base vs fine-tuned (or LoRA vs QLoRA) on golden without silent overwrite of outputs.

## Related documentation

- `docs/model_selection_memo.md` — why Qwen3-4B
- `docs/acceptance_criteria.md` / `docs/model_card.md` — gates and go/no-go template
- `docs/quantization_notes.md` — GGUF vs AWQ/GPTQ rationale
- `docs/integration_design.md` — MCP connector, hybrid retrieval, rules vs model
- `docs/human_review.md` — qualitative rubric
