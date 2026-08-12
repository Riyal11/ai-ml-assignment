# Model Selection Memo

**Date:** 2026-08-12 — **Decision context:** invoice metadata extraction (EN/HI) → strict JSON, single-GPU budget, open-weight.

## Comparison

| Criterion | Qwen3-4B-Instruct (RECOMMENDED) | Llama-3.2-3B-Instruct |
|---|---|---|
| **Parameters / architecture** | ~3.8B, dense decoder, GQA | ~3.2B, dense decoder, GQA |
| **Context length** | 32K | 128K |
| **Release date** | 2025-04-29 | 2024-09-25 |
| **License** | Apache 2.0 | Llama Community License |
| **Enterprise risk** | Permissive, no usage caps, redistributable, no legal review needed | <700M MAU cap, attribution + acceptable-use clause, legal review advised for commercial deployment |
| **QLoRA VRAM (4-bit)** | ~6–7 GB (fits single consumer GPU) | ~5–6 GB |
| **Serving VRAM, GGUF Q4_K_M** | 2.50 GB file (bartowski) | ~2.0–2.5 GB file |
| **Serving VRAM, AWQ** | ~2.5 GB (official) | ~2.0 GB (community) |
| **Hindi / Devanagari** | Strong. Qwen tokenizer (~151K vocab) has dedicated Devanagari coverage from Qwen2.5 multilingual training | Supported (8 languages incl. Hindi) but English-dominant training; Devanagari coverage adequate, token efficiency worse than English |
| **Chat template** | ChatML (`<\|im_start\|>`), plus optional `/think` … `/no_think` thinking blocks | `BOS` + `<\|start_header_id\|>` headers, standard Llama format |
| **Structured output** | **Native JSON mode:** `response_format={"type": "json_object"}` guarantees valid JSON shell; schema enforcement then via Pydantic/jsonschema | **Prompt-engineered only.** No native mode — needs constrained decoding (grammar) or strict post-validation; higher failure rate on raw greedy decoding |

## Qwen3-4B specifics

- **Thinking control:** `/think` forces a reasoning block, `/no_think` disables it. For extraction, `/no_think` → lower latency, fewer token costs; `/think` as a fallback for hard documents.
- **Native JSON:** `response_format={"type": "json_object"}` forces a JSON object output — the primary reason this model fits our fixed-schema gate.
- **VRAM envelope:** QLoRA ~6–7 GB; quantized serving ~3.5 GB total (weights + KV cache + overhead) at Q4_K_M or AWQ — comfortably inside "no multi-GPU cluster" and leaves room for a 4K–8K context window.

## Recommendation

**Adopt Qwen3-4B-Instruct.**

1. **License is clean for production.** Apache 2.0 vs. Llama's MAU cap + acceptable-use terms removes legal friction for a commercial enterprise document pipeline.
2. **Native JSON mode directly feeds the quality gate.** Schema-validity rate ceiling is far higher than prompt-engineered JSON on Llama-3.2; this is the brief's hardest acceptance metric (≥90% validity).
3. **Better Hindi out of the box.** Qwen's tokenizer/training skews stronger on Devanagari than Llama-3.2's English-dominant corpus — matters for half the target data.
4. **Fits the hardware budget with margin.** ~6–7 GB QLoRA, ~3.5 GB serving — no multi-GPU, no exotic quantization required; `/no_think` keeps per-request cost low.

*Trade-off:* 128K context on Llama-3.2 is overkill for invoices; 32K on Qwen3-4B is ample. Nothing in the brief requires long-document processing.
