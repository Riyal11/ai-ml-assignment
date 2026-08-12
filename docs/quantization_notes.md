# Quantization Notes

Design notes for Step 10 quantization and serving benchmarks on
**Qwen3-4B-Instruct** for the docextract invoice extraction pipeline.

## Format Choice

**Selected: GGUF Q4_K_M** (~2.5 GB via bartowski community quant)

| Format | Size | Rationale |
|--------|------|-----------|
| **GGUF Q4_K_M** (chosen) | ~2.5 GB | Best balance of size and quality for single-GPU serving; mature llama.cpp / llama-cpp-python ecosystem |
| AWQ (considered) | ~2.5 GB | Official Qwen team support; better for vLLM/SGLang but adds separate toolchain |
| GPTQ (rejected) | ~2.5 GB | Less mature for Qwen3; fewer community artifacts |
| 1-bit / Bonsai (rejected) | <1 GB | Not trainable for this assignment scope; quality trade-off too aggressive |

Conversion is handled by `scripts/quantize.py` (`--method gguf-q4_k_m`).
AWQ and GPTQ raise `NotImplementedError` with guidance to use dedicated tooling.

## Expected Hardware

| Environment | GPU | Notes |
|-------------|-----|-------|
| Development | RTX 4090 (24 GB) | Comfortable headroom for FP16 fine-tune + Q4 serving |
| Cloud | NVIDIA A10G (24 GB) | Assignment target for affordable single-GPU deployment |

## Expected VRAM

| Configuration | VRAM estimate |
|---------------|---------------|
| FP16 merged model (inference) | ~6–7 GB weights |
| Q4_K_M GGUF weights | ~2.5 GB |
| KV cache (4K context, batch=1) | ~1 GB |
| **Total serving (Q4_K_M)** | **~3.5 GB** |

Leaves ample room on a 24 GB GPU for concurrent requests or longer context when needed.

## Performance Targets

Benchmarks are run via `scripts/benchmark.py` on `configs/bench/prompts.json`.

| Metric | Target | Notes |
|--------|--------|-------|
| TTFT (time to first token) | < 500 ms | Mean across 10 prompts after 2-prompt warmup |
| Throughput | > 10 req/s | Sequential batch=1; throughput measured as tokens/sec |
| Memory peak | < 4 GB | Q4_K_M + KV cache on single GPU |

Results are written to `experiments/bench_results.json`.

## Methodology

Documented in benchmark output as `warmup=2, batch=1, sequential`:

1. **Warmup:** 2 prompts (not measured) to stabilize CUDA kernels / GGUF mmap
2. **Measurement:** `num_prompts` consecutive requests (default 10), batch size 1
3. **No concurrency:** Single-threaded sequential requests to isolate per-request latency
4. **Metrics collected:**
   - `ttft_ms`: mean, p50, p95 time to first token
   - `total_latency_ms`: mean, p50, p95 end-to-end latency
   - `throughput_tokens_per_sec`: total generated tokens / total wall time
   - `memory_peak_mb`: `torch.cuda.max_memory_allocated` (HF) or process RSS (GGUF)

## Usage

```bash
# Quantize merged HF model to GGUF Q4_K_M
uv run python scripts/quantize.py \
  --model-path artifacts/merged-model \
  --output-dir artifacts/gguf \
  --method gguf-q4_k_m

# Benchmark unquantized HF model
uv run python scripts/benchmark.py \
  --model-path artifacts/merged-model \
  --quantization none \
  --num-prompts 10

# Benchmark GGUF model
uv run python scripts/benchmark.py \
  --model-path artifacts/gguf/merged-model.Q4_K_M.gguf \
  --quantization gguf \
  --num-prompts 10
```

## Dependencies

| Backend | Required packages |
|---------|-------------------|
| GGUF conversion | `llama-cpp-python`, llama.cpp `convert_hf_to_gguf.py`, `llama-quantize` |
| HF benchmarking | `transformers`, `torch` |
| GGUF benchmarking | `llama-cpp-python` |
| Memory (optional) | `psutil` for RSS when CUDA unavailable |

Scripts fail gracefully with logged error messages when optional dependencies are missing.
