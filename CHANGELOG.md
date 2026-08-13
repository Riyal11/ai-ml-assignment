# Changelog

## Submission (2026-08-13)

### Added

- QLoRA fine-tuning scripts with MLflow logging
- Run-001 (r=8, 3 epochs) and run-002 (r=16, 5 epochs) QLoRA adapters
- Evaluation pipeline with per-field F1, EM, schema validity, forgetting metrics
- Base and fine-tuned evaluation artifacts
- 20-sample human review with qualitative analysis
- FastAPI service with OpenAI-compatible chat completions and `/extract` endpoint
- CI quality gate and GitHub Actions workflow
- Synthetic Hindi eval (F1 0.96)

### Changed

- Production deployment: base Qwen3-4B-Instruct-2507 (fine-tuning degraded performance)

### Documented

- Catastrophic forgetting: 10.9% benchmark drop on run-002
- GGUF quantization attempted but not completed (Windows toolchain gap)
- vendor_name confusion as primary failure mode

### Removed

- Duplicate tokenizer.json files from checkpoint directories
- Stale eval-ft-run-002 (superseded by v2 with per-field metrics)
