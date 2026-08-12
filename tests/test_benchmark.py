"""Tests for the benchmark CLI."""

import json
from pathlib import Path
from typing import Any
from unittest.mock import patch

import pytest

from scripts.benchmark import benchmark, load_prompts, main

PROMPTS_PATH = Path(__file__).resolve().parents[1] / "configs/bench/prompts.json"
_REQUIRED_KEYS = {
    "model_path",
    "quantization",
    "hardware",
    "metrics",
    "prompts_tested",
    "methodology",
    "timestamp",
}
_METRIC_KEYS = {
    "ttft_ms",
    "throughput_tokens_per_sec",
    "memory_peak_mb",
    "total_latency_ms",
}


def _mock_benchmark_batch(
    _model_path: Path,
    batch: list[dict[str, Any]],
    _num: int,
    _max_tokens: int,
) -> tuple[list[float], list[float], list[int]]:
    count = len(batch)
    return (
        [50.0] * count,
        [200.0] * count,
        [32] * count,
    )


def test_missing_model_path_raises(tmp_path: Path) -> None:
    with pytest.raises(FileNotFoundError, match="model path not found"):
        benchmark(
            model_path=tmp_path / "missing-model",
            quantization="none",
            num_prompts=2,
            max_tokens=16,
            prompts_path=PROMPTS_PATH,
        )


def test_valid_run_returns_required_keys(tmp_path: Path) -> None:
    model_dir = tmp_path / "model"
    model_dir.mkdir()

    with (
        patch("scripts.benchmark._benchmark_transformers", side_effect=_mock_benchmark_batch),
        patch("scripts.benchmark._memory_peak_mb", return_value=512.0),
        patch("scripts.benchmark._detect_hardware", return_value={"gpu": None, "vram_mb": None}),
    ):
        results = benchmark(
            model_path=model_dir,
            quantization="none",
            num_prompts=2,
            max_tokens=16,
            prompts_path=PROMPTS_PATH,
        )

    assert _REQUIRED_KEYS <= results.keys()
    assert _METRIC_KEYS <= results["metrics"].keys()
    assert results["prompts_tested"] == 2
    assert results["methodology"] == "warmup=2, batch=1, sequential"


def test_output_json_schema_is_valid(tmp_path: Path) -> None:
    model_dir = tmp_path / "model"
    model_dir.mkdir()
    output_path = tmp_path / "bench_results.json"

    hardware = {"gpu": "test-gpu", "vram_mb": 8192}
    with (
        patch("scripts.benchmark._benchmark_transformers", side_effect=_mock_benchmark_batch),
        patch("scripts.benchmark._memory_peak_mb", return_value=256.0),
        patch("scripts.benchmark._detect_hardware", return_value=hardware),
    ):
        exit_code = main(
            [
                "--model-path",
                str(model_dir),
                "--num-prompts",
                "2",
                "--max-tokens",
                "16",
                "--prompts",
                str(PROMPTS_PATH),
                "--output",
                str(output_path),
            ]
        )

    assert exit_code == 0
    payload = json.loads(output_path.read_text(encoding="utf-8"))
    assert _REQUIRED_KEYS <= payload.keys()
    ttft = payload["metrics"]["ttft_ms"]
    assert {"mean", "p50", "p95"} <= ttft.keys()


def test_invalid_quantization_raises_value_error(tmp_path: Path) -> None:
    model_dir = tmp_path / "model"
    model_dir.mkdir()
    with pytest.raises(ValueError, match="awq benchmarking is not implemented"):
        benchmark(
            model_path=model_dir,
            quantization="awq",
            num_prompts=2,
            max_tokens=16,
            prompts_path=PROMPTS_PATH,
        )


def test_load_prompts_reads_config() -> None:
    prompts = load_prompts(PROMPTS_PATH)
    assert len(prompts) == 10
    assert prompts[0]["language"] in {"en", "hi"}
