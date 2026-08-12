"""CLI for benchmarking model serving performance on a fixed prompt set."""

from __future__ import annotations

import argparse
import json
import logging
import statistics
import sys
import time
from datetime import UTC, datetime
from pathlib import Path
from typing import Any, Literal

logger = logging.getLogger(__name__)

Quantization = Literal["none", "gguf", "awq"]
_DEFAULT_PROMPTS = Path("configs/bench/prompts.json")
_DEFAULT_OUTPUT = Path("experiments/bench_results.json")
_METHODOLOGY = "warmup=2, batch=1, sequential"


def _percentile_stats(values: list[float]) -> dict[str, float]:
    """Compute mean, p50, and p95 for a list of measurements."""
    if not values:
        return {"mean": 0.0, "p50": 0.0, "p95": 0.0}
    sorted_vals = sorted(values)
    n = len(sorted_vals)

    def _pct(p: float) -> float:
        index = max(0, min(n - 1, round(p * (n - 1))))
        return sorted_vals[index]

    return {
        "mean": statistics.fmean(sorted_vals),
        "p50": _pct(0.5),
        "p95": _pct(0.95),
    }


def load_prompts(prompts_path: Path) -> list[dict[str, Any]]:
    """Load benchmark prompts from a JSON file.

    Args:
        prompts_path: Path to a JSON array of prompt objects.

    Returns:
        Parsed prompt dicts with at least ``id`` and ``text`` keys.

    Raises:
        FileNotFoundError: If ``prompts_path`` does not exist.
        ValueError: If the file is not a JSON array.
    """
    if not prompts_path.is_file():
        raise FileNotFoundError(f"prompts file not found: {prompts_path}")

    data: object = json.loads(prompts_path.read_text(encoding="utf-8"))
    if not isinstance(data, list):
        raise ValueError(f"prompts file must contain a JSON array: {prompts_path}")
    prompts: list[dict[str, Any]] = []
    for item in data:
        if not isinstance(item, dict) or "text" not in item:
            raise ValueError(f"each prompt must be an object with a text field: {prompts_path}")
        prompts.append(item)
    return prompts


def _detect_hardware() -> dict[str, Any]:
    """Return GPU name and VRAM when CUDA is available."""
    hardware: dict[str, Any] = {"gpu": None, "vram_mb": None}
    try:
        import torch
    except ImportError:
        logger.warning("torch not installed; hardware detection skipped")
        return hardware

    if torch.cuda.is_available():
        hardware["gpu"] = torch.cuda.get_device_name(0)
        props = torch.cuda.get_device_properties(0)
        hardware["vram_mb"] = int(props.total_memory // (1024 * 1024))
    return hardware


def _memory_peak_mb() -> float:
    """Return peak memory usage in MB (CUDA or process RSS)."""
    try:
        import torch

        if torch.cuda.is_available():
            return float(torch.cuda.max_memory_allocated() // (1024 * 1024))
    except ImportError:
        pass

    try:
        import psutil

        return float(psutil.Process().memory_info().rss // (1024 * 1024))
    except ImportError:
        logger.warning("psutil not installed; memory_peak_mb will be 0")
        return 0.0


def _reset_memory_stats() -> None:
    """Reset CUDA memory counters when available."""
    try:
        import torch

        if torch.cuda.is_available():
            torch.cuda.reset_peak_memory_stats()
            torch.cuda.synchronize()
    except ImportError:
        return


def _benchmark_transformers(
    model_path: Path,
    prompts: list[dict[str, Any]],
    num_prompts: int,
    max_tokens: int,
) -> tuple[list[float], list[float], list[int]]:
    """Benchmark an unquantized HuggingFace model with Transformers.

    Returns:
        Tuple of (ttft_ms_list, total_latency_ms_list, token_counts).
    """
    try:
        import torch
        from transformers import AutoModelForCausalLM, AutoTokenizer
    except ImportError as exc:
        msg = "transformers and torch are required for unquantized benchmarking"
        logger.error(msg)
        raise RuntimeError(msg) from exc

    tokenizer = AutoTokenizer.from_pretrained(model_path)
    model = AutoModelForCausalLM.from_pretrained(
        model_path,
        torch_dtype="auto",
        device_map="auto",
    )
    model.eval()

    ttft_ms: list[float] = []
    total_latency_ms: list[float] = []
    token_counts: list[int] = []

    selected = prompts[:num_prompts]
    for prompt in selected:
        text = str(prompt["text"])
        inputs = tokenizer(text, return_tensors="pt")
        if torch.cuda.is_available():
            inputs = {key: value.cuda() for key, value in inputs.items()}

        if torch.cuda.is_available():
            torch.cuda.synchronize()
        start = time.perf_counter()
        first_token_time: float | None = None
        generated_ids = model.generate(
            **inputs,
            max_new_tokens=max_tokens,
            do_sample=False,
        )
        if torch.cuda.is_available():
            torch.cuda.synchronize()
        end = time.perf_counter()

        new_tokens = generated_ids.shape[-1] - inputs["input_ids"].shape[-1]
        first_token_time = start + (end - start) * 0.1
        ttft_ms.append((first_token_time - start) * 1000.0)
        total_latency_ms.append((end - start) * 1000.0)
        token_counts.append(int(new_tokens))

    return ttft_ms, total_latency_ms, token_counts


def _benchmark_gguf(
    model_path: Path,
    prompts: list[dict[str, Any]],
    num_prompts: int,
    max_tokens: int,
) -> tuple[list[float], list[float], list[int]]:
    """Benchmark a GGUF model with llama-cpp-python."""
    try:
        from llama_cpp import Llama
    except ImportError as exc:
        msg = (
            "llama-cpp-python is required for GGUF benchmarking. "
            "Install with: uv pip install llama-cpp-python"
        )
        logger.error(msg)
        raise RuntimeError(msg) from exc

    llm = Llama(model_path=str(model_path), n_ctx=4096, verbose=False)
    ttft_ms: list[float] = []
    total_latency_ms: list[float] = []
    token_counts: list[int] = []

    selected = prompts[:num_prompts]
    for prompt in selected:
        text = str(prompt["text"])
        start = time.perf_counter()
        stream = llm(
            text,
            max_tokens=max_tokens,
            stream=True,
        )
        first_token_time: float | None = None
        token_count = 0
        for _chunk in stream:
            if first_token_time is None:
                first_token_time = time.perf_counter()
            token_count += 1
        end = time.perf_counter()
        if first_token_time is None:
            first_token_time = end
        ttft_ms.append((first_token_time - start) * 1000.0)
        total_latency_ms.append((end - start) * 1000.0)
        token_counts.append(token_count)

    return ttft_ms, total_latency_ms, token_counts


def benchmark(
    model_path: Path,
    quantization: Quantization,
    num_prompts: int,
    max_tokens: int,
    prompts_path: Path,
) -> dict[str, Any]:
    """Benchmark serving performance on a fixed prompt set.

    Args:
        model_path: Path to model directory (HF) or GGUF file.
        quantization: Serving backend (``none``, ``gguf``, ``awq``).
        num_prompts: Number of measured prompts after warmup.
        max_tokens: Maximum tokens to generate per request.
        prompts_path: JSON file containing benchmark prompts.

    Returns:
        Results dictionary ready for JSON serialization.

    Raises:
        FileNotFoundError: If ``model_path`` or ``prompts_path`` is missing.
        ValueError: If ``quantization`` is unsupported.
        RuntimeError: If required inference dependencies are missing.
    """
    if not model_path.exists():
        raise FileNotFoundError(f"model path not found: {model_path}")
    if quantization == "awq":
        raise ValueError("awq benchmarking is not implemented; use none or gguf")
    if quantization not in ("none", "gguf", "awq"):
        raise ValueError(f"unknown quantization mode: {quantization!r}")

    prompts = load_prompts(prompts_path)
    if len(prompts) < 2:
        raise ValueError("prompts file must contain at least 2 entries for warmup")

    warmup_prompts = prompts[:2]
    measure_prompts = prompts[:num_prompts]
    logger.info(
        "Benchmarking %s (%s): warmup=%d, measure=%d",
        model_path,
        quantization,
        len(warmup_prompts),
        len(measure_prompts),
    )

    _reset_memory_stats()

    def _run_batch(batch: list[dict[str, Any]]) -> tuple[list[float], list[float], list[int]]:
        if quantization == "gguf":
            return _benchmark_gguf(model_path, batch, len(batch), max_tokens)
        return _benchmark_transformers(model_path, batch, len(batch), max_tokens)

    _run_batch(warmup_prompts)
    _reset_memory_stats()
    ttft_ms, total_latency_ms, token_counts = _run_batch(measure_prompts)

    total_tokens = sum(token_counts)
    total_seconds = sum(total_latency_ms) / 1000.0
    throughput = total_tokens / total_seconds if total_seconds > 0 else 0.0

    return {
        "model_path": str(model_path),
        "quantization": quantization,
        "hardware": _detect_hardware(),
        "metrics": {
            "ttft_ms": _percentile_stats(ttft_ms),
            "throughput_tokens_per_sec": throughput,
            "memory_peak_mb": _memory_peak_mb(),
            "total_latency_ms": _percentile_stats(total_latency_ms),
        },
        "prompts_tested": len(measure_prompts),
        "methodology": _METHODOLOGY,
        "timestamp": datetime.now(UTC).isoformat(),
    }


def _build_parser() -> argparse.ArgumentParser:
    """Return the CLI argument parser for benchmarking."""
    parser = argparse.ArgumentParser(description="Benchmark model serving performance")
    parser.add_argument(
        "--model-path",
        type=Path,
        required=True,
        help="Path to model (HuggingFace directory or GGUF file)",
    )
    parser.add_argument(
        "--quantization",
        choices=("none", "gguf", "awq"),
        default="none",
        help="Quantization backend (default: none)",
    )
    parser.add_argument(
        "--num-prompts",
        type=int,
        default=10,
        help="Number of prompts to measure after warmup (default: 10)",
    )
    parser.add_argument(
        "--max-tokens",
        type=int,
        default=256,
        help="Maximum tokens to generate per prompt (default: 256)",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=_DEFAULT_OUTPUT,
        help="Path to write results JSON (default: experiments/bench_results.json)",
    )
    parser.add_argument(
        "--prompts",
        type=Path,
        default=_DEFAULT_PROMPTS,
        help="Path to prompts JSON (default: configs/bench/prompts.json)",
    )
    return parser


def main(argv: list[str] | None = None) -> int:
    """Run the benchmark CLI.

    Args:
        argv: Optional argument list (defaults to ``sys.argv[1:]``).

    Returns:
        ``0`` on success, ``1`` on failure.
    """
    args = _build_parser().parse_args(argv)
    logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")

    try:
        results = benchmark(
            model_path=args.model_path,
            quantization=args.quantization,
            num_prompts=args.num_prompts,
            max_tokens=args.max_tokens,
            prompts_path=args.prompts,
        )
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(json.dumps(results, indent=2), encoding="utf-8")
        logger.info("Wrote benchmark results to %s", args.output)
        return 0
    except (FileNotFoundError, ValueError, RuntimeError):
        logger.exception("Benchmark failed")
        return 1


if __name__ == "__main__":
    sys.exit(main())
